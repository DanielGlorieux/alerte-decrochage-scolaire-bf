"""
05 - Modèle prédictif, explication et profils de risque (étapes 3 et 4 de la note d'intention)
==============================================================================================
Question posée au modèle : connaissant la situation d'une région l'année scolaire N,
quel sera son taux d'abandon moyen au primaire l'année N+1 ?

  Variable à prédire : taux d'abandon moyen (CP1 à CM1, ensemble) de l'année suivante.
  Variables explicatives : facteurs de risque et indicateurs de l'année N.

Validation (« capacité d'anticipation ») : origine glissante. Pour chaque année cible de
2020-21 à 2024-25, le modèle est entraîné uniquement sur les années antérieures puis comparé
à ce qui s'est réellement produit. Références simples : persistance (l'an prochain = cette
année), moyenne historique de la région, indice composite seul.

Les données sont petites (13 régions x 9 transitions) : scikit-learn sur CPU suffit, un réseau
de neurones (TensorFlow) n'apporterait rien et sur-apprendrait.

Sorties (data/modele/) : validation_modeles.csv, validation_predictions.csv, previsions.csv,
                         shap.csv, profils.csv, modele.joblib, meta.json
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RACINE = Path(__file__).resolve().parents[1]
M = RACINE / "data/modele"
GRAINE = 42

VARIABLES = {
    "abandon_t": "Taux d'abandon de l'année (%)",
    "moyenne_historique": "Taux d'abandon moyen de la région depuis 2015-16 (%)",
    "redoublement_t": "Taux de redoublement (%)",
    "achevement_t": "Taux d'achèvement du primaire (%)",
    "ecart_abandon_f_g": "Écart d'abandon filles - garçons (points)",
    "var_effectif_pct": "Variation annuelle des effectifs (%)",
    "var_ecoles_pct": "Variation annuelle du nombre d'écoles (%)",
    "var_nouveaux_cp1_pct": "Variation annuelle des nouveaux inscrits au CP1 (%)",
    "ratio_eleves_maitre": "Élèves par maître",
    "eleves_par_salle": "Élèves par salle de classe",
    "pct_ecoles_eau": "Écoles avec eau potable (%)",
    "pct_ecoles_latrines": "Écoles avec latrines fonctionnelles (%)",
    "pct_ecoles_cantine": "Écoles avec cantine (%)",
    "pct_ecoles_electricite": "Écoles avec électricité (%)",
    "ecart_parite_tbs": "Écart à la parité du TBS",
    "ecart_parite_tap": "Écart à la parité de l'achèvement",
    "tbs": "Taux brut de scolarisation (%)",
    "tba": "Taux brut d'admission (%)",
    "cep": "Taux de réussite au CEP (%)",
    "part_ecoles_privees": "Part d'écoles privées (%)",
    "tbs_prescolaire": "Taux brut de scolarisation au préscolaire (%)",
}
X_COLS = list(VARIABLES)
CIBLE = "abandon_suivant"
DIMS = ["indice_conditions", "indice_equipements", "indice_ecarts", "indice_acces"]
LIB_DIMS = {"indice_conditions": "conditions d'enseignement", "indice_equipements": "équipements des écoles",
            "indice_ecarts": "écarts filles-garçons", "indice_acces": "accès et continuité de l'école"}


def modeles():
    """Trois familles, entraînées sur log(1 + taux) : la distribution du taux d'abandon est très
    étirée par les années de fermetures d'écoles, le logarithme limite leur poids."""
    base = {
        "Régression (Ridge)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                            RidgeCV(alphas=np.logspace(-1, 3, 20))),
        "Forêt aléatoire": make_pipeline(SimpleImputer(strategy="median"),
                                         RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=GRAINE)),
        "Gradient boosting": make_pipeline(SimpleImputer(strategy="median"),
                                           GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                                     subsample=0.8, random_state=GRAINE)),
    }
    return {k: TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1) for k, m in base.items()}


def jeu_apprentissage(d: pd.DataFrame) -> pd.DataFrame:
    r = d[d.code_region != "BF"].sort_values(["code_region", "annee_fin"]).copy()
    r[CIBLE] = r.groupby("code_region").abandon_t.shift(-1)
    r["annee_cible"] = r.annee_fin + 1
    r["moyenne_historique"] = r.groupby("code_region").abandon_t.transform(lambda s: s.expanding().mean()).round(2)
    return r


def valider(r: pd.DataFrame):
    """Origine glissante : entraînement sur les cibles < année, test sur l'année."""
    app = r.dropna(subset=[CIBLE])
    preds = []
    for an in range(2021, int(app.annee_cible.max()) + 1):
        tr, te = app[app.annee_cible < an], app[app.annee_cible == an]
        p = te[["code_region", "region", "annee_cible", CIBLE]].copy()
        p["Persistance (référence)"] = te.abandon_t.values
        p["Moyenne historique (référence)"] = te.moyenne_historique.values
        lin = LinearRegression().fit(tr[["indice_risque"]], tr[CIBLE])
        p["Indice composite seul"] = lin.predict(te[["indice_risque"]])
        for nom, m in modeles().items():
            p[nom] = m.fit(tr[X_COLS], tr[CIBLE]).predict(te[X_COLS])
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    noms = [c for c in preds.columns if c not in ("code_region", "region", "annee_cible", CIBLE)]
    lignes = []
    for nom in noms:
        e = preds[nom] - preds[CIBLE]
        par_an = preds.groupby("annee_cible")
        rho = par_an.apply(lambda g: g[nom].corr(g[CIBLE], method="spearman")).mean()
        top4 = par_an.apply(lambda g: len(set(g.nlargest(4, nom).region) & set(g.nlargest(4, CIBLE).region)) / 4).mean()
        lignes.append({"modele": nom, "erreur_absolue_moyenne": round(e.abs().mean(), 2),
                       "erreur_absolue_mediane": round(e.abs().median(), 2), "rmse": round(np.sqrt((e ** 2).mean()), 2),
                       "correlation_rang": round(rho, 2), "top4_retrouve": round(top4, 2)})
    return pd.DataFrame(lignes), preds


def main():
    d = pd.read_parquet(M / "indicateurs.parquet")
    r = jeu_apprentissage(d)
    app = r.dropna(subset=[CIBLE])
    print(f"Apprentissage : {len(app)} transitions région-année, {len(X_COLS)} variables")

    # --- Validation temporelle ------------------------------------------------------------
    scores, preds = valider(r)
    print("\nValidation à origine glissante (cibles 2020-21 à 2024-25) :")
    print(scores.to_string(index=False))
    ml = scores[scores.modele.isin(modeles())]
    retenu = ml.sort_values("erreur_absolue_moyenne").iloc[0].modele
    ref = scores[scores.modele == "Persistance (référence)"].iloc[0]
    print(f"\nModèle retenu : {retenu}")
    if ml.erreur_absolue_moyenne.min() >= scores[scores.modele.str.contains("référence")].erreur_absolue_moyenne.min():
        print("ATTENTION : aucun modèle ne fait mieux que la meilleure référence simple en erreur absolue moyenne.")

    # --- Modèle final et prévision pour l'année suivante -----------------------------------
    modele = modeles()[retenu].fit(app[X_COLS], app[CIBLE])
    dern = r[r.annee_fin == r.annee_fin.max()].copy()
    prev = dern[["code_region", "region", "annee_scolaire", "annee_fin", "abandon_t", "abandon_f", "abandon_g",
                 "moyenne_historique", "indice_risque"] + DIMS + ["flux_origine"]].copy()
    prev["risque_prevu"] = modele.predict(dern[X_COLS]).clip(0).round(1)
    # intervalle empirique : quantile 80 % des erreurs absolues du modèle en validation
    err = (preds[retenu] - preds[CIBLE]).abs()
    demi = float(err.quantile(0.8))
    prev["borne_basse"] = (prev.risque_prevu - demi).clip(lower=0).round(1)
    prev["borne_haute"] = (prev.risque_prevu + demi).round(1)
    # confiance par région : erreur moyenne du modèle sur cette région en validation
    err_reg = preds.assign(e=err).groupby("code_region").e.mean()
    seuils_conf = err_reg.quantile([1 / 3, 2 / 3]).values
    prev["erreur_passee"] = prev.code_region.map(err_reg).round(1)
    prev["confiance"] = pd.cut(prev.erreur_passee, [-np.inf, *seuils_conf, np.inf], labels=["élevée", "moyenne", "faible"]).astype(str)
    # niveaux de risque : quartiles historiques du taux d'abandon régional
    q = app[CIBLE].quantile([0.25, 0.5, 0.75]).round(1).tolist()
    prev["niveau_risque"] = pd.cut(prev.risque_prevu, [-np.inf, *q, np.inf],
                                   labels=["faible", "modéré", "élevé", "très élevé"]).astype(str)
    prev["rang_priorite"] = prev.risque_prevu.rank(ascending=False, method="first").astype(int)

    # --- Explication : valeurs de Shapley (méthode agnostique, sur le modèle retenu) --------
    fond = app[X_COLS]
    expl = shap.Explainer(lambda X: modele.predict(pd.DataFrame(X, columns=X_COLS)),
                          shap.maskers.Independent(fond.values, max_samples=len(fond)), seed=GRAINE)
    sv = expl(dern[X_COLS].values, max_evals=2 * len(X_COLS) + 1, silent=True)
    sh = pd.DataFrame(sv.values, columns=X_COLS)
    sh.insert(0, "code_region", dern.code_region.values)
    sh = sh.melt("code_region", var_name="variable", value_name="contribution")
    val = dern.melt("code_region", value_vars=X_COLS, var_name="variable", value_name="valeur")
    sh = sh.merge(val, on=["code_region", "variable"]).merge(dern[["code_region", "region"]], on="code_region")
    sh["libelle"] = sh.variable.map(VARIABLES)
    sh["contribution"] = sh.contribution.round(3)
    base_shap = float(np.mean(sv.base_values))
    importance = sh.groupby("libelle").contribution.apply(lambda s: s.abs().mean()).sort_values(ascending=False)
    print("\nVariables les plus influentes (|contribution| moyenne, en points d'abandon) :")
    print(importance.head(8).round(2).to_string())

    def principaux(code):
        s = sh[(sh.code_region == code) & (sh.contribution > 0)].nlargest(3, "contribution")
        return " ; ".join(s.libelle)
    prev["facteurs_principaux"] = prev.code_region.map(principaux)

    # --- Profils de risque : regroupement des régions selon les 4 dimensions de l'indice ----
    Z = StandardScaler().fit_transform(dern[DIMS])
    meilleurs = max(range(2, 5), key=lambda k: silhouette_score(Z, KMeans(k, n_init=20, random_state=GRAINE).fit_predict(Z)))
    km = KMeans(meilleurs, n_init=20, random_state=GRAINE).fit(Z)
    dern["groupe"] = km.labels_
    centres = dern.groupby("groupe")[DIMS + ["indice_risque"]].mean()
    moy = dern[DIMS].mean()

    def nom_profil(g):
        c = centres.loc[g]
        if c.indice_risque < dern.indice_risque.median() and (c[DIMS] - moy).max() < 8:
            return "Risque contenu"
        return "Risque lié aux " + LIB_DIMS[(c[DIMS] - moy).idxmax()]
    dern["profil"] = dern.groupe.map(nom_profil)
    prev["profil"] = prev.code_region.map(dern.set_index("code_region").profil)
    profils = centres.round(1).assign(profil=[nom_profil(g) for g in centres.index],
                                      regions=[", ".join(dern[dern.groupe == g].region) for g in centres.index])
    print(f"\nProfils ({meilleurs} groupes) :")
    print(profils[["profil", "regions"]].to_string(index=False))

    # --- Sorties ---------------------------------------------------------------------------
    scores.to_csv(M / "validation_modeles.csv", index=False)
    preds.round(2).to_csv(M / "validation_predictions.csv", index=False)
    prev.sort_values("rang_priorite").to_csv(M / "previsions.csv", index=False)
    sh.to_csv(M / "shap.csv", index=False)
    profils.to_csv(M / "profils.csv", index=False)
    joblib.dump(modele, M / "modele.joblib")
    an = int(dern.annee_fin.max())
    json.dump({"modele_retenu": retenu, "variables": VARIABLES, "annee_donnees": f"{an - 1}-{an}",
               "annee_prevue": f"{an}-{an + 1}", "seuils_niveaux": q, "demi_intervalle_80": round(demi, 1),
               "base_shap": round(base_shap, 2), "n_apprentissage": len(app),
               "erreur_modele": float(scores.set_index("modele").loc[retenu, "erreur_absolue_moyenne"]),
               "erreur_persistance": float(ref.erreur_absolue_moyenne),
               "erreur_moyenne_historique": float(scores.set_index("modele").loc["Moyenne historique (référence)", "erreur_absolue_moyenne"]),
               "correlation_rang_modele": float(scores.set_index("modele").loc[retenu, "correlation_rang"]),
               "bat_les_references": bool(scores.set_index("modele").loc[retenu, "erreur_absolue_moyenne"]
                                          < scores[scores.modele.str.contains("référence")].erreur_absolue_moyenne.min())},
              open(M / "meta.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"\nPrévision du taux d'abandon {an}-{an + 1} (intervalle à 80 % : ± {demi:.1f} points) :")
    print(prev.sort_values("rang_priorite")[["rang_priorite", "region", "abandon_t", "risque_prevu", "niveau_risque",
                                             "confiance", "profil"]].to_string(index=False))


if __name__ == "__main__":
    main()
