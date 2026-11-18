"""
06 - Modèle d'abandon structurel (Insighters) : remplace le modèle prédictif du script 05
=========================================================================================
À lancer après le script 05, dont il garde l'indice composite, ses quatre dimensions et les profils de
risque. Il réécrit les sorties du modèle dans data/modele/, aux formats lus par la plateforme.

Question posée au modèle : connaissant la situation d'une région l'année N, quel sera son taux d'abandon
moyen au primaire (CP1 à CM1, ensemble) l'année N+1, EN L'ABSENCE DE CHOC ?

  - Le taux d'abandon publié pour l'année N mesure les élèves de N-1 qui ne sont pas revenus en N. Lors d'un
    choc, il mesure surtout des fermetures d'écoles (Sahel 56 % en 2018-19), puis s'effondre l'année des
    retours (Sahel 2 % en 2023-24). Ces années sont imprévisibles à partir des données annuelles.
  - Le modèle apprend donc sur les transitions sans choc : effectifs stables à ± 10 % et nombre d'écoles en
    baisse de moins de 10 % entre N et N+1. Toutes les variables viennent de l'année N.
  - Variables écartées : la part d'écoles avec cantine (extraction douteuse pour 2015-16 et 2024-25) et les
    enseignants (non publiés pour 2024-25).

Validation à origine glissante, années cibles 2020-21 à 2024-25 : entraînement sur les seules années
antérieures. Les performances affichées portent sur les transitions sans choc ; la confiance par région
tient compte de toutes les années testées, chocs compris. Modèle retenu : forêt aléatoire, qui classe les
régions nettement mieux que le gradient boosting pour une précision voisine.

Sorties (data/modele/) : modele.joblib, meta.json, previsions.csv, shap.csv, validation_modeles.csv,
                         validation_predictions.csv, variables_modele.csv (variables de la dernière année,
                         utilisées par le simulateur)
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = Path(__file__).resolve().parents[1]
PANEL = R / "data/clean/panel_enrichi.csv"
M = R / "data/modele"
RETENU = "Forêt aléatoire"
SEUIL_CHOC = 0.10
ANNEES_TEST = range(2021, 2026)
GRAINE = 0

VARIABLES = {
    "prim_abandon_moy_t": "Taux d'abandon de l'année (%)",
    "abandon_moy_passe": "Médiane des abandons passés de la région (%)",
    "prim_redoublement_moy_t": "Taux de redoublement (%)",
    "ecart_abandon_gf": "Écart d'abandon garçons - filles (points)",
    "prim_ratio_eleves_maitre_total": "Élèves par maître",
    "prim_eleves_par_salle": "Élèves par salle de classe",
    "prim_pct_ecoles_electricite": "Écoles avec électricité (%)",
    "prim_pct_ecoles_latrines": "Écoles avec latrines fonctionnelles (%)",
    "prim_pct_ecoles_eau_potable": "Écoles avec eau potable (%)",
    "presco_tbs_t": "Scolarisation au préscolaire (%)",
    "prim_tba_t": "Taux brut d'admission au CP1 (%)",
    "prim_tap_t": "Taux d'achèvement du primaire (%)",
    "prim_cep_t": "Taux de réussite au CEP (%)",
    "var_eff": "Variation annuelle des effectifs (fraction)",
    "var_ecoles": "Variation annuelle du nombre d'écoles (fraction)",
    "perte_ecoles_pic": "Écoles perdues depuis le plus haut niveau (fraction)",
    "part_prive": "Part des écoles privées (fraction)",
}
X_COLS = list(VARIABLES)
COLONNES_MODELE = ["risque_prevu", "borne_basse", "borne_haute", "erreur_passee", "confiance", "niveau_risque",
                   "rang_priorite", "facteurs_principaux"]


def preparer(panel):
    """Variables de l'année N et cible (abandon de N+1) ; la ligne nationale est exclue."""
    r = panel[panel.code_region != "BF"].sort_values(["code_region", "annee_fin"]).copy()
    g = r.groupby("code_region")
    r["var_eff"] = r.prim_effectif_t / g.prim_effectif_t.shift() - 1
    r["var_ecoles"] = r.prim_ecoles_total / g.prim_ecoles_total.shift() - 1
    r["perte_ecoles_pic"] = 1 - r.prim_ecoles_total / g.prim_ecoles_total.cummax()
    r["part_prive"] = r.prim_ecoles_prive / r.prim_ecoles_total
    r["ecart_abandon_gf"] = r.prim_abandon_moy_g - r.prim_abandon_moy_f
    r["abandon_moy_passe"] = g.prim_abandon_moy_t.transform(lambda s: s.expanding().median())
    r["moyenne_historique"] = g.prim_abandon_moy_t.transform(lambda s: s.expanding().mean())
    r["cible"] = g.prim_abandon_moy_t.shift(-1)
    r["annee_cible"] = r.annee_fin + 1
    r["choc_cible"] = (g.var_eff.shift(-1).abs() > SEUIL_CHOC) | (g.var_ecoles.shift(-1) < -SEUIL_CHOC)
    return r.dropna(subset=X_COLS).reset_index(drop=True)


class Reference:
    """Prévision sans apprentissage : la valeur d'une colonne de l'année N."""

    def __init__(self, colonne):
        self.colonne = colonne

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.colonne].to_numpy()


def modeles():
    return {
        "Persistance (référence)": Reference("prim_abandon_moy_t"),
        "Moyenne historique (référence)": Reference("moyenne_historique"),
        "Médiane passée de la région (référence)": Reference("abandon_moy_passe"),
        "Régression (Ridge)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Forêt aléatoire": RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=4, random_state=GRAINE),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                                       min_samples_leaf=4, subsample=0.8, random_state=GRAINE),
    }


def valider(d):
    """Origine glissante : apprentissage sur les transitions sans choc antérieures, test sur tous les cas."""
    connus = d[d.cible.notna()]
    lignes = []
    for an in ANNEES_TEST:
        app = connus[(connus.annee_cible < an) & ~connus.choc_cible]
        te = connus[connus.annee_cible == an]
        p = te[["code_region", "region", "annee_cible", "cible", "choc_cible"]].rename(columns={"cible": "abandon_suivant"})
        X = te[X_COLS + ["moyenne_historique"]]
        for nom, m in modeles().items():
            cols = X_COLS + (["moyenne_historique"] if isinstance(m, Reference) else [])
            p[nom] = m.fit(app[cols], app.cible).predict(X[cols])
        lignes.append(p)
    return pd.concat(lignes, ignore_index=True)


def scores(preds):
    lignes = []
    for nom in modeles():
        e = preds[nom] - preds.abandon_suivant
        par_an = preds.groupby("annee_cible")
        rho = par_an.apply(lambda a: spearmanr(a[nom], a.abandon_suivant).statistic, include_groups=False)
        top4 = par_an.apply(lambda a: len(set(a.nlargest(4, nom).code_region) & set(a.nlargest(4, "abandon_suivant").code_region)) / 4
                            if len(a) >= 5 else np.nan, include_groups=False)
        lignes.append({"modele": nom, "erreur_absolue_moyenne": round(e.abs().mean(), 2),
                       "erreur_absolue_mediane": round(e.abs().median(), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
                       "correlation_rang": round(rho.mean(), 2), "top4_retrouve": round(top4.mean(), 2)})
    return pd.DataFrame(lignes).sort_values("erreur_absolue_moyenne")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    d = preparer(pd.read_csv(PANEL))
    app = d[d.cible.notna() & ~d.choc_cible]
    print(f"Apprentissage : {len(app)} transitions sans choc, {len(X_COLS)} variables")

    # --- Validation temporelle ------------------------------------------------------------
    preds = valider(d)
    calme, chocs = preds[~preds.choc_cible], preds[preds.choc_cible]
    sc = scores(calme)
    print("\nValidation à origine glissante, transitions sans choc (cibles 2020-21 à 2024-25) :")
    print(sc.to_string(index=False))
    erreur_chocs = float((chocs[RETENU] - chocs.abandon_suivant).abs().mean())
    print(f"Erreur du modèle sur les années de choc ou de retour : {erreur_chocs:.2f} points ({len(chocs)} cas)")

    # --- Modèle final et prévision pour l'année suivante -----------------------------------
    modele = modeles()[RETENU].fit(app[X_COLS], app.cible)
    dern = d[d.annee_fin == d.annee_fin.max()].copy()
    demi = float((calme[RETENU] - calme.abandon_suivant).abs().quantile(0.8))
    nouv = dern[["code_region"]].copy()
    nouv["risque_prevu"] = modele.predict(dern[X_COLS]).clip(0).round(1)
    nouv["borne_basse"] = (nouv.risque_prevu - demi).clip(lower=0).round(1)
    nouv["borne_haute"] = (nouv.risque_prevu + demi).round(1)
    # confiance par région : erreur moyenne du modèle sur la région, toutes années testées (chocs compris)
    err_reg = preds.assign(e=(preds[RETENU] - preds.abandon_suivant).abs()).groupby("code_region").e.mean()
    seuils_conf = err_reg.quantile([1 / 3, 2 / 3]).values
    nouv["erreur_passee"] = nouv.code_region.map(err_reg).round(1)
    nouv["confiance"] = pd.cut(nouv.erreur_passee, [-np.inf, *seuils_conf, np.inf],
                               labels=["élevée", "moyenne", "faible"]).astype(str)
    # niveaux de risque : quartiles des taux prévus par le modèle sur les années testées. Une forêt aléatoire
    # moyenne de nombreux arbres : ses prévisions sont resserrées et n'atteignent jamais les extrêmes observés.
    # Des seuils pris sur les taux observés ne classeraient donc presque aucune région au niveau le plus haut ;
    # ceux-ci restent fixes d'une année à l'autre et à la même échelle que les prévisions.
    q = calme[RETENU].quantile([0.25, 0.5, 0.75]).round(1).tolist()
    nouv["niveau_risque"] = pd.cut(nouv.risque_prevu, [-np.inf, *q, np.inf],
                                   labels=["faible", "modéré", "élevé", "très élevé"]).astype(str)
    nouv["rang_priorite"] = nouv.risque_prevu.rank(ascending=False, method="first").astype(int)

    # --- Explication : valeurs de Shapley exactes pour une forêt (TreeExplainer) ---------------
    expl = shap.TreeExplainer(modele)
    sv = expl.shap_values(dern[X_COLS])
    base_shap = float(np.ravel(expl.expected_value)[0])
    sh = pd.DataFrame(sv, columns=X_COLS).assign(code_region=dern.code_region.values)
    sh = sh.melt("code_region", var_name="variable", value_name="contribution")
    valeurs = dern.melt("code_region", value_vars=X_COLS, var_name="variable", value_name="valeur")
    sh = sh.merge(valeurs, on=["code_region", "variable"]).merge(dern[["code_region", "region"]], on="code_region")
    sh["libelle"] = sh.variable.map(VARIABLES)
    sh["contribution"] = sh.contribution.round(3)

    def principaux(code):
        s = sh[(sh.code_region == code) & (sh.contribution > 0)].nlargest(3, "contribution")
        return " ; ".join(s.libelle)
    nouv["facteurs_principaux"] = nouv.code_region.map(principaux)

    # --- Prévisions : colonnes du modèle remplacées, indice et profils du script 05 conservés --
    ancien = pd.read_csv(M / "previsions.csv")
    ordre = list(ancien.columns)
    prev = ancien.drop(columns=COLONNES_MODELE).merge(nouv, on="code_region")[ordre]

    # --- Sorties ---------------------------------------------------------------------------
    sc.to_csv(M / "validation_modeles.csv", index=False)
    calme.drop(columns="choc_cible").round(2).to_csv(M / "validation_predictions.csv", index=False)
    prev.sort_values("rang_priorite").to_csv(M / "previsions.csv", index=False)
    sh.to_csv(M / "shap.csv", index=False)
    dern[["code_region", "region"] + X_COLS].to_csv(M / "variables_modele.csv", index=False)
    joblib.dump(modele, M / "modele.joblib")
    an = int(dern.annee_fin.max())
    ind = sc.set_index("modele")
    json.dump({
        "modele_retenu": RETENU, "variables": VARIABLES, "annee_donnees": f"{an - 1}-{an}",
        "annee_prevue": f"{an}-{an + 1}", "seuils_niveaux": q, "demi_intervalle_80": round(demi, 1),
        "base_shap": round(base_shap, 2), "n_apprentissage": len(app),
        "erreur_modele": float(ind.loc[RETENU, "erreur_absolue_moyenne"]),
        "erreur_persistance": float(ind.loc["Persistance (référence)", "erreur_absolue_moyenne"]),
        "erreur_moyenne_historique": float(ind.loc["Moyenne historique (référence)", "erreur_absolue_moyenne"]),
        "correlation_rang_modele": float(ind.loc[RETENU, "correlation_rang"]),
        "erreur_modele_chocs": round(erreur_chocs, 2), "n_cas_chocs": len(chocs), "n_cas_valides": len(calme),
        "cible": "taux d'abandon moyen de l'année suivante, en l'absence de choc",
        "seuil_choc": SEUIL_CHOC, "script": "scripts/06_modele_insighters.py",
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }, open(M / "meta.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"\nPrévision du taux d'abandon {an}-{an + 1}, en l'absence de choc (intervalle à 80 % : ± {demi:.1f} points) :")
    print(prev.sort_values("rang_priorite")[["rang_priorite", "region", "abandon_t", "risque_prevu", "niveau_risque",
                                             "confiance", "profil"]].to_string(index=False))


if __name__ == "__main__":
    main()
