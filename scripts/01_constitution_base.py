"""
01 - Constitution et nettoyage de la base DCB (préscolaire et primaire, 2015/16 - 2024/25)
==========================================================================================oLogique :
    1. Importation des différentes feuilles (27 tableaux répartis sur 14 feuilles)
    2. Épuration : mise au format long, harmonisation régions / années / sexe / statut
    3. Fusion : panel région x année scolaire (une ligne = une région, une année)
    4. Labélisation : dictionnaire des variables
    5. Audit de qualité : cohérences internes, valeurs manquantes, valeurs suspectes

Entrée  : data/raw/DCB_prescolaire_primaire_2016_2025_nettoye.xlsx 
Sorties : data/clean/panel_region_annee.{parquet,csv}   13 régions + national x 10 années
          data/clean/dictionnaire_variables.csv
          data/clean/audit_qualite.csv

Les anomalies sont signalées dans l'audit.
"""
from pathlib import Path
import re
import unicodedata

import openpyxl
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "data/raw/DCB_prescolaire_primaire_2016_2025_nettoye.xlsx"
SORTIE = RACINE / "data/clean"

# ---------------------------------------------------------------------------------------
# Référentiels d'harmonisation
# ---------------------------------------------------------------------------------------
# Codes ISO 3166-2:BF des 13 régions (découpage en vigueur sur toute la période des données)
REGIONS = {
    "BOUCLE DU MOUHOUN": "BF-01", "CASCADES": "BF-02", "CENTRE": "BF-03",
    "CENTRE-EST": "BF-04", "CENTRE-NORD": "BF-05", "CENTRE-OUEST": "BF-06",
    "CENTRE-SUD": "BF-07", "EST": "BF-08", "HAUTS-BASSINS": "BF-09", "NORD": "BF-10",
    "PLATEAU CENTRAL": "BF-11", "SAHEL": "BF-12", "SUD-OUEST": "BF-13", "NATIONAL": "BF",
}
ALIAS_REGION = {"HAUT-BASSINS": "HAUTS-BASSINS", "BURKINA FASO": "NATIONAL",
                "TOTAL GENERAL": "NATIONAL", "PLATEAU-CENTRAL": "PLATEAU CENTRAL"}
SEXE = {"G": "G", "GARCONS": "G", "F": "F", "FILLES": "F", "T": "T", "TOTAL": "T"}
STATUT = {"PUBLIC": "public", "PRIVE": "prive", "COMMUNAUTAIRE": "communautaire",
          "TOTAL": "total", "TOTAL GENERAL": "total", "NATIONAL": "total"}
MILIEU = {"URBAIN": "urbain", "RURAL": "rural", "INDETERMINE": "indetermine",
          "TOTAL GENERAL": "total", "NATIONAL": "total"}

# (cycle, n° de tableau) -> (indicateur, unité, dimension réelle de la 1re colonne)
TABLEAUX = {
    ("prescolaire", 1): ("effectif", "eleves", "region"),
    ("prescolaire", 2): ("effectif", "eleves", "statut"),
    ("prescolaire", 3): ("effectif", "eleves", "milieu"),   # en-tête "REGION" erroné dans la source
    ("prescolaire", 4): ("effectif", "eleves", "statut"),
    ("prescolaire", 5): ("tbs", "pourcent", "region"),
    ("prescolaire", 6): ("ips_tbs", "ratio_f_g", "region"),
    ("prescolaire", 7): ("structures", "nombre", "region"),
    ("prescolaire", 8): ("salles", "nombre", "region"),
    ("primaire", 1): ("effectif", "eleves", "region"),
    ("primaire", 2): ("effectif", "eleves", "statut"),
    ("primaire", 3): ("effectif", "eleves", "milieu"),
    ("primaire", 4): ("effectif", "eleves", "statut"),
    ("primaire", 5): ("nouveaux_cp1", "eleves", "region"),
    ("primaire", 6): ("tba", "pourcent", "region"),
    ("primaire", 7): ("tbs", "pourcent", "region"),
    ("primaire", 8): ("tap", "pourcent", "region"),
    ("primaire", 9): ("cep", "pourcent", "region"),
    ("primaire", 10): ("ips_tba", "ratio_f_g", "region"),
    ("primaire", 11): ("ips_tbs", "ratio_f_g", "region"),
    ("primaire", 12): ("ips_tap", "ratio_f_g", "region"),
    ("primaire", 13): ("ecoles", "nombre", "region"),
}
LIBELLES = {
    "effectif": "Effectif d'élèves", "nouveaux_cp1": "Nouveaux inscrits au CP1",
    "tba": "Taux brut d'admission (%)", "tbs": "Taux brut de scolarisation (%)",
    "tap": "Taux d'achèvement du primaire (%)", "cep": "Taux de réussite au CEP (%)",
    "ips_tba": "Indice de parité des sexes du TBA (F/G)",
    "ips_tbs": "Indice de parité des sexes du TBS (F/G)",
    "ips_tap": "Indice de parité des sexes du TAP (F/G)",
    "ecoles": "Nombre d'écoles primaires", "structures": "Nombre de structures préscolaires",
    "salles": "Nombre de salles de classe du préscolaire",
}


def norm(x) -> str:
    """Majuscules sans accents ni espaces superflus (clé de comparaison des libellés)."""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().upper()


def annee_fin(x):
    """'2015-2016', '2015/16', '2015-16', '2024/25', 2016 -> année civile de fin (2016)."""
    s = str(x).strip()
    m = re.fullmatch(r"(20\d\d)\s*[-/]\s*(\d\d|20\d\d)", s)
    if m:
        return int(m.group(1)) + 1
    m = re.fullmatch(r"(20\d\d)(\.0)?", s)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------
# 1. Importation : découpage des feuilles en tableaux et passage au format long
# ---------------------------------------------------------------------------------------
def lire_tableaux(chemin: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(chemin, data_only=True)  # data_only : valeurs des formules (IPS)
    lignes = []
    for ws in wb:
        cycle = "prescolaire" if ws.title == "PRESCOLAIRE" else "primaire"
        grille = [[c.value for c in row] for row in ws.iter_rows()]
        titres = [i for i, r in enumerate(grille) if isinstance(r[0], str) and r[0].startswith("Tableau")]
        for k, i0 in enumerate(titres):
            fin = titres[k + 1] if k + 1 < len(titres) else len(grille)
            num = int(re.match(r"Tableau\s*(\d+)", grille[i0][0]).group(1))
            entete = grille[i0 + 1]
            deux_entetes = any(norm(v) in ("G", "F", "T") for v in grille[i0 + 2] if v is not None)
            sous = grille[i0 + 2] if deux_entetes else [None] * len(entete)
            # colonnes d'identification = libellés non annuels de la ligne d'en-tête
            col_id = [j for j, v in enumerate(entete) if v is not None and annee_fin(v) is None]
            col_val, an_courant = [], None
            for j, v in enumerate(entete):
                if j in col_id:
                    continue
                if v is not None or not deux_entetes:
                    an_courant = annee_fin(v) if v is not None else None
                # sinon : cellule d'année fusionnée sur G/F/T, on garde l'année courante
                if an_courant is not None and (not deux_entetes or sous[j] is not None):
                    col_val.append((j, an_courant, norm(sous[j]) if deux_entetes else None))
            ids = {j: None for j in col_id}
            for i in range(i0 + (3 if deux_entetes else 2), fin):
                r = grille[i]
                if all(v is None for v in r):
                    continue
                # report vers le bas des cellules fusionnées ; une nouvelle valeur à gauche
                # réinitialise les colonnes situées à sa droite (lignes de totaux)
                for pos, j in enumerate(col_id):
                    if r[j] is not None:
                        ids[j] = r[j]
                        for j2 in col_id[pos + 1:]:
                            ids[j2] = None
                for j, an, sx in col_val:
                    lignes.append({
                        "cycle": cycle, "tableau": num, "feuille": ws.title,
                        "cellule": f"{openpyxl.utils.get_column_letter(j + 1)}{i + 1}",
                        **{norm(entete[c]): ids[c] for c in col_id},
                        "sexe_entete": sx, "annee_fin": an, "valeur": r[j],
                    })
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------------------
# 2. Épuration : harmonisation des modalités
# ---------------------------------------------------------------------------------------
def epurer(brut: pd.DataFrame) -> pd.DataFrame:
    d = brut.copy()
    d["col1"] = d["REGION"].fillna(d.get("REGIONS")).fillna(d["STATUT"].where(d["REGION"].isna()))
    d["col1"] = d["col1"].fillna(d.get("MILIEU"))
    meta = d[["cycle", "tableau"]].apply(lambda r: TABLEAUX[(r.cycle, r.tableau)], axis=1, result_type="expand")
    d[["indicateur", "unite", "dim1"]] = meta

    def modalites(r):
        c1 = norm(r.col1)
        sous_total = c1.startswith("TOTAL ") and c1 != "TOTAL GENERAL"
        c1 = c1[6:] if sous_total else c1
        region, statut, milieu, niveau = "NATIONAL", "total", "total", "total"
        if r.dim1 == "region":
            region = ALIAS_REGION.get(c1, c1)
            if "STATUT" in r.index and pd.notna(r.STATUT) and pd.notna(r.REGION):
                statut = STATUT[norm(r.STATUT)]
        elif r.dim1 == "statut":
            statut = STATUT[c1]
            if pd.notna(r.get("NIVEAU")):
                niveau = norm(r.NIVEAU).replace(" ", "_").lower()
        else:
            milieu = MILIEU[c1]
        if pd.notna(r.sexe_entete):
            sexe = SEXE[r.sexe_entete]
        elif pd.notna(r.get("SEXE")):
            sexe = SEXE[norm(r.SEXE)]
        else:  # tableaux sans ventilation par sexe (IPS, écoles, salles)
            sexe = "T"
        return pd.Series([region, statut, milieu, niveau, sexe])

    d[["region", "statut", "milieu", "niveau", "sexe"]] = d.apply(modalites, axis=1)
    inconnues = sorted(set(d.region) - set(REGIONS))
    assert not inconnues, f"Régions non reconnues : {inconnues}"
    d["code_region"] = d.region.map(REGIONS)
    d["annee_scolaire"] = (d.annee_fin - 1).astype(str) + "-" + d.annee_fin.astype(str)
    d["valeur"] = pd.to_numeric(d.valeur, errors="coerce")
    d["libelle"] = d.indicateur.map(LIBELLES)
    cols = ["cycle", "indicateur", "libelle", "unite", "code_region", "region", "statut", "milieu",
            "niveau", "sexe", "annee_scolaire", "annee_fin", "valeur", "tableau", "feuille", "cellule"]
    return d[cols].sort_values(cols[:12]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# 3. Fusion : panel région x année
# ---------------------------------------------------------------------------------------
def construire_panel(long: pd.DataFrame) -> pd.DataFrame:
    r = long[long.tableau.map(lambda t: True) & (long.milieu == "total") & (long.niveau == "total")]
    r = r[r.apply(lambda x: TABLEAUX[(x.cycle, x.tableau)][2] == "region", axis=1)].copy()
    pref = r.cycle.map({"primaire": "prim", "prescolaire": "presco"})
    par_statut = r.indicateur.isin(["ecoles", "structures", "salles"])
    par_sexe = ~par_statut & ~r.indicateur.str.startswith("ips_")
    r["variable"] = pref + "_" + r.indicateur
    r.loc[par_statut, "variable"] += "_" + r.statut
    r.loc[par_sexe, "variable"] += "_" + r.sexe.str.lower()
    cle = ["code_region", "region", "annee_scolaire", "annee_fin"]
    doublons = r.duplicated(cle + ["variable"]).sum()
    assert doublons == 0, f"{doublons} doublons sur la clé région x année x variable"
    panel = r.pivot(index=cle, columns="variable", values="valeur").reset_index()
    panel.columns.name = None
    return panel.sort_values(["code_region", "annee_fin"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# 4. Labélisation : dictionnaire des variables du panel
# ---------------------------------------------------------------------------------------
def dictionnaire(panel: pd.DataFrame) -> pd.DataFrame:
    fixes = {"code_region": "Code ISO 3166-2 de la région (BF = national)",
             "region": "Région (découpage en 13 régions)",
             "annee_scolaire": "Année scolaire (ex. 2015-2016)",
             "annee_fin": "Année civile de fin d'année scolaire (session du CEP)"}
    suff = {"g": "garçons", "f": "filles", "t": "ensemble", "public": "public", "prive": "privé",
            "communautaire": "communautaire", "total": "tous statuts"}
    lignes = []
    for c in panel.columns:
        if c in fixes:
            lignes.append((c, fixes[c], "", "DCB"))
            continue
        cyc, reste = c.split("_", 1)
        ind = next(k for k in sorted(LIBELLES, key=len, reverse=True) if reste.startswith(k))
        s = reste[len(ind):].strip("_")
        lib = LIBELLES[ind] + (f" - {suff[s]}" if s else "")
        lib += " (primaire)" if cyc == "prim" else " (préscolaire)"
        lignes.append((c, lib, f"{panel[c].notna().mean():.0%}", "DCB"))
    return pd.DataFrame(lignes, columns=["variable", "libelle", "taux_remplissage", "source"])


# ---------------------------------------------------------------------------------------
# 5. Audit de qualité
# ---------------------------------------------------------------------------------------
def auditer(long: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    A = []

    def note(controle, gravite, cle, detail):
        A.append({"controle": controle, "gravite": gravite, "cle": cle, "detail": detail})

    cle = ["cycle", "tableau", "indicateur", "region", "statut", "milieu", "niveau", "annee_fin"]
    comptes = long[long.unite.isin(["eleves", "nombre"])]

    # a. Total = Garçons + Filles
    w = comptes[comptes.unite == "eleves"].pivot_table(index=cle, columns="sexe", values="valeur").dropna()
    for k, v in w[(w.G + w.F - w["T"]).abs() > 0].iterrows():
        note("T = G + F", "erreur", k, f"G+F={v.G + v.F:.0f} ; T={v['T']:.0f}")

    # b. Somme des régions = national
    reg = comptes[comptes.apply(lambda x: TABLEAUX[(x.cycle, x.tableau)][2] == "region", axis=1)]
    k2 = ["cycle", "tableau", "indicateur", "statut", "sexe", "annee_fin"]
    s = reg[reg.region != "NATIONAL"].groupby(k2).valeur.sum()
    n = reg[reg.region == "NATIONAL"].set_index(k2).valeur
    for k, ecart in (s - n).dropna().items():
        if abs(ecart) > 0:
            note("somme régions = national", "erreur", k, f"somme={s[k]:.0f} ; national={n[k]:.0f} ; écart={ecart:+.0f}")

    # c. Somme des statuts = total (par région)
    st = comptes[comptes.indicateur.isin(["ecoles", "structures", "salles"])]
    w = st.pivot_table(index=["cycle", "indicateur", "region", "annee_fin"], columns="statut", values="valeur")
    # min_count=1 : pas de contrôle quand le détail par statut n'est pas publié (ligne nationale)
    somme = w.drop(columns="total").sum(axis=1, min_count=1)
    for k, ecart in (somme - w["total"]).dropna().items():
        if abs(ecart) > 0:
            note("somme statuts = total", "erreur", k, f"somme={somme[k]:.0f} ; total={w.loc[k, 'total']:.0f}")

    # d. Cohérence du total national entre tableaux d'effectifs (région / statut / milieu / niveau)
    nat = long[(long.indicateur == "effectif") & (long.region == "NATIONAL") & (long.statut == "total")
               & (long.milieu == "total") & (long.niveau == "total")]
    g = nat.groupby(["cycle", "sexe", "annee_fin"]).valeur.agg(["min", "max", "nunique"])
    for k, v in g[g["nunique"] > 1].iterrows():
        note("total national identique entre tableaux", "erreur", k, f"min={v['min']:.0f} ; max={v['max']:.0f}")

    # e. IPS = taux filles / taux garçons
    for cyc, ips, taux in [("prescolaire", "ips_tbs", "tbs"), ("primaire", "ips_tba", "tba"),
                           ("primaire", "ips_tbs", "tbs"), ("primaire", "ips_tap", "tap")]:
        p = "presco" if cyc == "prescolaire" else "prim"
        calc = panel[f"{p}_{taux}_f"] / panel[f"{p}_{taux}_g"]
        for i in panel.index[(calc - panel[f"{p}_{ips}"]).abs() > 0.015]:
            note("IPS = taux F / taux G", "avertissement", (cyc, ips, panel.region[i], panel.annee_fin[i]),
                 f"publié={panel.loc[i, f'{p}_{ips}']:.3f} ; recalculé={calc[i]:.3f}")

    # f. Valeurs manquantes dans la grille région x année
    for c in panel.columns[4:]:
        for i in panel.index[panel[c].isna()]:
            note("valeur manquante", "info", (c, panel.region[i], panel.annee_fin[i]), "")

    # g. Valeurs suspectes
    taux = long[(long.unite == "pourcent")]
    w = taux.pivot_table(index=["cycle", "indicateur", "region", "annee_fin"], columns="sexe", values="valeur").dropna()
    for k, v in w[(w.G == w.F) & (w.F == w["T"])].iterrows():
        note("taux G = F = T strictement identiques", "avertissement", k, f"valeur={v['T']}")
    for k, v in w[~((w["T"] >= w[["G", "F"]].min(axis=1) - 0.05) & (w["T"] <= w[["G", "F"]].max(axis=1) + 0.05))].iterrows():
        note("taux T hors de l'intervalle [G, F]", "avertissement", k, f"G={v.G} F={v.F} T={v['T']}")
    for k, v in taux[(taux.indicateur.isin(["tap", "cep"])) & (taux.valeur > 100)].iterrows():
        note("taux > 100 %", "avertissement", (v.cycle, v.indicateur, v.region, v.sexe, v.annee_fin), f"valeur={v.valeur}")
    ser = comptes[(comptes.sexe == "T")].sort_values("annee_fin")
    k3 = ["cycle", "tableau", "indicateur", "region", "statut", "milieu", "niveau"]
    var = ser.groupby(k3).valeur.pct_change()
    prec = ser.groupby(k3).valeur.shift()
    gros = ser[(var.abs() > 0.5) & (prec.fillna(0) >= 20)]
    for i, v in gros.iterrows():
        note("variation annuelle > 50 %", "avertissement",
             (v.cycle, v.indicateur, v.region, v.statut, v.milieu, v.niveau, v.annee_fin),
             f"{prec[i]:.0f} -> {v.valeur:.0f} ({var[i]:+.0%}) [{v.feuille}!{v.cellule}]")
    return pd.DataFrame(A)


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    brut = lire_tableaux(SOURCE)
    long = epurer(brut)
    panel = construire_panel(long)
    dico = dictionnaire(panel)
    audit = auditer(long, panel)

    long.to_parquet(SORTIE / "dcb_long.parquet", index=False)
    long.to_csv(SORTIE / "dcb_long.csv", index=False)
    panel.to_parquet(SORTIE / "panel_region_annee.parquet", index=False)
    panel.to_csv(SORTIE / "panel_region_annee.csv", index=False)
    dico.to_csv(SORTIE / "dictionnaire_variables.csv", index=False)
    audit.to_csv(SORTIE / "audit_qualite.csv", index=False)

    print(f"Format long : {len(long)} valeurs, {long.groupby(['cycle', 'tableau']).ngroups} tableaux")
    print(f"Panel       : {panel.shape[0]} lignes x {panel.shape[1]} colonnes "
          f"({panel.region.nunique()} zones x {panel.annee_fin.nunique()} années)")
    print("\nAudit (nombre de signalements par contrôle) :")
    print(audit.groupby(["gravite", "controle"]).size().to_string())


if __name__ == "__main__":
    main()
