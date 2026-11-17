"""
04 - Indicateurs de décrochage et indice composite de risque (étapes 1 et 2 de la note d'intention)
===================================================================================================
Entrée  : data/clean/panel_enrichi.parquet
Sortie  : data/modele/indicateurs.parquet (+ .csv)   une ligne par région et par année scolaire
          data/modele/indice_definition.csv           composition de l'indice

Étape 1 - indicateurs de décrochage (ce que l'on mesure)
    abandon, redoublement, promotion (moyenne des niveaux, par sexe), achèvement (TAP),
    évolution des effectifs et du nombre d'écoles d'une année sur l'autre.
Étape 2 - indice composite (0 = risque faible, 100 = risque élevé)
    Il ne contient QUE des facteurs de risque, jamais les indicateurs de décrochage eux-mêmes,
    pour pouvoir être comparé honnêtement au modèle. Chaque facteur est ramené entre 0 et 1
    (rang centile sur l'ensemble des régions et des années), orienté pour que 1 = défavorable,
    puis moyenné par dimension ; l'indice est la moyenne simple des quatre dimensions.
"""
from pathlib import Path
import sys

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
ENTREE = RACINE / "data/clean/panel_enrichi.parquet"
SORTIE = RACINE / "data/modele"

# facteur -> (dimension, sens : +1 si une valeur élevée augmente le risque, -1 sinon, libellé)
FACTEURS = {
    "ratio_eleves_maitre": ("Conditions d'enseignement", +1, "Élèves par maître"),
    "eleves_par_salle": ("Conditions d'enseignement", +1, "Élèves par salle de classe"),
    "pct_ecoles_eau": ("Équipements", -1, "Écoles avec eau potable (%)"),
    "pct_ecoles_latrines": ("Équipements", -1, "Écoles avec latrines fonctionnelles (%)"),
    "pct_ecoles_cantine": ("Équipements", -1, "Écoles avec cantine (%)"),
    "pct_ecoles_electricite": ("Équipements", -1, "Écoles avec électricité (%)"),
    "ecart_parite_tbs": ("Écarts filles-garçons", +1, "Écart à la parité du TBS (|1 - IPS|)"),
    "ecart_parite_tap": ("Écarts filles-garçons", +1, "Écart à la parité de l'achèvement (|1 - IPS|)"),
    "tbs": ("Accès et continuité", -1, "Taux brut de scolarisation (%)"),
    "tba": ("Accès et continuité", -1, "Taux brut d'admission (%)"),
    "var_ecoles_pct": ("Accès et continuité", -1, "Variation annuelle du nombre d'écoles (%)"),
}


def construire(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["code_region", "annee_fin"]).copy()
    d = p[["code_region", "region", "annee_scolaire", "annee_fin"]].copy()

    # --- Étape 1 : indicateurs de décrochage -------------------------------------------
    for s in "gft":
        d[f"abandon_{s}"] = p[f"prim_abandon_moy_{s}"]
        d[f"redoublement_{s}"] = p[f"prim_redoublement_moy_{s}"]
        d[f"promotion_{s}"] = p[f"prim_promotion_moy_{s}"]
        d[f"achevement_{s}"] = p[f"prim_tap_{s}"]
    d["flux_origine"] = p.prim_flux_origine
    g = p.groupby("code_region")
    d["var_effectif_pct"] = (100 * g.prim_effectif_t.pct_change()).round(1)
    d["var_ecoles_pct"] = (100 * g.prim_ecoles_total.pct_change()).round(1)
    d["var_nouveaux_cp1_pct"] = (100 * g.prim_nouveaux_cp1_t.pct_change()).round(1)
    d["ecart_abandon_f_g"] = (d.abandon_f - d.abandon_g).round(2)

    # --- Facteurs de risque -------------------------------------------------------------
    d["ratio_eleves_maitre"] = p.prim_ratio_eleves_maitre_total
    d["eleves_par_salle"] = p.prim_eleves_par_salle
    d["pct_ecoles_eau"] = p.prim_pct_ecoles_eau_potable
    d["pct_ecoles_latrines"] = p.prim_pct_ecoles_latrines
    d["pct_ecoles_cantine"] = p.prim_pct_ecoles_cantine
    d["pct_ecoles_electricite"] = p.prim_pct_ecoles_electricite
    d["ecart_parite_tbs"] = (1 - p.prim_ips_tbs).abs().round(3)
    d["ecart_parite_tap"] = (1 - p.prim_ips_tap).abs().round(3)
    d["tbs"] = p.prim_tbs_t
    d["tba"] = p.prim_tba_t
    d["cep"] = p.prim_cep_t
    d["part_ecoles_privees"] = (100 * p.prim_ecoles_prive / p.prim_ecoles_total).round(1)
    d["tbs_prescolaire"] = p.presco_tbs_t
    d["effectif"] = p.prim_effectif_t
    d["ecoles"] = p.prim_ecoles_total

    # --- Étape 2 : indice composite (régions seulement, national exclu du classement) ----
    reg = d.code_region != "BF"
    scores = pd.DataFrame(index=d.index)
    for f, (_, sens, _) in FACTEURS.items():
        rang = d.loc[reg, f].rank(pct=True)
        scores[f] = rang if sens > 0 else 1 - rang
    dims = {}
    for f, (dim, _, _) in FACTEURS.items():
        dims.setdefault(dim, []).append(f)
    for dim, fs in dims.items():
        cle = "indice_" + dim.split()[0].lower().replace("é", "e").replace("è", "e")
        d[cle] = (100 * scores[fs].mean(axis=1)).round(1)
    cols_dim = [c for c in d.columns if c.startswith("indice_")]
    d["indice_risque"] = d[cols_dim].mean(axis=1).round(1)
    return d


def exporter_limites():
    """Convertit le shapefile GADM des 13 régions (niveau 1) en GeoJSON pour la carte.

    Deux précautions pour l'affichage Plotly : le contour extérieur de chaque polygone est
    orienté dans le sens horaire (sinon la région est dessinée « à l'envers » et remplit tout
    le globe), et chaque entité porte le code de région utilisé dans la base.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon
    from shapely.geometry.polygon import orient

    sys.path.insert(0, str(Path(__file__).parent))
    base = __import__("01_constitution_base")
    g = gpd.read_file(RACINE / "gadm41_BFA_shp/gadm41_BFA_1.shp").to_crs(4326)
    noms = g.NAME_1.map(base.norm).replace(base.ALIAS_REGION)
    g["region"] = noms
    g["code_region"] = noms.map(base.REGIONS)
    assert g.code_region.notna().all() and g.code_region.nunique() == 13, "régions du shapefile non reconnues"
    g["geometry"] = g.geometry.simplify(0.003, preserve_topology=True).map(
        lambda geom: MultiPolygon([orient(p, -1.0) for p in getattr(geom, "geoms", [geom])]))
    # point d'ancrage de l'étiquette : le point le plus éloigné des frontières de la région
    # (« pôle d'inaccessibilité »), ce qui écarte les étiquettes des régions imbriquées du centre
    from shapely.ops import polylabel
    ancres = g.geometry.map(lambda geom: polylabel(max(geom.geoms, key=lambda p: p.area), tolerance=0.01))
    g["lon_etiquette"], g["lat_etiquette"] = ancres.map(lambda p: round(p.x, 4)), ancres.map(lambda p: round(p.y, 4))
    g[["code_region", "region", "lon_etiquette", "lat_etiquette", "geometry"]].to_file(SORTIE / "regions.geojson", driver="GeoJSON")


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    d = construire(pd.read_parquet(ENTREE))
    d.to_parquet(SORTIE / "indicateurs.parquet", index=False)
    d.to_csv(SORTIE / "indicateurs.csv", index=False)
    pd.DataFrame([(f, *v) for f, v in FACTEURS.items()],
                 columns=["facteur", "dimension", "sens", "libelle"]).to_csv(SORTIE / "indice_definition.csv", index=False)
    # limites des 13 régions, copiées à côté des résultats pour que la plateforme soit autonome
    exporter_limites()
    r = d[d.code_region != "BF"]
    print(f"{len(d)} lignes, {d.shape[1]} colonnes")
    print("Indice de risque 2024-2025, du plus élevé au plus faible :")
    print(r[r.annee_fin == r.annee_fin.max()].sort_values("indice_risque", ascending=False)
          [["region", "indice_risque", "indice_conditions", "indice_equipements", "indice_ecarts", "indice_acces", "abandon_t"]].to_string(index=False))
    print("Corrélation de rang indice / abandon de la même année :", round(r.indice_risque.corr(r.abandon_t, method="spearman"), 2))


if __name__ == "__main__":
    main()
