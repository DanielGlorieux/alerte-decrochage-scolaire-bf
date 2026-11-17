"""
03 - Extraction des sources officielles et enrichissement du panel région x année
=================================================================================
Ce que les données DCB ne contiennent pas et que ce script ajoute :

  Source                                         Variables (par région et année scolaire)
  ---------------------------------------------  ---------------------------------------------
  Annuaire statistique du primaire (DGESS)
    tableau I.7                                  taux de promotion, redoublement, abandon
                                                 par niveau (CP1..CM2) et par sexe
    tableau I.1                                  salles de classe, groupes pédagogiques,
                                                 maîtres chargés de cours, ratio élèves/maître,
                                                 par statut
    tableau V.4.3 (public + privé)               écoles disposant d'eau potable, d'électricité,
                                                 de latrines fonctionnelles, d'une cantine, ...
  API open data de l'INSD                        enseignants par sexe, population par sexe,
                                                 taux de redoublement (sert de contre-vérification)

Étapes : importation (pdftotext -layout, texte mis en cache) -> épuration (lecture des lignes
de total régional) -> contrôles croisés avec la base DCB -> fusion sur (code_region, annee_fin).

Sorties : data/externe/extrait/*.csv            une table par source, format long
          data/clean/panel_enrichi.{parquet,csv}
          data/clean/flux_province.csv           promotion/redoublement/abandon par province
          data/clean/dictionnaire_variables_enrichi.csv
          data/clean/audit_enrichissement.csv
"""
from pathlib import Path
import json
import re
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
base = __import__("01_constitution_base")
norm, REGIONS = base.norm, base.REGIONS

RACINE = Path(__file__).resolve().parents[1]
BRUT = RACINE / "data/externe/brut"
CACHE = RACINE / "data/externe/texte"
EXTRAIT = RACINE / "data/externe/extrait"
CLEAN = RACINE / "data/clean"

NIVEAUX = ["cp1", "cp2", "ce1", "ce2", "cm1", "cm2"]
NB = r"\d+(?: \d{3})*(?:,\d+)?"               # 1 234 / 1447 / 56,7  (milliers séparés ou non par une espace)
AUDIT = []


def note(controle, gravite, cle, detail=""):
    AUDIT.append({"controle": controle, "gravite": gravite, "cle": str(cle), "detail": detail})


def nombre(s: str) -> float:
    return float(s.replace(" ", "").replace(",", "."))


def cle_region(s: str):
    """Libellé libre -> nom de région harmonisé, ou None ('CENTRE - EST', 'SUD OUEST', ...)."""
    n = re.sub(r"\s*-\s*", "-", norm(s))
    n = {"SUD OUEST": "SUD-OUEST", "HAUTS BASSINS": "HAUTS-BASSINS", "PLATEAU-CENTRAL": "PLATEAU CENTRAL",
         "TOTAL NATIONAL": "NATIONAL", "BURKINA FASO": "NATIONAL"}.get(n, n)
    for a, b in [("CENTRE EST", "CENTRE-EST"), ("CENTRE NORD", "CENTRE-NORD"),
                 ("CENTRE OUEST", "CENTRE-OUEST"), ("CENTRE SUD", "CENTRE-SUD")]:
        n = b if n == a else n
    return n if n in REGIONS else None


def cle_province(s):
    """Harmonise les noms de province entre tableaux ('LES BALE' / 'BALE', ...)."""
    n = re.sub(r"^LES ", "", norm(s)) if isinstance(s, str) else s
    return ALIAS_PROVINCE.get(n, n)


ALIAS_PROVINCE = {}


# ---------------------------------------------------------------------------------------
# 1. Importation : texte des annuaires
# ---------------------------------------------------------------------------------------
def pages_annuaire(pdf: Path) -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    txt = CACHE / (pdf.stem + ".txt")
    if not txt.exists():
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
    return txt.read_text(encoding="utf8", errors="replace").split("\f")


def annuaires():
    for pdf in sorted((BRUT / "menapln").glob("annuaire_primaire_*.pdf")):
        an = re.search(r"(20\d\d)-(20\d\d)", pdf.name)
        yield int(an.group(2)), pdf.name, pages_annuaire(pdf)


# ---------------------------------------------------------------------------------------
# 2. Épuration : lecture des trois tableaux
# ---------------------------------------------------------------------------------------
def lire_flux(an, fichier, pages):
    """Tableau I.7 : blocs de 3 lignes (TX Prom / TX Red / TX Aban) par province puis par région."""
    out, region = [], None
    cible = [(i, p) for i, p in enumerate(pages)
             if re.search(r"TAUX DE PROMOTION, REDOUBLEMENT", p[:700]) and "TX Prom" in p]
    for i, p in cible:
        bloc = None
        for ligne in p.split("\n"):
            if "TX" not in ligne:
                texte = ligne.strip()
                if bloc is not None and texte and not re.search(r"\d", texte):
                    bloc["nom"].append(texte)            # nom d'entité coupé sur deux lignes
                elif bloc is None and cle_region(texte) and cle_region(texte) != "NATIONAL":
                    region = cle_region(texte)           # ligne de titre de région
                continue
            etiquette = re.search(r"TX\s+(Prom|Red|Aban)", ligne)
            if not etiquette:
                continue
            t = etiquette.group(1)
            if t == "Prom":
                bloc = {"nom": [], "page": i + 1}
            if bloc is None:
                continue
            bloc["nom"].append(ligne[: etiquette.start()].strip())
            bloc[t] = [nombre(x) for x in re.findall(r"\d+,\d+", ligne[etiquette.end():])]
            if t == "Aban":
                nom = " ".join(x for x in bloc["nom"] if x)
                reg = cle_region(nom)
                niveau_geo = "national" if reg == "NATIONAL" else "region" if reg else "province"
                attendu = {"Prom": 15, "Red": 18, "Aban": 15}
                for taux, cle in [("promotion", "Prom"), ("redoublement", "Red"), ("abandon", "Aban")]:
                    v = bloc.get(cle, [])
                    if len(v) != attendu[cle]:
                        note("annuaire I.7 : nombre de valeurs inattendu", "avertissement",
                             (an, nom, taux), f"{len(v)} valeurs au lieu de {attendu[cle]} (p. {bloc['page']})")
                        continue
                    for k, x in enumerate(v):
                        out.append({"annee_fin": an, "niveau_geo": niveau_geo,
                                    "region": reg or region, "province": None if reg else norm(nom),
                                    "taux": taux, "niveau": NIVEAUX[k // 3], "sexe": "GFT"[k % 3],
                                    "valeur": x, "fichier": fichier, "page": bloc["page"]})
                bloc = None
    return out


def lire_i1(an, fichier, pages):
    """Tableau I.1 : totaux par province, lignes 'Total Région' (Public / Privé / Total), 'Total National'."""
    out, region = [], None
    cols = ["ecoles", "salles", "groupes_ped", "maitres", "eleves", "ratio_eleves_maitre"]
    for i, p in enumerate(pages):
        tete = p[:900]
        if not re.search(r"I\.\s?1\.?\s+NOMBRE D.ECOLES", tete) or "COMMUNE" in tete or "PRIORITAIRES" in tete:
            continue
        if "pédagogiq" not in tete:
            note("annuaire I.1 : en-tête inattendu", "avertissement", (an, i + 1))
            continue
        courant, province, national = None, None, False
        for ligne in p.split("\n"):
            texte = ligne.strip()
            if cle_region(texte) and cle_region(texte) != "NATIONAL":
                region, courant, province = cle_region(texte), None, None
                continue
            if re.match(r"Total National", texte):
                national, courant, province = True, None, None   # reste vrai jusqu'à la fin de la page
            elif re.match(r"Total R[ée]gion", texte):
                courant, province = region, None
            elif re.match(r"Total Province", texte):
                courant, province = None, None
            m = re.search(r"^(.*?)\b(Public|Priv[ée]|Total)\s+((?:%s\s*){6})$" % NB, texte)
            if not m:
                continue
            nom = m.group(1).strip()
            if nom and not nom.startswith("Total"):
                courant, province = None, norm(nom)              # première ligne d'une province
            zone = "NATIONAL" if national else courant
            if not zone and not province:
                continue
            v = [nombre(x) for x in re.findall(NB, m.group(3))]
            statut = {"Public": "public", "Total": "total"}.get(m.group(2), "prive")
            if len(v) != 6 or (v[3] and abs(v[4] / v[3] - v[5]) > 0.15):
                note("annuaire I.1 : ligne incohérente (ratio)", "avertissement", (an, zone or province, statut), texte[:120])
                continue
            out.append({"annee_fin": an, "niveau_geo": "province" if not zone else "region",
                        "region": zone or region, "province": None if zone else province, "statut": statut,
                        **dict(zip(cols, v)), "fichier": fichier, "page": i + 1})
    return out


def lire_equipements(an, fichier, pages):
    """Tableau V.4.3 (Public+Privé) : lignes 'Total Région', contrôlées par la ligne 'Pourcentage'."""
    out, region, attente = [], None, None
    cols = ["ecoles", "eau_potable", "electricite", "latrines", "jardin", "cantine", "terrain_sport", "cloture"]
    for i, p in enumerate(pages):
        tete = p[:900]
        if not (re.search(r"4\.3\s+Public\s*\+\s*Priv", tete) and "EQUIPEMENT DES ECOLES" in tete and "Eau potable" in tete):
            continue
        for ligne in p.split("\n"):
            texte = ligne.strip()
            if cle_region(texte) and cle_region(texte) != "NATIONAL":
                region = cle_region(texte)
            elif re.match(r"Total (R[ée]gion|National)", texte):
                v = [nombre(x) for x in re.findall(NB, texte)]
                attente = ("NATIONAL" if "National" in texte else region, v)
            elif texte.startswith("Pourcentage") and attente:
                reg, v = attente
                pc = [nombre(x) for x in re.findall(r"\d+,\d+", texte)]
                ok = len(v) == 8 and len(pc) == 8 and all(abs(100 * a / v[0] - b) < 0.11 for a, b in zip(v, pc))
                if ok:
                    out.append({"annee_fin": an, "region": reg, **dict(zip(cols, v)), "fichier": fichier, "page": i + 1})
                else:
                    note("annuaire V.4.3 : total incohérent avec la ligne de pourcentage", "avertissement", (an, reg), str(v))
                attente = None
    return out


def lire_api(jeu: str) -> pd.DataFrame:
    """Séries annuelles de l'API INSD -> format long (une ligne par série et par année)."""
    d = json.loads((BRUT / f"insd/opendata_api/{jeu}.json").read_text(encoding="utf8"))["data"]
    lignes = []
    for s in d:
        dims = {norm(k).lower(): v["name"] for k, v in s.items() if isinstance(v, dict)}
        debut = int(s["startDate"][:4])
        for k, v in enumerate(s["values"]):
            lignes.append({**dims, "annee": debut + k, "valeur": v})
    df = pd.DataFrame(lignes)
    col_reg = "region" if "region" in df else "regions"
    df["region"] = df[col_reg].map(cle_region)
    return df


# ---------------------------------------------------------------------------------------
# 3. Fusion
# ---------------------------------------------------------------------------------------
def main():
    EXTRAIT.mkdir(parents=True, exist_ok=True)
    flux, i1, equip = [], [], []
    for an, fichier, pages in annuaires():
        f, a, e = lire_flux(an, fichier, pages), lire_i1(an, fichier, pages), lire_equipements(an, fichier, pages)
        print(f"annuaire {an - 1}-{an} : I.7 {len(f):>5} valeurs | I.1 {len(a):>3} lignes | V.4.3 {len(e):>3} lignes")
        flux += f; i1 += a; equip += e
    flux, i1, equip = pd.DataFrame(flux), pd.DataFrame(i1), pd.DataFrame(equip)
    i1_prov = i1[i1.niveau_geo == "province"].copy()
    i1 = i1[i1.niveau_geo == "region"].drop(columns=["niveau_geo", "province"])
    for df in (flux, i1, equip):
        df["code_region"] = df.region.map(REGIONS)

    # doublons éventuels (tableau repris sur plusieurs pages)
    for nom, df, k in [("I.7", flux, ["annee_fin", "niveau_geo", "region", "province", "taux", "niveau", "sexe"]),
                       ("I.1", i1, ["annee_fin", "region", "statut"]), ("V.4.3", equip, ["annee_fin", "region"])]:
        dup = df.duplicated(k, keep=False)
        if dup.any():
            note(f"annuaire {nom} : clés en double", "avertissement", nom, f"{dup.sum()} lignes ; la première est gardée")
    flux = flux.drop_duplicates(["annee_fin", "niveau_geo", "region", "province", "taux", "niveau", "sexe"])
    i1 = i1.drop_duplicates(["annee_fin", "region", "statut"])
    equip = equip.drop_duplicates(["annee_fin", "region"])

    # total national des équipements = somme des 13 régions (non lu dans le PDF)
    nat = equip.groupby("annee_fin").sum(numeric_only=True).drop(columns="page").reset_index()
    nat = nat[equip.groupby("annee_fin").region.nunique().values == 13].assign(region="NATIONAL", code_region="BF")
    equip = pd.concat([equip, nat], ignore_index=True)

    # Les éditions 2014/15 à 2017/18 ne publient pas de ligne régionale au tableau I.7 :
    # le taux régional y est estimé par la moyenne des provinces pondérée par leurs effectifs
    # (tableau I.1). La même estimation est faite pour les autres éditions afin d'en mesurer l'erreur.
    poids = i1_prov[i1_prov.statut == "total"].assign(province=lambda d: d.province.map(cle_province))[["annee_fin", "province", "eleves"]]
    fp = flux[flux.niveau_geo == "province"].assign(province=lambda d: d.province.map(cle_province)).merge(poids, on=["annee_fin", "province"], how="left")
    sans_poids = fp[fp.eleves.isna()][["annee_fin", "province"]].drop_duplicates()
    for _, r in sans_poids.iterrows():
        note("province sans effectif (pondération)", "avertissement", (r.annee_fin, r.province))
    fp = fp.dropna(subset=["eleves"])
    k = ["annee_fin", "region", "taux", "niveau", "sexe"]
    est = (fp.assign(px=fp.valeur * fp.eleves).groupby(k)[["px", "eleves"]].sum()
             .assign(valeur=lambda d: (d.px / d.eleves).round(1)).reset_index()[k + ["valeur"]])
    publie = flux[flux.niveau_geo == "region"]
    comp = est.merge(publie[k + ["valeur"]], on=k, suffixes=("_est", "_pub"))
    err = (comp.valeur_est - comp.valeur_pub).abs()
    print(f"estimation des taux régionaux par les provinces : écart absolu moyen {err.mean():.2f} point, "
          f"médian {err.median():.2f}, 95e centile {err.quantile(.95):.2f} (n={len(comp)}, éditions avec ligne régionale)")
    note("taux régionaux estimés par les provinces : erreur mesurée", "info", "éditions 2018/19 à 2024/25",
         f"écart absolu moyen {err.mean():.2f} pt ; médian {err.median():.2f} ; 95e centile {err.quantile(.95):.2f}")
    annees_pub = set(publie.annee_fin)
    est = est[~est.annee_fin.isin(annees_pub)].assign(niveau_geo="region", province=None, code_region=lambda d: d.region.map(REGIONS),
                                                      fichier="estimation (provinces pondérées)", page=None)
    flux = pd.concat([flux, est], ignore_index=True)

    flux.to_csv(EXTRAIT / "annuaire_flux_promotion_redoublement_abandon.csv", index=False)
    i1.to_csv(EXTRAIT / "annuaire_ecoles_salles_maitres.csv", index=False)
    i1_prov.to_csv(EXTRAIT / "annuaire_ecoles_salles_maitres_province.csv", index=False)
    equip.to_csv(EXTRAIT / "annuaire_equipements.csv", index=False)

    cle = ["code_region", "annee_fin"]
    panel = pd.read_parquet(CLEAN / "panel_region_annee.parquet")
    dico = pd.read_csv(CLEAN / "dictionnaire_variables.csv")
    ajout = []                                           # (variable, libellé, source)

    # --- I.7 : flux par région --------------------------------------------------------
    fr = flux[flux.niveau_geo != "province"].copy()
    fr["variable"] = "prim_" + fr.taux + "_" + fr.niveau + "_" + fr.sexe.str.lower()
    w = fr.pivot(index=cle, columns="variable", values="valeur")
    for taux in ["promotion", "redoublement", "abandon"]:
        for s in "gft":
            c = [x for x in w.columns if x.startswith(f"prim_{taux}_") and x.endswith(f"_{s}")]
            w[f"prim_{taux}_moy_{s}"] = w[c].mean(axis=1).round(2)
    lib_taux = {"promotion": "Taux de promotion", "redoublement": "Taux de redoublement", "abandon": "Taux d'abandon"}
    lib_sexe = {"g": "garçons", "f": "filles", "t": "ensemble"}
    for c in w.columns:
        _, taux, niv, s = c.split("_")
        niv_l = "moyenne simple des niveaux" if niv == "moy" else niv.upper()
        ajout.append((c, f"{lib_taux[taux]} (%) - {niv_l} - {lib_sexe[s]}", "Annuaire statistique du primaire, tableau I.7 (DGESS)"))
    w["prim_flux_origine"] = ["publie" if (a in annees_pub or c == "BF") else "estime_provinces" for c, a in w.index]
    ajout.append(("prim_flux_origine", "Origine des taux de flux : publie (ligne régionale de l'annuaire) ou estime_provinces", "Calcul"))
    panel = panel.merge(w.reset_index(), on=cle, how="left")

    # --- I.1 : salles, maîtres, ratio -------------------------------------------------
    w1 = i1.pivot(index=cle, columns="statut", values=["salles", "groupes_ped", "maitres", "ratio_eleves_maitre"])
    w1.columns = [f"prim_{a}_{b}" for a, b in w1.columns]
    lib1 = {"salles": "Nombre de salles de classe", "groupes_ped": "Nombre de groupes pédagogiques",
            "maitres": "Nombre de maîtres chargés de cours", "ratio_eleves_maitre": "Ratio élèves/maître"}
    for c in w1.columns:
        a, b = c[5:].rsplit("_", 1)
        ajout.append((c, f"{lib1[a]} - {b} (primaire)", "Annuaire statistique du primaire, tableau I.1 (DGESS)"))
    panel = panel.merge(w1.reset_index(), on=cle, how="left")
    panel["prim_eleves_par_salle"] = (panel.prim_effectif_t / panel.prim_salles_total).round(1)
    ajout.append(("prim_eleves_par_salle", "Élèves par salle de classe (effectif DCB / salles annuaire)", "Calcul"))

    # --- V.4.3 : équipements ----------------------------------------------------------
    e = equip.set_index(cle)
    lib_e = {"eau_potable": "d'eau potable", "electricite": "d'électricité", "latrines": "de latrines fonctionnelles",
             "jardin": "d'un jardin scolaire", "cantine": "d'une cantine", "terrain_sport": "d'un terrain de sport",
             "cloture": "d'une clôture"}
    w2 = pd.DataFrame({f"prim_pct_ecoles_{k}": (100 * e[k] / e.ecoles).round(1) for k in lib_e})
    for k, v in lib_e.items():
        ajout.append((f"prim_pct_ecoles_{k}", f"Écoles disposant {v} (% des écoles, public + privé)",
                      "Annuaire statistique du primaire, tableau V.4.3 (DGESS)"))
    panel = panel.merge(w2.reset_index(), on=cle, how="left")

    # --- API INSD : enseignants et population ------------------------------------------
    ens = lire_api("tyliupe").pivot_table(index=["region", "annee"], columns="sexe", values="valeur")
    ens = pd.DataFrame({"prim_enseignants_total": ens["Ensemble"],
                        "prim_pct_enseignantes": (100 * ens["Femmes"] / ens["Ensemble"]).round(1)}).reset_index()
    pop = lire_api("iebcqme").pivot_table(index=["region", "annee"], columns="sexe", values="valeur")
    pop = pop.rename(columns={"Homme": "pop_hommes", "Femme": "pop_femmes", "Ensemble": "pop_totale"}).reset_index()
    for df in (ens, pop):
        df["code_region"] = df.region.map(REGIONS)
        panel = panel.merge(df.rename(columns={"annee": "annee_fin"}).drop(columns="region"), on=cle, how="left")
    src = "INSD, portail open data"
    ajout += [("prim_enseignants_total", "Enseignants du primaire (ensemble)", src),
              ("prim_pct_enseignantes", "Part de femmes parmi les enseignants du primaire (%)", src),
              ("pop_hommes", "Population masculine en milieu d'année", src),
              ("pop_femmes", "Population féminine en milieu d'année", src),
              ("pop_totale", "Population totale en milieu d'année", src)]

    # ---------------------------------------------------------------------------------
    # 4. Contrôles croisés avec la base DCB
    # ---------------------------------------------------------------------------------
    t = i1[i1.statut == "total"].merge(panel[cle + ["region", "prim_ecoles_total", "prim_effectif_t"]], on=cle, suffixes=("_ann", ""))
    for col_a, col_d, lib in [("ecoles", "prim_ecoles_total", "écoles"), ("eleves", "prim_effectif_t", "élèves")]:
        ecart = (t[col_a] - t[col_d]).abs()
        print(f"contrôle annuaire I.1 / DCB, {lib} : {int((ecart == 0).sum())} région-années identiques sur {len(t)}")
        for _, r in t[ecart > 0].iterrows():
            note(f"annuaire I.1 différent de DCB ({lib})", "avertissement", (r.region, r.annee_fin),
                 f"annuaire={r[col_a]:.0f} ; DCB={r[col_d]:.0f}")
    # redoublement : annuaire (édition N) comparé à l'API INSD (année N puis N-1)
    api = lire_api("vzkctf")
    api["niveau"] = api.niveau.str.lower()
    api["sexe"] = api.sexe.map({"Garçons": "G", "Filles": "F", "Ensemble": "T"})
    red = fr[fr.taux == "redoublement"]
    for decalage in (0, -1):
        a = api.assign(annee_fin=api.annee - decalage)
        m = red.merge(a, on=["region", "annee_fin", "niveau", "sexe"], suffixes=("_ann", "_api"))
        proche = ((m.valeur_ann - m.valeur_api).abs() < 0.06).mean()
        print(f"contrôle redoublement annuaire(N) / API INSD(année N{decalage:+d}) : {proche:.0%} de valeurs identiques (n={len(m)})")
        note("redoublement annuaire / API INSD", "info", f"décalage {decalage:+d}", f"{proche:.0%} identiques sur {len(m)}")

    # manquants des nouvelles variables
    nouvelles = [a[0] for a in ajout]
    for c in nouvelles:
        n = panel[c].isna().sum()
        if n:
            ans = sorted(panel.annee_fin[panel[c].isna()].unique())
            note("variable enrichie incomplète", "info", c, f"{n} manquants ; années : {ans}")

    # ---------------------------------------------------------------------------------
    # 5. Sorties
    # ---------------------------------------------------------------------------------
    panel.to_parquet(CLEAN / "panel_enrichi.parquet", index=False)
    panel.to_csv(CLEAN / "panel_enrichi.csv", index=False)
    prov = flux[flux.niveau_geo == "province"].drop(columns=["niveau_geo"])
    prov.to_csv(CLEAN / "flux_province.csv", index=False)
    d2 = pd.DataFrame(ajout, columns=["variable", "libelle", "source"])
    d2["taux_remplissage"] = [f"{panel[c].notna().mean():.0%}" for c in d2.variable]
    pd.concat([dico, d2], ignore_index=True).to_csv(CLEAN / "dictionnaire_variables_enrichi.csv", index=False)
    audit = pd.DataFrame(AUDIT)
    audit.to_csv(CLEAN / "audit_enrichissement.csv", index=False)

    print(f"\nPanel enrichi : {panel.shape[0]} lignes x {panel.shape[1]} colonnes "
          f"({len(nouvelles)} variables ajoutées) ; provinces : {prov.province.nunique()}")
    print(audit.groupby(["gravite", "controle"]).size().to_string())


if __name__ == "__main__":
    main()
