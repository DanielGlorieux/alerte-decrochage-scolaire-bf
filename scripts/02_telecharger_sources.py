"""
02 - Téléchargement des sources officielles d'enrichissement
============================================================
Sources (toutes officielles) :
  A. DGESS du ministère en charge de l'éducation, publications hébergées par le Conseil
     national de la statistique (cns.bf) : annuaires statistiques de l'enseignement primaire
     et tableaux de bord. L'index est parcouru (les identifiants ne sont pas chronologiques).
  B. INSD, portail open data (burkinafaso.opendataforafrica.org) : API sans clé.
  C. Anciennes éditions des annuaires (2014/15 à 2016/17), retirées de cns.bf : copies
     archivées par la Wayback Machine.
  D. Limites des 13 régions : geoBoundaries (ADM1).

Les fichiers déjà présents ne sont pas retéléchargés.
Usage : python3 scripts/02_telecharger_sources.py [--lister]
"""
from pathlib import Path
import html
import json
import re
import sys
import time

import requests

RACINE = Path(__file__).resolve().parents[1]
BRUT = RACINE / "data/externe/brut"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"}
PAUSE = 1.5

CNS_INDEX = "https://cns.bf/publications/mebapln/?mode=liste&tri=recent&par_page=48&page={page}"
CNS_FICHIER = "https://cns.bf/publications/telecharger/{id}/"   # GET uniquement (HEAD -> 404)
API_INSD = "https://burkinafaso.opendataforafrica.org/api/1.0/data/raw/"

# Jeux de données de l'API INSD utiles au niveau régional
JEUX_INSD = {
    "vzkctf": "Taux de redoublement par région, niveau et sexe",
    "tyliupe": "Enseignants du primaire par région et sexe",
    "cvbkpse": "Écoles et salles de classe du primaire par région",
    "rkezvcd": "Effectifs du primaire par région et sexe",
    "pruxgv": "TBS du primaire par région",
    "iebcqme": "Population par région et sexe",
    "cpjmkh": "Promotion, redoublement, abandon (national)",
    "iiyaoef": "Ratio élèves/maître (national)",
}
WAYBACK = {
    "2014-2015": "https://web.archive.org/web/20210815131810id_/http://www.cns.bf/IMG/pdf/annuaire_2014-2015_primaire.pdf",
    "2015-2016": "https://web.archive.org/web/20210815131748id_/http://www.cns.bf/IMG/pdf/annuaire_2015-2016_du_primaire.pdf",
    "2016-2017": "https://web.archive.org/web/20180827203247id_/http://www.cns.bf:80/IMG/pdf/annuaire_primaire_2016_2017_final.pdf",
}
GEO = "https://www.geoboundaries.org/api/current/gbOpen/BFA/ADM1/"


def get(url, **kw):
    for essai in range(4):
        r = requests.get(url, headers=UA, timeout=600, **kw)
        if r.status_code == 429:           # Wayback limite fortement le débit
            time.sleep(60 * (essai + 1))
            continue
        r.raise_for_status()
        time.sleep(PAUSE)
        return r
    r.raise_for_status()


def index_cns():
    """Parcourt l'index des publications du ministère et renvoie [(id, type, année scolaire, titre)]."""
    pubs, page = {}, 1
    while True:
        h = get(CNS_INDEX.format(page=page)).text
        cartes = re.findall(r'href="/publications/telecharger/(\d+)/".*?<h3[^>]*>(.*?)</h3>', h, flags=re.S)
        nouveaux = {int(i): html.unescape(re.sub(r"<[^>]+>|\s+", " ", t)).strip() for i, t in cartes}
        if not set(nouveaux) - set(pubs):
            break
        pubs.update(nouveaux)
        page += 1
    out = []
    for i, titre in sorted(pubs.items()):
        t = titre.lower()
        an = re.search(r"(20\d\d)\s*[-_/ ]\s*(20\d\d)", titre)
        if "primaire" not in t or not an:
            continue
        genre = ("synthese_annuaire" if "synth" in t else "annuaire" if "annuaire" in t
                 else "tableau_bord" if "tableau de bord" in t else None)
        if genre in ("annuaire", "tableau_bord"):
            out.append((i, genre, f"{an.group(1)}-{an.group(2)}", titre))
    return out


def telecharger(url, dest: Path, lister: bool):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  déjà présent  {dest.relative_to(RACINE)}")
        return
    print(f"  {'à télécharger' if lister else 'téléchargement'}  {url} -> {dest.relative_to(RACINE)}")
    if lister:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get(url, allow_redirects=True)
    if dest.suffix == ".pdf" and not r.content.startswith(b"%PDF"):
        print("    ! réponse non PDF, fichier ignoré")
        return
    dest.write_bytes(r.content)


def api_insd(jeu: str, dest: Path, lister: bool):
    if dest.exists():
        print(f"  déjà présent  {dest.relative_to(RACINE)}")
        return
    print(f"  API INSD {jeu}")
    if lister:
        return
    series, jeton = [], None
    while True:
        r = requests.post(API_INSD, params={"continuationToken": jeton} if jeton else None,
                          json={"Dataset": jeu}, headers=UA, timeout=120)
        r.raise_for_status()
        rep = r.json()
        series += rep.get("data", [])
        jeton = rep.get("continuationToken")
        time.sleep(PAUSE)
        if not jeton:
            break
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"dataset": jeu, "data": series}, ensure_ascii=False), encoding="utf8")


def main():
    lister = "--lister" in sys.argv
    print("A. Publications DGESS (index cns.bf)")
    for i, genre, an, titre in index_cns():
        telecharger(CNS_FICHIER.format(id=i), BRUT / f"menapln/{genre}_primaire_{an}__cns{i}.pdf", lister)
    print("B. API open data de l'INSD")
    for jeu in JEUX_INSD:
        api_insd(jeu, BRUT / f"insd/opendata_api/{jeu}.json", lister)
    print("C. Anciennes éditions (Wayback Machine)")
    for an, url in WAYBACK.items():
        telecharger(url, BRUT / f"menapln/annuaire_primaire_{an}__wayback.pdf", lister)
    print("D. Limites des régions (geoBoundaries)")
    dest = BRUT / "geo/geoBoundaries-BFA-ADM1.geojson"
    if dest.exists():
        print(f"  déjà présent  {dest.relative_to(RACINE)}")
    elif not lister:
        telecharger(get(GEO).json()["gjDownloadURL"], dest, lister)


if __name__ == "__main__":
    main()
