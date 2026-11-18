# Alerte précoce du décrochage scolaire au Burkina Faso

**Data Challenge Burkina – Défi 1 · Équipe Insighters**

- Zakaria BELEM, ML Engineer : modélisation prédictive, explication des résultats
- Daniel Glorieux ILBOUDO, Data Scientist : audit et nettoyage des données, analyse exploratoire, carte et tableau de bord

Un outil qui indique, pour chacune des 13 régions du Burkina Faso, le risque de décrochage au primaire attendu l'année scolaire suivante, les facteurs qui pèsent sur ce risque et le profil d'action qui en découle. Il répond à la question du défi : **où le risque de décrochage est-il le plus élevé, et pourquoi ?**

La démarche complète, les résultats et les limites sont décrits dans [`Documentation_Insighters_Defi1.pdf`](Documentation_Insighters_Defi1.pdf) (15 pages).

## Ce que fait la solution

| Étape | Contenu | Script |
| --- | --- | --- |
| Base de données | Les 21 tableaux fournis (préscolaire et primaire, 2015-16 à 2024-25) nettoyés et réunis en un panel région × année, avec un audit de qualité | `01_constitution_base.py` |
| Enrichissement | Ajout de sources officielles : annuaires statistiques du primaire (DGESS) et open data de l'INSD. Le panel passe de 43 à 126 variables, dont les taux d'abandon, de redoublement et de promotion | `02_telecharger_sources.py`, `03_enrichir_base.py` |
| 1. Indicateurs | Abandon, redoublement, promotion, achèvement, évolution des effectifs et des écoles, par région, année et sexe | `04_indicateurs_indice.py` |
| 2. Indice composite | Indice de risque de 0 à 100 sur quatre dimensions : conditions d'enseignement, équipements, écarts filles-garçons, accès et continuité | `04_indicateurs_indice.py` |
| 3. Modèle prédictif | Taux d'abandon de l'année suivante en l'absence de choc ; forêt aléatoire retenue après comparaison (persistance, moyenne historique, Ridge, gradient boosting) sur des années jamais vues. Il remplace le modèle du script 05, dont l'indice et les profils sont conservés | `06_modele_insighters.py` |
| 4. Explication et profils | Facteurs par région (SHAP) et regroupement des régions en profils de risque | `05_modele_risque.py` |
| 5. Plateforme | Application Streamlit : carte des risques, fiche par région, classement exportable en Excel et PDF, simulateur de scénarios | `app/streamlit_app.py` |
| Assistant | Questions en langage simple sur les résultats ; réponses de Claude fondées uniquement sur `data/modele/`, chiffres et sources cités | `app/assistant.py` |

## Installation et lancement

Python 3.11 ou plus récent (testé avec 3.12). Tout tourne sur processeur, en quelques minutes.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Les résultats déjà calculés sont dans le dépôt (`data/modele/`) : la plateforme se lance directement.

### Assistant conversationnel (facultatif)

L'onglet « Assistant » répond aux questions en langage simple avec Claude (API d'Anthropic). Il s'appuie uniquement sur les résultats de `data/modele/`, cite les chiffres utilisés et distingue associations et causes. Pour l'activer, créer un fichier `.env` à la racine du dépôt :

```bash
ANTHROPIC_API_KEY=votre_clé
CLAUDE_MODEL=claude-opus-5-5   # facultatif : modèle utilisé par défaut
```

`.env` est exclu de git par `.gitignore` : ne jamais publier la clé. En ligne (Streamlit Community Cloud), déclarer les mêmes variables dans les secrets de l'application. Sans clé, l'onglet indique simplement que l'assistant n'est pas configuré. Une réponse prend une dizaine de secondes.

### Relancer toute la chaîne

```bash
python3 scripts/01_constitution_base.py     # base fournie : nettoyage, panel, audit
python3 scripts/02_telecharger_sources.py   # télécharge les annuaires DGESS et les données de l'API INSD
python3 scripts/03_enrichir_base.py         # extraction des annuaires, fusion, contrôles croisés
python3 scripts/04_indicateurs_indice.py    # indicateurs, indice, limites des régions
python3 scripts/05_modele_risque.py         # indice, profils (et un premier modèle, remplacé par le script 06)
python3 scripts/06_modele_insighters.py     # modèle retenu : abandon en l'absence de choc, validation, SHAP
```

Les sources officielles téléchargées (environ 500 Mo de PDF) ne sont pas versionnées : le script 02 les récupère dans `data/externe/brut/` et doit être lancé avant le script 03 sur un clone neuf. `python3 scripts/02_telecharger_sources.py --lister` affiche ce qui serait téléchargé sans rien écrire.

## Organisation du dépôt

```
app/streamlit_app.py            plateforme web
scripts/                        chaîne de traitement, scripts 01 à 05
data/raw/                       classeur fourni par les organisateurs, nettoyé
data/clean/                     bases produites
  panel_enrichi.parquet         base de modélisation : 13 régions + national × 10 années, 126 variables
  dictionnaire_variables_enrichi.csv   libellé, source et taux de remplissage de chaque variable
  audit_qualite.csv, audit_enrichissement.csv   anomalies relevées, non corrigées
  flux_province.csv             promotion, redoublement, abandon des 45 provinces
data/externe/extrait/           tableaux extraits des annuaires (format long)
data/externe/brut/_manifest_telechargements.tsv   journal des téléchargements
data/modele/                    indicateurs, prévisions, validation, SHAP, profils, modèle, limites des régions
gadm41_BFA_shp/                 limites administratives (GADM 4.1)
Documentation_Insighters_Defi1.pdf   rapport
Note_intention.pdf              note d'intention de l'équipe
```

Dans le panel, la ligne `code_region == "BF"` est le total national : à exclure avant tout calcul par région.

## Sources

- Données du Data Challenge Burkina : effectifs, taux, écoles et salles de classe du préscolaire et du primaire, 2015-16 à 2024-25.
- [Annuaires statistiques de l'enseignement primaire](https://cns.bf/publications/mebapln/), DGESS, éditions 2014-15 à 2024-25 (tableaux I.1, I.7 et V.4.3).
- [Portail open data de l'INSD](https://burkinafaso.opendataforafrica.org/) : enseignants du primaire et population par région.
- [GADM 4.1](https://gadm.org/) : limites des 13 régions.

## Limites

- **Abandon et insécurité.** Dans les régions touchées, le taux d'abandon reflète en grande partie les fermetures d'écoles ; une baisse ne signifie pas que les élèves sont revenus.
- **Peu d'observations.** 13 régions sur 10 années (74 transitions sans choc pour l'apprentissage) : les prévisions sont des ordres de grandeur, avec une fourchette à 80 % de plus ou moins 3,9 points.
- **Prévision en l'absence de choc.** Le modèle ne prévoit ni les fermetures ni les retours massifs d'élèves : lors de ces années, son erreur moyenne atteint 13 points.
- **Années 2015-16 à 2017-18.** Les taux régionaux de flux y sont estimés à partir des provinces (colonne `prim_flux_origine`).
- **Associations, pas causes.** Les facteurs et le simulateur indiquent où regarder ; ils ne prouvent pas l'effet d'une intervention.

Les résultats servent à cibler des actions, jamais à stigmatiser une région.
