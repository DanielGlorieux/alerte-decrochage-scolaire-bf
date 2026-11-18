"""Assistant conversationnel de la plateforme : répond en langage simple, uniquement à partir des résultats
calculés par la chaîne (data/modele/), et cite les chiffres utilisés.

Le modèle est Claude (API d'Anthropic, SDK officiel). Le contexte de données est reconstruit à partir des
fichiers de data/modele/ : il reste à jour quand les scripts 04 et 05 sont relancés. Il est mis en cache
côté Anthropic (les questions suivantes le relisent plus vite et pour moins cher). En cas de refus du modèle
principal, la requête est reprise par un modèle de secours désigné par Anthropic (paramètre « fallbacks »).

Configuration, dans un fichier .env à la racine du dépôt (exclu de git) ou dans les secrets Streamlit :
  ANTHROPIC_API_KEY  clé de l'API Anthropic (obligatoire)
  CLAUDE_MODEL       modèle (par défaut claude-opus-5-5)
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

RACINE = Path(__file__).resolve().parents[1]
M = RACINE / "data" / "modele"
MODELE_DEFAUT = "claude-opus-5-5"
EFFORT = "medium"      # exactitude des chiffres d'abord ; une réponse prend une dizaine de secondes
MAX_TOKENS = 16000
MESSAGES_GARDES = 10   # derniers messages de la conversation transmis avec chaque question

SUGGESTIONS = [
    "Quelles sont les cinq régions les plus à risque ?",
    "Pourquoi le Sahel est-il en tête des priorités ?",
    "Les filles abandonnent-elles plus que les garçons ?",
    "Quelle confiance accorder aux prévisions ?",
]

CONSIGNES = """Tu es l'assistant de la plateforme d'alerte précoce du décrochage scolaire au primaire au \
Burkina Faso (Data Challenge Burkina 2026, Défi 1, équipe Insighters). Tu aides des décideurs (ministère, \
directions régionales, partenaires) à lire les résultats.

Règles :
1. Réponds uniquement à partir des données fournies ci-dessous, calculées par la plateforme. N'utilise aucune \
autre connaissance sur les régions, la sécurité ou l'actualité. N'invente jamais un chiffre et n'ajoute jamais \
une colonne ou une année absente des données ; une simple différence entre deux chiffres fournis est permise si \
tu dis que tu l'as calculée.
2. Si la réponse n'est pas dans les données, dis-le clairement, puis indique ce que les données permettent de dire.
3. Cite les chiffres que tu utilises, avec leur année scolaire, et indique la section d'où ils viennent \
(par exemple « Fiches par région »).
4. Ne fais aucune prévision en dehors de celles fournies (taux d'abandon prévu). Les facteurs (SHAP) et le \
simulateur décrivent des associations, jamais des causes démontrées : ne présente jamais une association comme \
une cause.
5. Signale les limites quand elles comptent pour la réponse : dans les régions touchées par l'insécurité, \
l'abandon mesure surtout des fermetures d'écoles ; les fourchettes de prévision sont larges ; 13 régions seulement.
6. Les résultats servent à cibler les actions, jamais à stigmatiser une région : reste factuel et respectueux.
7. Réponds en français, de façon concise (quelques phrases ou une courte liste), avec des nombres au format \
français (virgule décimale, espace avant « % »). Lis chaque chiffre dans le champ qui le nomme, sans le \
confondre avec un autre.
8. Pour chiffrer l'effet d'un changement de conditions, renvoie vers l'onglet « Simulateur ».
9. Si une demande sort de ce rôle, décline poliment ; tu peux toujours expliquer la méthode de la plateforme.
10. N'utilise jamais de tiret long (ni « — » ni « – ») : remplace-le par une virgule, un deux-points ou des \
parenthèses. Pour un intervalle, écris « de 6,8 à 32,2 ». Le trait d'union normal (-) reste permis dans les noms \
(Centre-Nord, 2024-2025).

Les données suivent."""


class ErreurAssistant(Exception):
    """Erreur présentable telle quelle à l'utilisateur."""


@dataclass(frozen=True)
class Configuration:
    cle: str
    modele: str


def configuration() -> Configuration | None:
    load_dotenv(RACINE / ".env")

    def lire(nom: str) -> str | None:
        valeur = os.environ.get(nom)
        if valeur:
            return valeur
        try:
            return st.secrets.get(nom)
        except Exception:  # pas de fichier de secrets
            return None

    cle = lire("ANTHROPIC_API_KEY")
    if not cle:
        return None
    return Configuration(cle=cle, modele=lire("CLAUDE_MODEL") or MODELE_DEFAUT)


@st.cache_resource
def _client(cle: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=cle)


# --- Contexte ----------------------------------------------------------------------------------

def _n(v, d: int = 1) -> str:
    """Nombre à la française ; « n.d. » si absent."""
    if v is None or pd.isna(v):
        return "n.d."
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def _csv(df: pd.DataFrame, d: int = 1) -> str:
    return df.round(d).to_csv(index=False, sep=";", decimal=",").strip()


def _annuel(ind: pd.DataFrame, colonne: str, d: int = 1) -> str:
    t = ind.pivot(index="region", columns="annee_scolaire", values=colonne)
    return _csv(t.reset_index().rename(columns={"region": "Région"}), d)


@st.cache_data
def contexte() -> str:
    ind = pd.read_parquet(M / "indicateurs.parquet")
    prev = pd.read_csv(M / "previsions.csv").sort_values("rang_priorite")
    shap = pd.read_csv(M / "shap.csv")
    val = pd.read_csv(M / "validation_modeles.csv")
    profils = pd.read_csv(M / "profils.csv")
    definition = pd.read_csv(M / "indice_definition.csv")
    meta = json.loads((M / "meta.json").read_text(encoding="utf8"))
    reg = ind[ind.code_region != "BF"].sort_values(["region", "annee_fin"])
    nat = ind[ind.code_region == "BF"].sort_values("annee_fin")
    der = reg[reg.annee_fin == reg.annee_fin.max()].set_index("region")
    an_prev, an_don = meta["annee_prevue"], meta["annee_donnees"]
    s1, s2, s3 = meta["seuils_niveaux"]

    fiches = []
    for p in prev.itertuples():
        d = der.loc[p.region]
        facteurs = (shap[shap.region == p.region].assign(a=lambda x: x.contribution.abs())
                    .nlargest(5, "a"))
        detail = ", ".join(f"{f.libelle} = {_n(f.valeur)} ({'+' if f.contribution > 0 else '−'}"
                           f"{_n(abs(f.contribution), 2)} pt)" for f in facteurs.itertuples())
        fiches.append(
            f"- {p.region} : rang de priorité {p.rang_priorite} sur 13 ; niveau de risque {p.niveau_risque} ; "
            f"abandon prévu {an_prev} {_n(p.risque_prevu)} % [fourchette {_n(p.borne_basse)} ; {_n(p.borne_haute)}] ; "
            f"confiance {p.confiance} (erreur passée {_n(p.erreur_passee)} pts) ; "
            f"abandon {an_don} {_n(p.abandon_t)} % (filles {_n(p.abandon_f)}, garçons {_n(p.abandon_g)}) ; "
            f"moyenne historique {_n(p.moyenne_historique)} % ; indice composite {_n(p.indice_risque)} "
            f"(conditions {_n(p.indice_conditions, 0)}, équipements {_n(p.indice_equipements, 0)}, "
            f"écarts filles-garçons {_n(p.indice_ecarts, 0)}, accès {_n(p.indice_acces, 0)}) ; profil « {p.profil} » ; "
            f"achèvement {_n(d.achevement_t)} % ; redoublement {_n(d.redoublement_t)} % ; "
            f"élèves par maître {_n(d.ratio_eleves_maitre)} ; écoles avec eau {_n(d.pct_ecoles_eau, 0)} %, "
            f"latrines {_n(d.pct_ecoles_latrines, 0)} %, électricité {_n(d.pct_ecoles_electricite, 0)} %, "
            f"cantine {_n(d.pct_ecoles_cantine, 0)} % ; évolution sur un an : effectifs {_n(d.var_effectif_pct)} %, "
            f"écoles {_n(d.var_ecoles_pct)} % ; taux de flux {'publiés' if p.flux_origine == 'publie' else 'estimés à partir des provinces'} ; "
            f"principaux facteurs SHAP (contribution au taux prévu) : {detail}.")

    validation = val.rename(columns={"modele": "Méthode", "erreur_absolue_moyenne": "Erreur moyenne (pts)",
                                     "erreur_absolue_mediane": "Erreur médiane (pts)", "rmse": "RMSE",
                                     "correlation_rang": "Corrélation de rang",
                                     "top4_retrouve": "Part des 4 régions les plus touchées retrouvées"})
    dimensions = "\n".join(f"  - {dim} : " + ", ".join(
        f"{r.libelle} ({'plus élevé = plus de risque' if r.sens == 1 else 'plus élevé = moins de risque'})"
        for r in g.itertuples()) for dim, g in definition.groupby("dimension", sort=False))
    groupes = "\n".join(f"- {r.profil} (indice moyen {_n(r.indice_risque, 0)}) : {r.regions}" for r in profils.itertuples())
    national = pd.DataFrame({"Année": nat.annee_scolaire, "Abandon": nat.abandon_t, "Abandon filles": nat.abandon_f,
                             "Abandon garçons": nat.abandon_g, "Redoublement": nat.redoublement_t,
                             "Achèvement": nat.achevement_t, "Élèves": nat.effectif, "Écoles": nat.ecoles})

    return f"""{CONSIGNES}

## Méthode
- Modèle prédictif : {meta['modele_retenu']}, entraîné sur {meta['n_apprentissage']} transitions région × année \
sans choc. Il prévoit le taux d'abandon moyen du primaire (CP1 à CM1, ensemble) de l'année suivante, **en l'absence \
de choc** (pas de fermetures ni de retours massifs d'élèves), à partir de la situation de l'année en cours. \
Données jusqu'en {an_don} ; prévision pour {an_prev}.
- Validation à origine glissante : pour chaque année cible de 2020-21 à 2024-25, entraînement sur les seules années \
antérieures. Sur les {meta.get('n_cas_valides', '')} transitions sans choc testées, erreur moyenne de \
{_n(meta['erreur_modele'], 2)} points, contre {_n(meta['erreur_persistance'], 2)} pour la persistance et \
{_n(meta['erreur_moyenne_historique'], 2)} pour la moyenne historique de la région. Sur les \
{meta.get('n_cas_chocs', '')} années de choc ou de retour, l'erreur atteint {_n(meta.get('erreur_modele_chocs'), 1)} \
points : les chocs ne sont pas prévisibles. Fourchette à 80 % : ± {_n(meta['demi_intervalle_80'])} points.
- Niveaux de risque (taux d'abandon prévu) : faible sous {_n(s1)} %, modéré de {_n(s1)} à {_n(s2)} %, élevé de \
{_n(s2)} à {_n(s3)} %, très élevé au-delà (quartiles des taux prévus par le modèle sur les années testées).
- Facteurs (SHAP) : contribution de chaque variable au taux prévu, à partir d'une base de {_n(meta['base_shap'], 2)} % \
(moyenne des prévisions). Ce sont des associations, pas des causes.
- Indice composite de risque (0 = favorable, 100 = défavorable), en quatre dimensions :
{dimensions}
- Profils de risque (regroupement des régions) :
{groupes}
- Limites : dans les régions touchées par l'insécurité, l'abandon mesure surtout des fermetures d'écoles (une baisse \
ne signifie pas que les élèves sont revenus) ; 13 régions sur 10 années ; taux de flux de 2015-16 à 2017-18 estimés \
à partir des provinces ; tout est calculé par région ; milieu urbain ou rural et pauvreté absents de la base. \
Sources : données du Data Challenge Burkina, annuaires statistiques du primaire (DGESS), INSD.

## Validation des méthodes (années cibles 2020-21 à 2024-25)
{_csv(validation, 2)}

## Fiches par région (de la plus prioritaire à la moins prioritaire)
{chr(10).join(fiches)}

## National, par année (%)
{_csv(national)}

## Taux d'abandon (%) par région et par année
{_annuel(reg, "abandon_t")}

## Taux d'achèvement du primaire (%) par région et par année
{_annuel(reg, "achevement_t")}

## Indice composite de risque par région et par année
{_annuel(reg, "indice_risque")}

## Nombre d'écoles primaires par région et par année
{_annuel(reg, "ecoles", 0)}
"""


# --- Appel du modèle ---------------------------------------------------------------------------

def repondre(config: Configuration, contexte_: str, conversation: list[dict]) -> Iterator[str]:
    """Réponse de Claude, morceau par morceau. Les erreurs sont converties en ErreurAssistant lisibles."""
    messages = [{"role": m["role"], "content": m["content"]} for m in conversation[-MESSAGES_GARDES:]]
    while messages and messages[0]["role"] != "user":  # la conversation transmise commence par une question
        messages = messages[1:]
    try:
        with _client(config.cle).beta.messages.stream(
            model=config.modele,
            max_tokens=MAX_TOKENS,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": EFFORT},
            system=[{"type": "text", "text": contexte_, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        ) as flux:
            yield from flux.text_stream
            final = flux.get_final_message()
    except anthropic.AuthenticationError as e:
        raise ErreurAssistant("La clé de l'API Anthropic est refusée : vérifiez ANTHROPIC_API_KEY dans le "
                              "fichier .env.") from e
    except anthropic.PermissionDeniedError as e:
        raise ErreurAssistant(f"La clé de l'API Anthropic n'a pas accès au modèle {config.modele}.") from e
    except anthropic.NotFoundError as e:
        raise ErreurAssistant(f"Modèle introuvable : {config.modele}. Vérifiez CLAUDE_MODEL dans le fichier .env.") from e
    except anthropic.RateLimitError as e:
        attente = e.response.headers.get("retry-after")
        delai = f"dans {int(float(attente)) + 1} secondes" if attente else "dans une minute"
        raise ErreurAssistant(f"L'assistant a atteint la limite de requêtes du compte Anthropic. Réessayez {delai}.") from e
    except anthropic.BadRequestError as e:
        raise ErreurAssistant(f"La requête a été refusée par l'API Anthropic (erreur 400) : {e.message}") from e
    except anthropic.APIStatusError as e:
        raise ErreurAssistant(f"Le service de Claude est momentanément indisponible (erreur {e.status_code}). "
                              "Réessayez dans un instant.") from e
    except anthropic.APIConnectionError as e:
        raise ErreurAssistant("Impossible de joindre l'API Anthropic : vérifiez la connexion Internet.") from e

    if final.stop_reason == "refusal":
        raise ErreurAssistant("Claude a refusé de répondre à cette question. Reformulez-la en restant sur les "
                              "résultats de la plateforme.")
    if final.stop_reason == "max_tokens":
        yield "\n\n*(Réponse interrompue : elle dépassait la longueur maximale.)*"


# --- Onglet ------------------------------------------------------------------------------------

def afficher() -> None:
    """Contenu de l'onglet « Assistant »."""
    st.subheader("Poser une question sur les résultats")
    st.caption("L'assistant répond en langage simple, uniquement à partir des résultats calculés par la plateforme, "
               "et cite les chiffres qu'il utilise. Il peut se tromper : vérifiez les chiffres importants dans les "
               "autres onglets.")
    config = configuration()
    if config is None:
        st.warning("L'assistant n'est pas configuré : ajoutez la variable ANTHROPIC_API_KEY dans un fichier .env à la "
                   "racine du dépôt, puis relancez la plateforme.")
        return

    st.session_state.setdefault("aide_conversation", [])
    conversation = st.session_state.aide_conversation

    def recommencer() -> None:
        st.session_state.aide_conversation = []
        st.session_state.aide_suggestion = None

    for message in conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Votre question sur les résultats…", key="aide_question")
    if not conversation:
        question = question or st.pills("Exemples de questions", SUGGESTIONS, key="aide_suggestion")

    if question:
        conversation.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            attente = st.empty()
            attente.caption("Claude analyse les résultats… (une dizaine de secondes)")

            def flux():
                for i, morceau in enumerate(repondre(config, contexte(), conversation)):
                    if i == 0:
                        attente.empty()
                    yield morceau

            try:
                reponse = st.write_stream(flux())
            except ErreurAssistant as e:
                reponse = None
                attente.empty()
                st.error(str(e))
        if reponse:
            conversation.append({"role": "assistant", "content": reponse})
        else:
            conversation.pop()  # la question sans réponse peut être reposée

    if conversation:
        st.button("Nouvelle conversation", on_click=recommencer, key="aide_recommencer")
