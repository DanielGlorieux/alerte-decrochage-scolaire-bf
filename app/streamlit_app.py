"""
Plateforme d'alerte précoce du décrochage scolaire au primaire - Burkina Faso
Equipe Insighters

Lancement :  streamlit run app/streamlit_app.py
Données   :  data/modele/ (produites par les scripts 04 et 05)
"""
from io import BytesIO
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

M = Path(__file__).resolve().parents[1] / "data/modele"


BLEU, BLEU_CLAIR = "#2a78d6", "#9ec5f4"
HAUSSE, BAISSE, NEUTRE = "#e34948", "#2a78d6", "#898781"
ECHELLE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
NIVEAUX = {"faible": ("#0ca30c", "●"), "modéré": ("#fab219", "▲"), "élevé": ("#ec835a", "◆"), "très élevé": ("#d03b3b", "■")}
GRILLE, ENCRE = "#e1e0d9", "#52514e"
DECALAGES = {"PLATEAU CENTRAL": (0.25, 0.0)}
ETIQUETTES = {"BOUCLE DU MOUHOUN": "Boucle du<br>Mouhoun", "PLATEAU CENTRAL": "Plateau<br>Central",
              "HAUTS-BASSINS": "Hauts-<br>Bassins", "CENTRE-OUEST": "Centre-<br>Ouest", "CENTRE-SUD": "Centre-<br>Sud",
              "CENTRE-NORD": "Centre-<br>Nord", "CENTRE-EST": "Centre-<br>Est"}

st.set_page_config(page_title="Alerte décrochage scolaire - Burkina Faso", layout="wide")


@st.cache_data
def charger():
    ind = pd.read_parquet(M / "indicateurs.parquet")
    prev = pd.read_csv(M / "previsions.csv")
    shap = pd.read_csv(M / "shap.csv")
    val = pd.read_csv(M / "validation_modeles.csv")
    valp = pd.read_csv(M / "validation_predictions.csv")
    profils = pd.read_csv(M / "profils.csv")
    meta = json.loads((M / "meta.json").read_text(encoding="utf8"))
    geo = json.loads((M / "regions.geojson").read_text(encoding="utf8"))
    definition = pd.read_csv(M / "indice_definition.csv")
    return ind, prev, shap, val, valp, profils, meta, geo, definition


@st.cache_resource
def charger_modele():
    return joblib.load(M / "modele.joblib")


ind, prev, shap, val, valp, profils, meta, geo, definition = charger()
modele = charger_modele()
VARS = meta["variables"]
regions = ind[ind.code_region != "BF"]
AN_PREVU, AN_DONNEES = meta["annee_prevue"], meta["annee_donnees"]


def mise_en_forme(fig, hauteur=380, titre_y=None, titre_x=None):
    fig.update_layout(height=hauteur, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13), hoverlabel=dict(font_size=13),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(showgrid=False, title=titre_x, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRILLE, title=titre_y, zeroline=False)
    return fig


def pastille(niveau):
    couleur, icone = NIVEAUX[niveau]
    return f"<span style='color:{couleur};font-size:1.1em'>{icone}</span> {niveau}"


st.title("Alerte précoce du décrochage scolaire au primaire")
st.caption(f"Burkina Faso, 13 régions. Données {ind.annee_scolaire.min()} à {AN_DONNEES} ; risque estimé pour {AN_PREVU}. "
           "Sources : données du Data Challenge Burkina, annuaires statistiques de l'enseignement primaire (DGESS).")

carte, fiche, priorites, simulateur, methode = st.tabs(
    ["Carte des risques", "Fiche par région", "Régions prioritaires", "Simulateur", "Méthode et limites"])

# =========================================================================================
# 1. Carte
# =========================================================================================
with carte:
    choix = {
        f"Risque estimé pour {AN_PREVU} (taux d'abandon prévu, %)": ("prev", "risque_prevu"),
        "Indice composite de risque (0 à 100)": ("ind", "indice_risque"),
        "Taux d'abandon observé (%)": ("ind", "abandon_t"),
        "Taux d'achèvement du primaire (%)": ("ind", "achevement_t"),
        "Élèves par maître": ("ind", "ratio_eleves_maitre"),
        "Écoles avec eau potable (%)": ("ind", "pct_ecoles_eau"),
        "Variation annuelle du nombre d'écoles (%)": ("ind", "var_ecoles_pct"),
    }
    g, dte = st.columns([3, 1])
    libelle = g.selectbox("Indicateur affiché", list(choix))
    source, col = choix[libelle]
    if source == "prev":
        donnees, annee = prev, AN_PREVU
        dte.markdown(f"**Année** : {AN_PREVU} (prévision)")
    else:
        annees = sorted(regions.annee_scolaire.unique())
        annee = dte.select_slider("Année scolaire", annees, value=annees[-1])
        donnees = regions[regions.annee_scolaire == annee]
    donnees = donnees.dropna(subset=[col])
    gauche, droite = st.columns([3, 2])
    with gauche:
        fig = go.Figure(go.Choropleth(
            geojson=geo, featureidkey="properties.code_region", locations=donnees.code_region, z=donnees[col],
            text=donnees.region.str.title(), colorscale=[[i / 4, c] for i, c in enumerate(ECHELLE)],
            marker_line_color="#fcfcfb", marker_line_width=2, colorbar=dict(title="", thickness=12, len=0.7),
            hovertemplate="<b>%{text}</b><br>%{z:.1f}<extra></extra>"))
        # étiquettes : nom de la région et valeur, encre claire sur les teintes foncées
        et = pd.DataFrame([f["properties"] for f in geo["features"]]).merge(donnees[["code_region", col]], on="code_region")
        et["lon_etiquette"] += et.region.map(lambda r: DECALAGES.get(r, (0, 0))[0])
        et["lat_etiquette"] += et.region.map(lambda r: DECALAGES.get(r, (0, 0))[1])
        etendue = (et[col].max() - et[col].min()) or 1
        et["fonce"] = (et[col] - et[col].min()) / etendue > 0.5
        et["texte"] = [ETIQUETTES.get(r, r.title()) + f"<br>{v:.1f}" for r, v in zip(et.region, et[col])]
        for fonce, encre in [(False, "#0b0b0b"), (True, "#ffffff")]:
            e = et[et.fonce == fonce]
            fig.add_scattergeo(lon=e.lon_etiquette, lat=e.lat_etiquette, text=e.texte, mode="text",
                               textfont=dict(size=10, color=encre), hoverinfo="skip", showlegend=False)
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Teinte plus foncée = valeur plus élevée (pas forcément plus défavorable : voir l'indicateur choisi).")
    with droite:
        tri = donnees.sort_values(col)
        fig = go.Figure(go.Bar(x=tri[col], y=tri.region.str.title(), orientation="h", marker_color=BLEU,
                               text=tri[col].round(1), textposition="outside", cliponaxis=False,
                               hovertemplate="<b>%{y}</b> : %{x:.1f}<extra></extra>"))
        mise_en_forme(fig, 470)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
    with st.expander("Voir les valeurs en tableau"):
        st.dataframe(donnees[["region", col]].sort_values(col, ascending=False).rename(columns={"region": "Région", col: libelle}),
                     hide_index=True, width="stretch")

# =========================================================================================
# 2. Fiche par région
# =========================================================================================
with fiche:
    nom = st.selectbox("Région", prev.sort_values("rang_priorite").region, format_func=str.title)
    p = prev[prev.region == nom].iloc[0]
    h = regions[regions.region == nom].sort_values("annee_fin")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Taux d'abandon prévu {AN_PREVU}", f"{p.risque_prevu:.1f} %",
              f"{p.risque_prevu - p.abandon_t:+.1f} pt par rapport à {AN_DONNEES}", delta_color="inverse")
    c2.markdown(f"**Niveau de risque**<br><span style='font-size:1.6em'>{pastille(p.niveau_risque)}</span>", unsafe_allow_html=True)
    c3.metric("Rang de priorité", f"{p.rang_priorite} sur 13")
    c4.metric("Confiance dans l'estimation", p.confiance,
              help=f"Selon l'erreur du modèle sur cette région lors des tests passés ({p.erreur_passee:.1f} points en moyenne).")
    st.markdown(f"Fourchette probable : **{p.borne_basse:.1f} % à {p.borne_haute:.1f} %** (8 fois sur 10 lors des tests). "
                f"Profil : **{p.profil}**.")
    if p.flux_origine != "publie":
        st.info("Les taux d'abandon récents de cette région sont estimés à partir des provinces.")

    g, d = st.columns(2)
    with g:
        st.subheader("Évolution du taux d'abandon")
        fig = go.Figure()
        for s, lib, coul, trait in [("abandon_g", "Garçons", BLEU, "solid"), ("abandon_f", "Filles", "#eb6834", "solid"),
                                    ("abandon_t", "Ensemble", "#0b0b0b", "dot")]:
            fig.add_scatter(x=h.annee_scolaire, y=h[s], name=lib, mode="lines+markers", line=dict(color=coul, width=2, dash=trait),
                            marker=dict(size=8), hovertemplate=lib + " : %{y:.1f} %<extra></extra>")
        fig.add_scatter(x=[AN_PREVU], y=[p.risque_prevu], name="Prévision", mode="markers",
                        marker=dict(size=13, color="#0b0b0b", symbol="diamond-open", line=dict(width=2)),
                        error_y=dict(type="data", symmetric=False, array=[p.borne_haute - p.risque_prevu],
                                     arrayminus=[p.risque_prevu - p.borne_basse], color=NEUTRE, thickness=1.5),
                        hovertemplate="Prévision : %{y:.1f} %<extra></extra>")
        mise_en_forme(fig, 360, "Taux d'abandon (%)").update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.caption("Les années de forte hausse correspondent surtout aux fermetures d'écoles liées à l'insécurité, "
                   "pas uniquement à des élèves qui décrochent d'une école restée ouverte.")
    with d:
        st.subheader("Ce qui pèse sur le risque estimé")
        s = shap[shap.region == nom].assign(a=lambda x: x.contribution.abs()).nlargest(8, "a").sort_values("contribution")
        fig = go.Figure(go.Bar(
            x=s.contribution, y=[f"{l} = {v:.1f}" for l, v in zip(s.libelle, s.valeur)], orientation="h",
            marker_color=[HAUSSE if c > 0 else BAISSE for c in s.contribution],
            text=[f"{c:+.1f}" for c in s.contribution], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>contribution : %{x:+.2f} point<extra></extra>"))
        mise_en_forme(fig, 360, titre_x="Contribution au taux prévu (points)")
        fig.add_vline(x=0, line_color="#c3c2b7")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Rouge : augmente le risque estimé ; bleu : le réduit. Point de départ : {meta['base_shap']:.1f} %, "
                   "moyenne des prévisions du modèle. Méthode SHAP : ce sont des associations, pas des causes démontrées.")

    st.subheader("Indice composite par dimension")
    dims = {"indice_conditions": "Conditions d'enseignement", "indice_equipements": "Équipements",
            "indice_ecarts": "Écarts filles-garçons", "indice_acces": "Accès et continuité"}
    dern = regions[regions.annee_fin == regions.annee_fin.max()]
    fig = go.Figure()
    fig.add_bar(x=list(dims.values()), y=[p[k] for k in dims], name=nom.title(), marker_color=BLEU,
                text=[f"{p[k]:.0f}" for k in dims], textposition="outside", hovertemplate="%{x} : %{y:.0f}<extra></extra>")
    fig.add_bar(x=list(dims.values()), y=[dern[k].mean() for k in dims], name="Moyenne des 13 régions", marker_color=BLEU_CLAIR,
                text=[f"{dern[k].mean():.0f}" for k in dims], textposition="outside", hovertemplate="%{x} : %{y:.0f}<extra></extra>")
    mise_en_forme(fig, 300, "Indice (0 = favorable, 100 = défavorable)").update_layout(bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 115])
    st.plotly_chart(fig, width="stretch")

    cols_fiche = {"annee_scolaire": "Année", "abandon_t": "Abandon (%)", "redoublement_t": "Redoublement (%)",
                  "achevement_t": "Achèvement (%)", "var_effectif_pct": "Var. effectifs (%)", "var_ecoles_pct": "Var. écoles (%)",
                  "ratio_eleves_maitre": "Élèves/maître", "eleves_par_salle": "Élèves/salle", "pct_ecoles_eau": "Eau (%)",
                  "pct_ecoles_latrines": "Latrines (%)", "pct_ecoles_cantine": "Cantine (%)", "indice_risque": "Indice"}
    with st.expander("Voir toutes les années en tableau"):
        st.dataframe(h[list(cols_fiche)].rename(columns=cols_fiche), hide_index=True, width="stretch")

# =========================================================================================
# 3. Régions prioritaires (export)
# =========================================================================================
COLS_EXPORT = {"rang_priorite": "Rang", "region": "Région", "risque_prevu": f"Abandon prévu {AN_PREVU} (%)",
               "borne_basse": "Fourchette basse", "borne_haute": "Fourchette haute", "niveau_risque": "Niveau de risque",
               "confiance": "Confiance", "abandon_t": f"Abandon observé {AN_DONNEES} (%)", "indice_risque": "Indice composite",
               "profil": "Profil", "facteurs_principaux": "Principaux facteurs"}


def export_excel(df):
    tampon = BytesIO()
    with pd.ExcelWriter(tampon, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Priorites")
        val.to_excel(w, index=False, sheet_name="Validation")
    return tampon.getvalue()


def export_pdf(df):
    latin = lambda t: str(t).encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF(orientation="L")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, latin(f"Régions prioritaires - risque de décrochage au primaire, {AN_PREVU}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, latin(f"Taux d'abandon prévu par le modèle ({meta['modele_retenu']}) à partir des données {AN_DONNEES}. "
                               f"Fourchette à 80 % : plus ou moins {meta['demi_intervalle_80']} points. "
                               "Ces résultats servent à cibler les actions, pas à juger une région."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    cols = ["Rang", "Région", f"Abandon prévu {AN_PREVU} (%)", "Niveau de risque", "Confiance", "Profil", "Principaux facteurs"]
    larg = [12, 40, 30, 28, 22, 62, 83]
    pdf.set_font("Helvetica", "B", 8)
    for c, l in zip(["Rang", "Région", "Abandon prévu (%)", "Niveau", "Confiance", "Profil", "Principaux facteurs"], larg):
        pdf.cell(l, 7, latin(c), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for _, r in df[cols].iterrows():
        for c, l in zip(cols, larg):
            pdf.cell(l, 7, latin(r[c])[: int(l / 1.35)], border=1)
        pdf.ln()
    return bytes(pdf.output())


with priorites:
    st.subheader(f"Classement des régions selon le risque estimé pour {AN_PREVU}")
    t = prev.sort_values("rang_priorite")[list(COLS_EXPORT)].rename(columns=COLS_EXPORT)
    t["Région"] = t["Région"].str.title()
    lignes = "".join(
        f"<tr><td>{r.Rang}</td><td><b>{r['Région']}</b></td><td style='text-align:right'>{r.iloc[2]:.1f}</td>"
        f"<td style='text-align:right'>{r['Fourchette basse']:.1f} à {r['Fourchette haute']:.1f}</td>"
        f"<td>{pastille(r['Niveau de risque'])}</td><td>{r['Confiance']}</td><td>{r['Profil']}</td>"
        f"<td style='font-size:0.85em'>{r['Principaux facteurs']}</td></tr>" for _, r in t.iterrows())
    st.markdown("<table style='width:100%'><tr><th>Rang</th><th>Région</th><th>Abandon prévu (%)</th><th>Fourchette</th>"
                f"<th>Niveau</th><th>Confiance</th><th>Profil</th><th>Principaux facteurs de hausse</th></tr>{lignes}</table>",
                unsafe_allow_html=True)
    s1, s2, s3 = meta["seuils_niveaux"]
    st.caption(f"Niveaux : faible sous {s1} %, modéré de {s1} à {s2} %, élevé de {s2} à {s3} %, très élevé au-delà "
               "(quartiles des taux d'abandon régionaux observés depuis 2016-17).")
    b1, b2, _ = st.columns([1, 1, 4])
    b1.download_button("Exporter en Excel", export_excel(t), "regions_prioritaires.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    b2.download_button("Exporter en PDF", export_pdf(t), "regions_prioritaires.pdf", "application/pdf")

    st.subheader("Profils de risque")
    for _, r in profils.iterrows():
        st.markdown(f"**{r.profil}** (indice moyen {r.indice_risque:.0f}) : {r.regions.title()}")

# =========================================================================================
# 4. Simulateur
# =========================================================================================
LEVIERS = {"ratio_eleves_maitre": (20.0, 90.0, 1.0), "eleves_par_salle": (20.0, 90.0, 1.0),
           "pct_ecoles_eau": (0.0, 100.0, 1.0), "pct_ecoles_latrines": (0.0, 100.0, 1.0),
           "pct_ecoles_cantine": (0.0, 100.0, 1.0), "pct_ecoles_electricite": (0.0, 100.0, 1.0)}

with simulateur:
    st.subheader("Que devient le risque estimé si les conditions changent ?")
    nom_s = st.selectbox("Région à simuler", prev.sort_values("rang_priorite").region, format_func=str.title, key="sim")
    ligne = regions[(regions.region == nom_s) & (regions.annee_fin == regions.annee_fin.max())].iloc[0].copy()
    ligne["moyenne_historique"] = prev[prev.region == nom_s].moyenne_historique.iloc[0]
    x0 = pd.DataFrame([ligne[list(VARS)].astype(float)])
    x1 = x0.copy()
    cg, cd = st.columns(2)
    for i, (v, (mn, mx, pas)) in enumerate(LEVIERS.items()):
        actuel = float(x0[v].iloc[0])
        x1[v] = (cg if i % 2 == 0 else cd).slider(f"{VARS[v]} (actuel : {actuel:.1f})", mn, mx, float(min(max(round(actuel, 1), mn), mx)), 0.1, key=f"{nom_s}_{v}")
    avant, apres = float(modele.predict(x0)[0]), float(modele.predict(x1)[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Taux d'abandon prévu, situation actuelle", f"{avant:.1f} %")
    m2.metric("Taux d'abandon prévu, scénario", f"{apres:.1f} %", f"{apres - avant:+.1f} point", delta_color="inverse")
    m3.metric("Élèves concernés (ordre de grandeur)", f"{abs(apres - avant) / 100 * ligne.effectif:,.0f}".replace(",", " "),
              help="Écart de taux appliqué à l'effectif actuel du primaire de la région.")
    st.warning("À lire comme une indication, pas comme une promesse d'effet. Le modèle repère des associations entre "
               "régions et années ; il ne prouve pas qu'améliorer un facteur fera baisser l'abandon d'autant. "
               "Il ne dispose que de 117 observations et ses erreurs passées sont larges (voir Méthode et limites).")

# =========================================================================================
# 5. Méthode et limites
# =========================================================================================
with methode:
    st.subheader("Le modèle a-t-il bien anticipé le passé ?")
    st.markdown(
        f"Pour chaque année de 2020-21 à 2024-25, le modèle a été entraîné uniquement sur les années précédentes, puis "
        f"comparé à ce qui s'est produit. Modèle retenu : **{meta['modele_retenu']}**, sur {meta['n_apprentissage']} observations.")
        
    v = val.rename(columns={"modele": "Méthode", "erreur_absolue_moyenne": "Erreur moyenne (points)",
                            "erreur_absolue_mediane": "Erreur médiane (points)", "rmse": "RMSE",
                            "correlation_rang": "Corrélation de rang", "top4_retrouve": "Part des 4 régions les plus touchées retrouvées"})
    st.dataframe(v, hide_index=True, width="stretch")

    st.subheader("Prévu contre observé, par année testée")
    nom_m = meta["modele_retenu"]
    an = st.select_slider("Année cible", sorted(valp.annee_cible.unique()), format_func=lambda a: f"{a - 1}-{a}")
    w = valp[valp.annee_cible == an].sort_values("abandon_suivant")
    fig = go.Figure()
    fig.add_bar(y=w.region.str.title(), x=w.abandon_suivant, orientation="h", name="Observé", marker_color=BLEU,
                hovertemplate="Observé : %{x:.1f} %<extra></extra>")
    fig.add_scatter(y=w.region.str.title(), x=w[nom_m], mode="markers", name="Prévu par le modèle",
                    marker=dict(size=11, color="#0b0b0b", symbol="diamond-open", line=dict(width=2)),
                    hovertemplate="Prévu : %{x:.1f} %<extra></extra>")
    mise_en_forme(fig, 430, titre_x="Taux d'abandon (%)").update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Composition de l'indice composite")
    st.dataframe(definition.assign(sens=definition.sens.map({1: "plus élevé = plus de risque", -1: "plus élevé = moins de risque"}))
                 [["dimension", "libelle", "sens"]].rename(columns={"dimension": "Dimension", "libelle": "Facteur", "sens": "Lecture"}),
                 hide_index=True, width="stretch")

    st.subheader("Limites")
    st.markdown("""
- **Abandon et insécurité.** Dans les régions touchées, le taux d'abandon mesure surtout des fermetures d'écoles. Un taux qui retombe l'année suivante ne signifie pas que la situation est réglée.
- **Peu de données.** 13 régions et 10 années : les résultats sont des ordres de grandeur. Les fourchettes affichées sont larges pour cette raison.
- **Années 2015-16 à 2017-18.** Les taux régionaux de promotion, redoublement et abandon sont estimés à partir des provinces (écart médian de 0,1 point, plus rarement jusqu'à 5 points).
- **Niveau géographique.** Tout est calculé par région. Une région peu risquée en moyenne peut contenir des provinces en grande difficulté.
- **Milieu urbain ou rural, pauvreté, écoles fermées.** Ces facteurs ne sont pas encore dans la base au niveau régional.
- **Usage.** Ces résultats servent à cibler des actions, jamais à stigmatiser une région.
""")
