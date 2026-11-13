from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from restmex_recommender import Recommender  # noqa: E402

st.set_page_config(page_title="Órbita · Futuros de viaje", page_icon="🪐", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      .result-title {font-size: 1.15rem; font-weight: 700; margin-bottom: .1rem;}
      .role {display: inline-block; background: #ede9fe; color: #5b21b6; border-radius: 999px;
             padding: .2rem .65rem; font-size: .78rem; font-weight: 700; margin-bottom: .35rem;}
      .muted {color: #64748b; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando índices locales...")
def load_engine() -> Recommender:
    return Recommender(ROOT / "artifacts", ROOT / "models/multilingual-e5-base")


def render_result(item: dict) -> None:
    st.markdown(
        f"<div class='result-title'>#{item['global_rank']} de 40 · {item['name']}</div>"
        f"<div class='muted'>{item['region']}</div>",
        unsafe_allow_html=True,
    )
    score, quality, corpus = st.columns(3)
    score.metric("Afinidad relativa", f"{item['score']:.1f}/100")
    quality.metric("Calidad percibida", f"{item['quality_bayes']:.2f}/5", help="No influye en el orden")
    corpus.metric("Reseñas en train", f"{item['review_count']:,}")
    st.write(item["explanation"])
    for title, key in (("Satisface", "satisfied"), ("Respaldo débil", "weak"),
                       ("Sin evidencia", "unsupported")):
        if item[key]:
            st.caption(f"{title}: " + ", ".join(row["facet"] for row in item[key]))
    if item["violated"]:
        st.error("Restricciones incumplidas: " + ", ".join(
            f"{v['constraint']} ({v['rate']}% de sus reseñas)" for v in item["violated"]))
        if item.get("violation_evidence"):
            st.caption("Reseña: " + str(item["violation_evidence"]["Review"])[:240])
    if item.get("corpus_check"):
        st.caption(" · ".join(
            f"{row['term']}: {row['rate']}% de reseñas (mediana {row['median_rate']}%)"
            for row in item["corpus_check"]
        ))
    if item["least_related"]:
        st.write("Sus reseñas hablan sobre todo de:", ", ".join(item["keywords"]))
    elif item["matched_terms"]:
        st.write("Coincidencias:", ", ".join(item["matched_terms"]))
    else:
        st.write("Temas del destino:", ", ".join(item["keywords"]))
    evidence = item.get("evidence")
    if evidence:
        snippet = str(evidence["Review"]).strip()
        if len(snippet) > 250:
            snippet = snippet[:247].rsplit(" ", 1)[0] + "..."
        with st.expander("Evidencia tomada del conjunto de entrenamiento"):
            st.caption(f"{evidence['Type']} · polaridad {evidence['Polarity']}/5")
            st.write(snippet)
    st.divider()


def render_future(item: dict) -> None:
    st.markdown(f"<span class='role'>{item['role']}</span>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='result-title'>{item['name']}</div>"
        f"<div class='muted'>{item['region']} · {item['role_explanation']}</div>",
        unsafe_allow_html=True,
    )
    affinity, resilience, surprise = st.columns(3)
    affinity.metric("Afinidad", f"{item['affinity']:.1f}")
    resilience.metric("Resiliencia", f"{item['resilience']:.1f}")
    surprise.metric("Distancia del ancla", f"{item['distance_from_anchor']:.1f}")

    counterfactual = item["counterfactual"]
    shift = counterfactual["priority_shift"]
    if item["role"] == "Ancla":
        st.success(f"Eje dominante: **{item['dominant_facet']}**. Es la coincidencia directa.")
    elif shift is None:
        st.info(
            f"Su puerta de entrada es **{counterfactual['facet']}**. No intenta vencer al ancla "
            "por similitud pura: aporta un futuro distinto al conjunto."
        )
    elif shift == 0:
        st.info(f"Ya supera al ancla cuando se observa **{counterfactual['facet']}** por separado.")
    else:
        st.info(
            f"Contrafactual: al trasladar **{shift}%** de atención hacia "
            f"**{counterfactual['facet']}**, este destino puede superar al ancla."
        )

    if item["matched_terms"]:
        st.write("Huellas en el corpus:", ", ".join(item["matched_terms"]))
    else:
        st.write("Temas del destino:", ", ".join(item["keywords"]))
    evidence = item.get("evidence")
    if evidence:
        snippet = str(evidence["Review"]).strip()
        if len(snippet) > 250:
            snippet = snippet[:247].rsplit(" ", 1)[0] + "..."
        with st.expander("Ver evidencia y diagnóstico"):
            st.caption(
                f"Fragilidad de escenario: {item['fragility']:.1f} · "
                f"{evidence['Type']} · polaridad {evidence['Polarity']}/5"
            )
            st.write(snippet)
    st.divider()


try:
    engine = load_engine()
except FileNotFoundError:
    st.error("Faltan los índices. Ejecuta `make build-semantic` o `make build` antes de iniciar.")
    st.stop()

st.title("🪐 Órbita")
st.write(
    "No te entrega cinco copias de la misma respuesta. Construye cinco futuros de viaje: el más "
    "afín, el más resistente, uno que expande tus límites, un especialista y un comodín."
)

mode = st.radio(
    "Forma de explorar",
    ["Constelación de futuros", "Ranking clásico 5 + 5"],
    horizontal=True,
)
risk = 0.55
if mode == "Constelación de futuros":
    risk = st.slider(
        "Riesgo creativo",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="0 conserva similitud; 1 permite portales más lejanos, pero todavía plausibles.",
    )

examples = {
    "Playa y snorkel": "Quiero playas tranquilas, agua clara y actividades de snorkel",
    "Historia y gastronomía": "Busco arquitectura colonial, museos y comida tradicional",
    "Montaña sin playa": "Quiero naturaleza, montaña y senderismo, sin playa ni vida nocturna",
}
cols = st.columns(len(examples))
for column, (label, value) in zip(cols, examples.items()):
    if column.button(label, use_container_width=True):
        st.session_state["query"] = value

query = st.text_area(
    "¿Qué experiencia buscas?",
    key="query",
    height=110,
    placeholder="Ejemplo: Quiero un pueblo con bosque, senderos, cabañas y comida regional...",
)

if st.button("Abrir futuros", type="primary", use_container_width=True):
    try:
        with st.spinner("Simulando cambios de prioridad sobre 40 destinos..."):
            if mode == "Constelación de futuros":
                st.session_state["result"] = engine.recommend_futures(query, risk=risk)
                st.session_state["result_kind"] = "futures"
            else:
                st.session_state["result"] = engine.recommend(query)
                st.session_state["result_kind"] = "classic"
    except ValueError as exc:
        st.warning(str(exc))

result = st.session_state.get("result")
if result:
    if st.session_state.get("result_kind") == "futures":
        facet_labels = " · ".join(item["label"] for item in result["facets"])
        st.caption(
            f"Átomos de gusto detectados: {facet_labels} · "
            f"{result['method']['scenarios']} escenarios contrafactuales"
        )
        if result["excluded_concepts"]:
            st.warning(f"Frontera negativa activa: {result['excluded_concepts']}")
        for item in result["destinations"]:
            render_future(item)
    else:
        st.caption(f"Motor: {result['mode']} · confianza {result['confidence']['level']}: "
                   f"{result['confidence']['note']}")
        if result["excluded_concepts"]:
            st.info(f"Preferencias excluidas detectadas: {result['excluded_concepts']}")
        st.caption(result["score_note"])
        recommended, least = st.tabs(
            ["TOP 5 — MAYOR COMPATIBILIDAD", "BOTTOM 5 — MENOR COMPATIBILIDAD"]
        )
        with recommended:
            for item in result["top"]:
                render_result(item)
        with least:
            st.caption(
                "Los 5 peores para esta consulta, del menos compatible hacia arriba. "
                "No implica que tengan baja calidad."
            )
            for item in result["bottom"]:
                render_result(item)

with st.sidebar:
    st.subheader("Estado del sistema")
    st.success("Listo para trabajar sin conexión")
    st.write(f"Modo activo: **{'híbrido' if engine.semantic_enabled else 'léxico'}**")
    st.write("Capa experimental: **contrafactual + diversidad**")
    st.write("40 destinos · 19 regiones · 208,051 reseñas")
    st.caption("Afinidad y resiliencia son relativas a los 40 destinos para cada consulta.")

