"""Normalización de texto y extracción de intención sin servicios externos."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List

SPANISH_STOP_WORDS = frozenset(
    """
    a al algo algunas algunos ante antes como con contra cual cuando de del desde donde durante e
    el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba
    estabais estaban estabas estad estada estadas estado estados estamos estando estar estara
    estaran estaras estare estareis estaremos estaria estarian estarias estas este esto estos fue
    fuera fueran fueras fueron fui fuimos ha habeis haber habia habian habias han hasta hay la las
    le les lo los mas me mi mis mucha muchas mucho muchos muy nada ni no nos nosotros o os otra
    otras otro otros para pero poco por porque que quien se sea sean seas ser sera seran seras seria
    si sin sobre sois somos son soy su sus te tenia tenian tiene tienen todo todos tu tus un una
    unas uno unos y ya lugar lugares vez aqui ahi asi tan tambien muy bueno buena buenos buenas
    excelente bien super realmente bastante siempre solo experiencia hacer hace hecho
    """.split()
)


# Palabras de relleno de una consulta ("quiero", "viaje"...) que no describen un destino.
QUERY_FILLER = frozenset(
    "quiero quisiera busco deseo deseamos interesa interesan gustaría gustaria viaje viajar "
    "pueblo destino experiencia plan planes vacaciones conocer encontrar".split()
)


TYPE_TERMS = {
    "Hotel": {
        "hotel", "hospedaje", "alojamiento", "habitacion", "habitaciones", "hostal",
        "cabaña", "cabanas", "resort", "dormir", "estancia", "posada", "spa",
    },
    "Restaurant": {
        "comer", "comida", "restaurante", "restaurantes", "gastronomia", "cenar",
        "desayuno", "platillo", "cocina", "cafe", "bar", "bebida", "mariscos",
    },
    "Attractive": {
        "visitar", "actividad", "actividades", "atraccion", "atracciones", "museo",
        "tour", "recorrido", "zona", "ruinas", "pirámides", "piramides", "naturaleza",
        "senderismo", "aventura", "playa", "cascada", "cenote", "arquitectura",
    },
}


# Expansión controlada para conceptos turísticos que suelen expresarse con vocabulario distinto
# entre una consulta breve y una reseña. No asigna destinos: solo agrega sinónimos al vector léxico.
CONCEPT_EXPANSIONS = {
    ("playa", "mar", "costa", "snorkel", "buceo", "arrecife", "surf"): (
        "playa mar costa arena oceano snorkel buceo arrecife surf"
    ),
    ("cenote", "cenotes"): "cenote cenotes caverna nadar agua dulce",
    ("pirámide", "piramide", "arqueología", "arqueologia", "prehispánico", "prehispanico"): (
        "piramide ruinas arqueologia prehispanico zona arqueologica templo"
    ),
    ("vino", "viñedo", "vinedo", "bodega"): "vino vinedo viñedo bodega cava degustacion",
    ("bosque", "montaña", "montana", "senderismo", "cabaña", "cabana"): (
        "bosque montana sierra senderismo caminata cabana naturaleza"
    ),
    ("colonial", "iglesia", "arquitectura"): (
        "colonial arquitectura iglesia catedral centro historico calles historia"
    ),
    ("cascada", "cascadas"): "cascada cascadas rio vegetacion selva aventura",
    ("termal", "termales", "spa"): "aguas termales balneario spa relajacion masaje",
    ("desierto", "desértico", "desertico"): "desierto dunas semidesierto paisaje arido",
    ("comida", "gastronomía", "gastronomia", "restaurante"): (
        "comida gastronomia cocina tradicional mercado restaurante platillo"
    ),
}


# Nombres para mostrar: el corpus trae los identificadores sin acentos ni espacios
# ("TodosSantos", "QuintanaRoo"). Solo cambia lo que se ve; los identificadores no se tocan.
DISPLAY_NAMES = {
    "Cuatro Cienegas": "Cuatro Ciénegas", "Patzcuaro": "Pátzcuaro", "Teotihuacan": "Teotihuacán",
    "Tepotzotlan": "Tepotzotlán", "Tepoztlan": "Tepoztlán", "TodosSantos": "Todos Santos",
    "Zacatlan": "Zacatlán", "San Cristobal de las Casas": "San Cristóbal de las Casas",
    "QuintanaRoo": "Quintana Roo", "Baja CaliforniaSur": "Baja California Sur",
    "Queretaro": "Querétaro", "Michoacan": "Michoacán", "Yucatan": "Yucatán",
    "Estado de Mexico": "Estado de México", "San Luis Potosi": "San Luis Potosí",
}


def display_label(value: str) -> str:
    return DISPLAY_NAMES.get(value, value)


# Locuciones con "sin" que no expresan exclusión ("sin duda la mejor playa" pide playa).
NON_EXCLUDING_SIN = (
    "duda", "dudas", "dudarlo", "dudado", "embargo", "palabras", "problema", "problemas",
    "igual", "falta", "pensarlo", "prisa", "pretensiones", "culpa", "medida", "querer",
    "exagerar", "importar", "parar", "fin", "comparacion", "comparación",
)

NEGATIVE_PATTERN = re.compile(
    r"(?:\b(?:sin(?!\s+(?:" + "|".join(NON_EXCLUDING_SIN) + r")\b)|evitar|evito|excepto)\b|\bno\s+(?:quiero|deseo|busco|me\s+gusta)\b)"
    # La exclusión termina en puntuación o en "pero": "sin playa, quiero bosque" no niega el bosque.
    r"\s+([^,.;:!?]+?)(?=\s+pero\b|[,.;:!?]|$)",
    flags=re.IGNORECASE,
)


def normalize_text(value: object) -> str:
    """Normaliza sin borrar palabras con valor semántico."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("_", " ")
    text = re.sub(r"[^\wáéíóúüñ\s-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def canonical_text(value: object) -> str:
    """Forma sin acentos para detectar entidades y restricciones."""
    text = normalize_text(value)
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    return text


def expand_for_lexical(value: str) -> str:
    """Agrega sinónimos de dominio cuando la consulta activa un concepto conocido."""
    normalized = normalize_text(value)
    tokens = set(canonical_text(normalized).split())
    expansions = [
        phrase
        for triggers, phrase in CONCEPT_EXPANSIONS.items()
        if tokens.intersection(canonical_text(" ".join(triggers)).split())
    ]
    return " ".join([normalized, *expansions]).strip()


@dataclass(frozen=True)
class QueryIntent:
    original: str
    positive: str
    negative: str
    preferred_types: List[str]


def parse_query(query: str) -> QueryIntent:
    original = normalize_text(query)
    # Las exclusiones se buscan antes de normalizar: normalize_text borra la puntuación que las
    # delimita, y entonces "sin" capturaría el resto de la consulta.
    raw = unicodedata.normalize("NFKC", str(query or "")).lower()
    negative_parts = [normalize_text(match.group(1)) for match in NEGATIVE_PATTERN.finditer(raw)]
    positive = normalize_text(NEGATIVE_PATTERN.sub(" , ", raw))
    # Sin fallback a la consulta original: si todo es exclusión, volver a ella invertiría la
    # intención ("sin playa" recomendaría playas). El motor rechaza positive vacío.
    positive = re.sub(r"\s+", " ", positive).strip(" ,.;")
    tokens = set(canonical_text(original).split())
    preferred_types = [
        kind
        for kind, terms in TYPE_TERMS.items()
        if tokens.intersection(canonical_text(" ".join(terms)).split())
    ]
    return QueryIntent(
        original=original,
        positive=positive,
        negative=" ".join(negative_parts),
        preferred_types=preferred_types,
    )



def positive_segments(query: str) -> List[str]:
    """Elementos coordinados de la parte positiva ("a, b y c al aire libre" → 3 segmentos).

    Se cortan en la puntuación ANTES de normalizar (normalize_text la borra) y en conjunciones;
    cada segmento es un sintagma: núcleo + modificadores ("gastronomía tradicional")."""
    raw = NEGATIVE_PATTERN.sub(" , ", unicodedata.normalize("NFKC", str(query or "")).lower())
    segments = []
    for part in re.split(r"[,.;:!?]", raw):
        for piece in re.split(r"\b(?:y|e|o|u|pero|ni|ademas|además|tambien|también)\b", part):
            piece = canonical_text(piece)
            if piece:
                segments.append(piece)
    return segments


def phonetic_key(word: str) -> str:
    """Clave fonética aproximada del español para corregir faltas comunes (plallas → playas)."""
    w = canonical_text(word)
    for a, b in (("qu", "k"), ("ll", "y"), ("ce", "se"), ("ci", "si"), ("z", "s"), ("v", "b"),
                 ("x", "j"), ("ge", "je"), ("gi", "ji"), ("h", "")):
        w = w.replace(a, b)
    out = []
    for ch in w:  # colapsa letras repetidas (snorkell → snorkel)
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)
