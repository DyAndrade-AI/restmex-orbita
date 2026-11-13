from restmex_recommender.text import expand_for_lexical, normalize_text, parse_query


def test_normalization_preserves_spanish_words():
    assert normalize_text("  Cabañas, café y NIÑOS  ") == "cabañas café y niños"


def test_negative_intent_is_separated():
    intent = parse_query("Quiero bosque y senderismo, sin playa ni antros")
    assert "bosque" in intent.positive
    assert "playa" in intent.negative
    assert "Attractive" in intent.preferred_types


def test_domain_expansion_adds_related_vocabulary():
    expanded = expand_for_lexical("Quiero snorkel en agua clara")
    assert "arrecife" in expanded
    assert "playa" in expanded



def test_leading_exclusion_stops_at_punctuation():
    intent = parse_query("Sin playa, quiero desierto y paisajes naturales")
    assert intent.negative == "playa"
    assert "desierto" in intent.positive and "playa" not in intent.positive
    assert parse_query("Sin Tulum pero con cenotes").negative == "tulum"


def test_non_excluding_sin_idioms_are_not_constraints():
    assert parse_query("Sin duda la mejor playa y comida rica").negative == ""
    assert parse_query("sin embargo el hotel es caro").negative == ""
    assert parse_query("Quiero bosque sin playa").negative == "playa"


def test_phonetic_key_catches_common_spanish_misspellings():
    from restmex_recommender.text import phonetic_key

    assert phonetic_key("plallas") == phonetic_key("playas")
    assert phonetic_key("cenotez") == phonetic_key("cenotes")
    # "snorkell" no coincide por clave (ll → y); lo resuelve el respaldo por similitud.
    assert phonetic_key("snorkell") != phonetic_key("snorkel")
