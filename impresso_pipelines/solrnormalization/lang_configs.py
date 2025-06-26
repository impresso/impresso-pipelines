LANG_CONFIGS = {
    "de": {
        "stopwords_file": "stopwords_de.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "germanNormalization"},
            {"type": "tokenfilter", "name": "asciifolding"},
            {"type": "tokenfilter", "name": "germanMinimalStem"},
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "wordset"
        }
    },
    "fr": {
        "stopwords_file": "stopwords_fr.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "elision"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "asciifolding"},
            {"type": "tokenfilter", "name": "frenchMinimalStem"},
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "wordset"
        },
        "elision_params": {
            "ignoreCase": "true"
        }
    },
    "el": {
        "stopwords_file": "stopwords_el.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "greekLowercase"},
            {"type": "tokenfilter", "name": "greekStem"},
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "wordset"
        }
    },
    "ru": {
        "stopwords_file": "stopwords_ru.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "russianLightStem"},
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "wordset"
        }
    }
    # Add more languages here as needed
}
