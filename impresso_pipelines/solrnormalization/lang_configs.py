LANG_CONFIGS = {
    "de": {
        "stopwords_file": "stopwords_de.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "germanNormalization"},
            {"type": "tokenfilter", "name": "germanMinimalStem"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "snowball"
        }
    },
   "fr": {
        "stopwords_file": "stopwords_fr.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "elision"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "frenchMinimalStem"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "snowball"
        },
        "elision_params": {
            "ignoreCase": "true",
            "articles": "contractions_fr.txt"
        }
    },
    "es": {
        "stopwords_file": "stopwords_es.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "spanishLightStem"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "snowball"
        }
    },
    "pt": {
        "stopwords_file": "stopwords_pt.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "portugueseMinimalStem"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "snowball"
        }
    },
    "it": {
        "stopwords_file": "stopwords_it.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "elision"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "italianLightStem"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "snowball"
        },
        "elision_params": {
            "ignoreCase": "true",
            "articles": "contractions_it.txt"
        }
    },
    "en": {
        "stopwords_file": "stopwords_en.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "englishPossessive"},
            {"type": "tokenfilter", "name": "englishMinimalStem"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ],
        "stop_params": {
            "ignoreCase": "true"
        }
    },
    "nl": {
        "stopwords_file": "stopwords_nl.txt",
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "stop"},
            {"type": "tokenfilter", "name": "stemmerOverride"},
            {"type": "tokenfilter", "name": "snowballPorter"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ],
        "stop_params": {
            "ignoreCase": "true",
            "format": "snowball"
        },
        "stemmer_override_params": {
            "dictionary": "stemdict_nl.txt",
            "ignoreCase": "false"
        },
        "snowball_params": {
            "language": "Dutch"
        }
    },
    "general": {
        "analyzer_pipeline": [
            {"type": "tokenizer", "name": "standard"},
            {"type": "tokenfilter", "name": "lowercase"},
            {"type": "tokenfilter", "name": "asciifolding"}
        ]
    }
}