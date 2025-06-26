LANG_CONFIGS = {
    "de": {
        "stopwords_file": "german_stop.txt",
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
        "stopwords_file": "french_stop.txt",
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
    }
    # Add more languages here as needed
}
