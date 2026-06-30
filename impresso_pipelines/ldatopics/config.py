SUPPORTED_LANGUAGES = {
    "de": "de_core_news_md",
    "en": "en_core_web_md",
    "fr": "fr_core_news_md",
    "lb": "lb_core_news_md", 
}

MODEL_URLS = {
    "de_core_news_md": "https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.6.0/de_core_news_md-3.6.0.tar.gz",
    "en_core_web_md": "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.6.0/en_core_web_md-3.6.0.tar.gz",
    "fr_core_news_md": "https://github.com/explosion/spacy-models/releases/download/fr_core_news_md-3.6.0/fr_core_news_md-3.6.0.tar.gz",
    "lb_core_news_md": "https://huggingface.co/impresso-project/lb-spacy-pos/resolve/main/lb_lux-tagger-July2023-3.6.0.tar.gz",
}

TOPIC_MODEL_DESCRIPTIONS = {
    "de": "https://huggingface.co/impresso-project/mallet-topic-inferencer/resolve/main/models/tm/tm-de-all-v3.0.topic_model_topic_description.jsonl.bz2",
    "en": "https://huggingface.co/impresso-project/mallet-topic-inferencer/resolve/main/models/tm/tm-en-all-v3.0.topic_model_topic_description.jsonl.bz2",
    "fr": "https://huggingface.co/impresso-project/mallet-topic-inferencer/resolve/main/models/tm/tm-fr-all-v3.0.topic_model_topic_description.jsonl.bz2",
    "lb": "https://huggingface.co/impresso-project/mallet-topic-inferencer/resolve/main/models/tm/tm-lb-all-v3.0.topic_model_topic_description.jsonl.bz2",
}

TOPIC_MODEL_DESCRIPTIONS_HF = {
    "de": ["impresso-project/mallet-topic-inferencer", "models/tm/tm-de-all-v3.0.topic_model_topic_description.jsonl.bz2"],
    "en": ["impresso-project/mallet-topic-inferencer", "models/tm/tm-en-all-v3.0.topic_model_topic_description.jsonl.bz2"],
    "fr": ["impresso-project/mallet-topic-inferencer", "models/tm/tm-fr-all-v3.0.topic_model_topic_description.jsonl.bz2"],
    "lb": ["impresso-project/mallet-topic-inferencer", "models/tm/tm-lb-all-v3.0.topic_model_topic_description.jsonl.bz2"],
}
