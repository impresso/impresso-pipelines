import spacy
import subprocess
import json
import gzip
from huggingface_hub import hf_hub_download

class SPACY:
    def __init__(self, model_id, language):
        # load spcay file
        from impresso_pipelines.mallet.config import MODEL_URLS  # Lazy import
        model_url = MODEL_URLS[model_id]
        if not model_url:
            raise ValueError(f"No SpaCy model available for {model_id}")
        
        # model_id = "https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.6.0/de_core_news_md-3.6.0.tar.gz"
            
        self.download_model(model_id)
        self.nlp = spacy.load(model_id)

        # load lemmatization files from hf
        # prepare and load lemmatization file and lower case it
        lemmatization_file = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=f"models/tm/tm-{language}-all-v2.0.vocab.lemmatization.tsv.gz"
        )
        # load the file, lower case the first column, make dict, first column key and third value
        self.lemmatization_dict = {}
        with gzip.open(lemmatization_file, "rt", encoding="utf-8") as f:
            for line in f:
                lemma = line.strip().split("\t")
                if len(lemma) > 2:
                    self.lemmatization_dict[lemma[0].lower()] = lemma[2]

 
        # load config file
        config_file = hf_hub_download(
            repo_id="impresso-project/lb-spacy-pos",
            filename=f"tm-{language}-all-v2.0.config.json"
        )
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.upos_filter = set(self.config.get("uposFilter", []))

        print(self.upos_filter)
        

    def download_model(self, model_id):
        """Ensures the SpaCy model is installed before use."""
        try:
            spacy.load(model_id)
        except OSError:
            print(f"Downloading SpaCy model: {model_id}...")
            subprocess.run(["python", "-m", "spacy", "download", model_id], check=True)

    def __call__(self, text):
        # download lemmatiazation files from hf

        # self.download_model(model_id)  # move to init (switch to specific link)
        # nlp = spacy.load(model_id) # move to init


        doc = self.nlp(text)

        # Remove punctuation and stopwords, and return lemmatized lowercase words
        # lemmatized_text = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop] # replace this filter with UPOS category is from config
        # and then use lemmas as filter from the lemma file
        # https://github.com/impresso/impresso-mallet-topic-inference/blob/15f80246ed7511d8fc4570b2dcb4d1978c59a59d/lib/multilingual_lemmatizer.py#L83

        # Filter tokens based on POS tags from config and lemmatize using the dictionary
        lemmatized_text = [
            self.lemmatization_dict.get(token.text.lower(), token.lemma_.lower())
            for token in doc
            if token.pos_ in self.upos_filter
        ]

        print(lemmatized_text)


        return lemmatized_text
