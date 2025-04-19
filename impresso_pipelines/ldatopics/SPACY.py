import spacy
import subprocess
import json
import gzip
import os
import tarfile
import tempfile
import requests
import shutil  # Add this import for moving directories
from huggingface_hub import hf_hub_download

class SPACY:
    def __init__(self, model_id, language, latest_version):
        # load spcay file
        from impresso_pipelines.ldatopics.config import MODEL_URLS  # Lazy import
        model_url = MODEL_URLS[model_id]
        if not model_url:
            raise ValueError(f"No SpaCy model available for {model_id}")
        
        path_to_model = self.download_and_extract_model(model_url)
        self.nlp = spacy.load(path_to_model, disable=["parser", "ner"])

        # load lemmatization files from hf
        # prepare and load lemmatization file and lower case it
        lemmatization_file = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=f"models/tm/tm-{language}-all-v{latest_version}.vocab.lemmatization.tsv.gz"
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
            filename=f"tm-{language}-all-v{latest_version}.config.json"
        )
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.upos_filter = set(self.config.get("uposFilter", []))
        
        # print(self.upos_filter)
        

    def download_model(self, model_id):
        """Ensures the SpaCy model is installed before use."""
        try:
            spacy.load(model_id)
        except OSError:
            print(f"Downloading SpaCy model: {model_id}...")
            subprocess.run(["python", "-m", "spacy", "download", model_id], check=True)

    def download_and_extract_model(self, model_url):
        """Downloads and extracts the SpaCy model tar file to a cache directory."""
        cache_dir = os.path.expanduser("~/.cache/spacy_models")
        os.makedirs(cache_dir, exist_ok=True)

        # Generate a unique filename for the model based on its URL
        model_filename = os.path.basename(model_url)
        cached_model_path = os.path.join(cache_dir, model_filename)

        # Check if the model is already cached
        if os.path.exists(cached_model_path):
            # print(f"Using cached SpaCy model from: {cached_model_path}")
            print(f"Using cached SpaCy model...")
        else:
            # Download the tar file
            print(f"Downloading SpaCy model from: {model_url}...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(cached_model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract the tar file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        # print(f"Extracting SpaCy model to: {temp_dir}...")
        with tarfile.open(cached_model_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Locate the directory containing the config.cfg file
        for root, dirs, files in os.walk(temp_dir):
            if "config.cfg" in files:
                # Move the model directory to a new location and return its path
                model_dir = root
                final_model_dir = os.path.join(temp_dir, "model")
                shutil.move(model_dir, final_model_dir)
                return final_model_dir

        raise IOError("Could not find config.cfg in the extracted model directory.")

    def __call__(self, text):
        # download lemmatiazation files from hf


        doc = self.nlp(text)

        # lemmatized_text = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop] # replace this filter with UPOS category is from config
        # and then use lemmas as filter from the lemma file
        # https://github.com/impresso/impresso-mallet-topic-inference/blob/15f80246ed7511d8fc4570b2dcb4d1978c59a59d/lib/multilingual_lemmatizer.py#L83

        # for token in doc:
        #     print(f"Token: {token.text}, POS: {token.pos_}, Tag: {token.tag_}")
        # Filter tokens based on POS tags from config and lemmatize using the dictionary
        lemmatized_text = [
            self.lemmatization_dict.get(token.text.lower(), token.lemma_.lower())
            for token in doc
            if (token.pos_ or self.map_tag_to_pos(token.tag_)) in self.upos_filter
        ]

        # print("French lemmatized tokens:", lemmatized_text)

        return lemmatized_text

    def map_tag_to_pos(self, tag):
        # Map the fine-grained tags used by your Luxembourgish model to Universal POS tags
        tag_map = {
            "$": "PUNCT",
            "ADJ": "ADJ",
            "AV": "ADV",
            "APPR": "ADP",
            "APPRART": "ADP",
            "D": "DET",
            "KO": "CONJ",
            "N": "NOUN",
            "P": "ADV",
            "TRUNC": "X",
            "AUX": "AUX",
            "V": "VERB",
            "MV": "VERB",
            "PTK": "PART",
            "INTER": "PART",
            "NUM": "NUM",
            "_SP": "SPACE",
        }
        return tag_map.get(tag, "")
