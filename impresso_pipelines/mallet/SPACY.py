import spacy
import subprocess

class SPACY:
    def __init__(self):
        pass

    def download_model(self, model_id):
        """Ensures the SpaCy model is installed before use."""
        try:
            spacy.load(model_id)
        except OSError:
            print(f"Downloading SpaCy model: {model_id}...")
            subprocess.run(["python", "-m", "spacy", "download", model_id], check=True)

    def __call__(self, text, model_id):
        self.download_model(model_id)  # Ensure the model is installed
        nlp = spacy.load(model_id)
        doc = nlp(text)

        # Remove punctuation and stopwords, and return lemmatized lowercase words
        lemmatized_text = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
        

        return lemmatized_text
