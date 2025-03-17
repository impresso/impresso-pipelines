from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline
from impresso_pipelines.mallet.SPACY import SPACY
from impresso_pipelines.mallet.config import SUPPORTED_LANGUAGES  # Import config
from impresso_pipelines.mallet.mallet_vectorizer import MalletVectorizer 

class MalletPipeline:
    def __init__(self):
        pass

    def __call__(self, text, language=None, output_file="output.mallet"):
        # PART 1: Language Identification
        self.language = language
        if self.language is None:
            self.language_detection(text)

        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES.keys()}")

        # PART 2: Lemmatization using SpaCy
        lemma_text = self.SPACY(text)

        # PART 3: Vectorization using Mallet
        self.vectorizer_mallet(lemma_text, output_file)

        print(f"Vectorized output saved to {output_file}")

        return "hey world"  # Returns clean lemmatized text without punctuation

    def language_detection(self, text):
        lang_model = LangIdentPipeline()
        lang_result = lang_model(text)
        self.language = lang_result["language"]
        return self.language
    
    def SPACY(self, text):
        """Uses the appropriate SpaCy model based on language"""
        model_id = SUPPORTED_LANGUAGES[self.language]
        if not model_id:
            raise ValueError(f"No SpaCy model available for {self.language}")

        nlp = SPACY()
        return nlp(text, model_id)


    def vectorizer_mallet(self, text, output_file):
        # Load the Mallet pipeline
        pipe_file = "impresso_pipelines/mallet/mallet_pipes/"+"tm-"+self.language+"-all-v2.0.pipe"
        mallet = MalletVectorizer(pipe_file, output_file)
        mallet(text)