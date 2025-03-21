from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline
from impresso_pipelines.mallet.SPACY import SPACY
from impresso_pipelines.mallet.config import SUPPORTED_LANGUAGES  # Import config
from impresso_pipelines.mallet.mallet_vectorizer_changed import MalletVectorizer 
from impresso_pipelines.mallet.mallet_topic_inferencer import MalletTopicInferencer
import argparse


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

        # PART 4: Mallet inferencer and JSONification
        self.mallet_inferencer()


        
        
                

        return lemma_text # Returns clean lemmatized text without punctuation

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


    def mallet_inferencer(self):
        args = argparse.Namespace(
            input="impresso_pipelines/mallet/output.mallet",
            input_format="jsonl",
            languages=["de"],
            output="output.jsonl",
            output_format="jsonl",
            de_inferencer="impresso_pipelines/mallet/mallet_pipes/tm-de-all-v2.0.inferencer",
            de_pipe="impresso_pipelines/mallet/mallet_pipes/tm-de-all-v2.0.pipe",
            de_model_id="tm-de-all-v2.0",
            de_topic_count=100,
            min_p=0.02,
            keep_tmp_files=False,
            include_lid_path=False,
            inferencer_random_seed=42,
            quit_if_s3_output_exists=False,
            s3_output_dry_run=False,
            s3_output_path=None,
            git_version=None,
            lingproc_run_id=None,
            keep_timestamp_only=False,
            log_file=None,
            quiet=False,
            output_path_base=None,
            language_file=None,
            impresso_model_id=None,  # Add this default value
        )

        inferencer = MalletTopicInferencer(args)

        inferencer.run()
