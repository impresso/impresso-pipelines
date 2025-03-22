from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline
from impresso_pipelines.mallet.SPACY import SPACY
from impresso_pipelines.mallet.config import SUPPORTED_LANGUAGES  # Import config
from impresso_pipelines.mallet.mallet_vectorizer_changed import MalletVectorizer 
from impresso_pipelines.mallet.mallet_topic_inferencer import MalletTopicInferencer
import argparse
import json
import os


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


        # PART 5: Return the JSON output
        output = self.json_output(filepath="impresso_pipelines/mallet/tmp_output.jsonl")


        print(f"Lemma list: {lemma_text}")
        
                

        return output # Returns clean lemmatized text without punctuation

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
        lang = self.language  # adjusting calling based on language

        args = argparse.Namespace(
            input="impresso_pipelines/mallet/output.mallet",
            input_format="jsonl",
            languages=[lang],
            output="impresso_pipelines/mallet/tmp_output.jsonl",
            output_format="jsonl",
            **{
                f"{lang}_inferencer": f"impresso_pipelines/mallet/mallet_pipes/tm-{lang}-all-v2.0.inferencer",
                f"{lang}_pipe": f"impresso_pipelines/mallet/mallet_pipes/tm-{lang}-all-v2.0.pipe",
                f"{lang}_model_id": f"tm-{lang}-all-v2.0",
                f"{lang}_topic_count": 20
            },
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
            impresso_model_id=None,
        )

        inferencer = MalletTopicInferencer(args)

        inferencer.run()

    
    def json_output(self, filepath):
        """
        Reads a JSONL file and returns a list of parsed JSON objects.

        Parameters:
            filepath (str): Path to the .jsonl file.

        Returns:
            List[dict]: A list of dictionaries, one per JSONL line.
        """
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid line: {line}\nError: {e}")

        # delete the file after reading
        os.remove(filepath)

        return data