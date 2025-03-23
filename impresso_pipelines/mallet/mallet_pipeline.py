from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline
from impresso_pipelines.mallet.SPACY import SPACY
from impresso_pipelines.mallet.config import SUPPORTED_LANGUAGES  # Import config
from impresso_pipelines.mallet.mallet_vectorizer_changed import MalletVectorizer 
from impresso_pipelines.mallet.mallet_topic_inferencer import MalletTopicInferencer
import argparse
import json
import os
from huggingface_hub import hf_hub_url, hf_hub_download, list_repo_files  # Add list_repo_files import
import tempfile  # Add import for temporary directory
import shutil  # Add import for removing directories



class MalletPipeline:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mallet_models_")  # Create temp folder for models
        pass

    def __call__(self, text, language=None, output_file="tmp_output.mallet"):
        try:
            self.output_file = output_file
            # PART 1: Language Identification
            self.language = language
            if self.language is None:
                self.language_detection(text)

            if self.language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES.keys()}")
            
            # Part 1.5: Download required files from huggingface model hub
            self.download_required_files()

            # PART 2: Lemmatization using SpaCy
            lemma_text = self.SPACY(text)

            # PART 3: Vectorization using Mallet
            self.vectorizer_mallet(lemma_text, output_file)

            
            # PART 4: Mallet inferencer and JSONification
            self.mallet_inferencer()


            # PART 5: Return the JSON output
            output = self.json_output(filepath="impresso_pipelines/mallet/tmp_output.jsonl")


            print(f"Lemma list: {lemma_text}")
            
                    

            return output  # Returns clean lemmatized text without punctuation
        finally:
            # Remove the temporary directory after execution
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Temporary directory {self.temp_dir} has been removed.")

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

    def download_required_files(self):
        """
        Downloads the required files for the specified language from the Hugging Face repository.
        Checks for the newest version available for the specified language.
        """
        repo_id = "impresso-project/mallet-topic-inferencer"
        base_path = "models/tm"
        
        # Fetch all files in the repository
        try:
            repo_files = list_repo_files(repo_id)
        except Exception as e:
            raise RuntimeError(f"Failed to list files in repository {repo_id}: {e}")

        # Filter files for the specified language and find the newest version
        language_files = [f for f in repo_files if f.startswith(f"{base_path}/tm-{self.language}-all-v")]

        if not language_files:
            raise FileNotFoundError(f"No files found for language {self.language} in repository {repo_id}")

        # Extract version numbers and sort to find the newest version
        language_files.sort(key=lambda x: int(x.split('-v')[-1].split('.')[0]), reverse=True)
        newest_version_files = [f for f in language_files if f.split('-v')[1].split('.')[0] == language_files[0].split('-v')[1].split('.')[0]]

        # Define the required files for the newest version
        files_to_download = [
            f for f in newest_version_files if any(ext in f for ext in [".pipe", ".inferencer", ".vocab.lemmatization.tsv.gz"])
        ]

        for file_path in files_to_download:
            try:
                file_name = os.path.basename(file_path)
                local_path = os.path.join(self.temp_dir, file_path)  # Include subdirectory in path
                print(f"Attempting to download {file_path} to {local_path}...")  # Debug log

                # Download the file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    local_dir=self.temp_dir,
                    local_dir_use_symlinks=False
                )

                # Verify the downloaded file path
                if not os.path.exists(downloaded_path):
                    raise FileNotFoundError(f"File {downloaded_path} was not downloaded correctly.")
                if not os.access(downloaded_path, os.R_OK):
                    raise PermissionError(f"File {downloaded_path} is not readable.")

                print(f"Successfully downloaded {file_name} to {downloaded_path}")
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")  # Log the error
                raise RuntimeError(f"Failed to download {file_path} from {repo_id}: {e}")

    def vectorizer_mallet(self, text, output_file):
        # Load the Mallet pipeline
        pipe_file = os.path.join(self.temp_dir, "models/tm", f"tm-{self.language}-all-v2.0.pipe")  # Adjust path
        
        # Verify the pipe file exists and is readable
        if not os.path.exists(pipe_file):
            raise FileNotFoundError(f"Pipe file not found: {pipe_file}")
        if not os.access(pipe_file, os.R_OK):
            raise PermissionError(f"Pipe file is not readable: {pipe_file}")
        
        mallet = MalletVectorizer(pipe_file, output_file)
        mallet(text)

    def mallet_inferencer(self):
        lang = self.language  # adjusting calling based on language

        inferencer_pipe = os.path.join(self.temp_dir, "models/tm", f"tm-{lang}-all-v2.0.pipe")  # Adjust path
        inferencer_file = os.path.join(self.temp_dir, "models/tm", f"tm-{lang}-all-v2.0.inferencer")  # Adjust path

        # Verify the inferencer files exist and are readable
        if not os.path.exists(inferencer_pipe):
            raise FileNotFoundError(f"Inferencer pipe file not found: {inferencer_pipe}")
        if not os.access(inferencer_pipe, os.R_OK):
            raise PermissionError(f"Inferencer pipe file is not readable: {inferencer_pipe}")
        if not os.path.exists(inferencer_file):
            raise FileNotFoundError(f"Inferencer file not found: {inferencer_file}")
        if not os.access(inferencer_file, os.R_OK):
            raise PermissionError(f"Inferencer file is not readable: {inferencer_file}")

        args = argparse.Namespace(
            input="impresso_pipelines/mallet/" + self.output_file,
            input_format="jsonl",
            languages=[lang],
            output="impresso_pipelines/mallet/tmp_output.jsonl",
            output_format="jsonl",
            **{
                f"{lang}_inferencer": inferencer_file,
                f"{lang}_pipe": inferencer_pipe,
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
