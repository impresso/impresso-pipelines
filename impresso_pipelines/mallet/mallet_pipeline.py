from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline
from impresso_pipelines.mallet.mallet_topic_inferencer import MalletTopicInferencer
import argparse
import json
import os
from huggingface_hub import hf_hub_url, hf_hub_download, list_repo_files  # Add list_repo_files import
import tempfile  # Add import for temporary directory
import shutil  # Add import for removing directories
import subprocess
import sys
import logging
try:
    import jpype
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jpype1"])
    import jpype



class MalletPipeline:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mallet_models_")  # Create temp folder for models
        self.temp_output_file = None  # Placeholder for temporary output file
        self.latest_model = None

        # Start JVM if not already running
        if not jpype.isJVMStarted():
            mallet_dir = self.setup_mallet_jars()  # Use Hugging Face caching
            # need to add mallet/lib since thats how it saves from hf_hub_download
            classpath = f"{mallet_dir}/mallet.jar:{mallet_dir}/mallet-deps.jar"
            
            # Start JVM with Mallet's classpath
            jpype.startJVM(jpype.getDefaultJVMPath(), f"-Djava.class.path={classpath}")

    
    def setup_mallet_jars(self):
        """
        Ensures that the Mallet JAR files are available locally using Hugging Face caching.

        Returns:
            str: Path to the directory containing the Mallet JAR files.
        """
        jar_files = ["mallet.jar", "mallet-deps.jar"]
        jar_paths = []

        for jar_name in jar_files:
            logging.info(f"Downloading {jar_name} from Hugging Face Hub...")
            jar_path = hf_hub_download(
                repo_id="impresso-project/mallet-topic-inferencer",
                filename=f"mallet/lib/{jar_name}"
            )
            jar_paths.append(jar_path)

        # Return the directory containing the first JAR file (all files are in the same directory)
        return os.path.dirname(jar_paths[0])


    def __call__(self, text, language=None, output_file=None):
        if output_file is None:
            self.temp_output_file = tempfile.NamedTemporaryFile(
                prefix="tmp_output_", suffix=".mallet", dir=self.temp_dir, delete=False
            )
            self.output_file = self.temp_output_file.name
        else:
            self.output_file = output_file

        # PART 1: Language Identification
        self.language = language
        if self.language is None:
            self.language_detection(text)

        from impresso_pipelines.mallet.config import SUPPORTED_LANGUAGES, TOPIC_MODEL_DESCRIPTIONS  # Lazy import
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES.keys()}")

        # Part 1.5: Find the latest model version
        self.find_latest_model_version()

        # PART 2: Lemmatization using SpaCy
        lemma_text = self.SPACY(text)

        # PART 3: Vectorization using Mallet
        self.vectorizer_mallet(lemma_text, self.output_file)

        # PART 4: Mallet inferencer and JSONification
        self.mallet_inferencer()

        # PART 5: Return the JSON output
        output = self.json_output(filepath=os.path.join(self.temp_dir, "tmp_output.jsonl"))

        # for each entry in the output list, add key "topic_model_description" with the value from the config file for the language
        for entry in output:
            entry["topic_model_description"] = TOPIC_MODEL_DESCRIPTIONS[self.language]
            

        return output  # Returns clean lemmatized text without punctuation
    
    def find_latest_model_version(self):
        """
        Finds the latest model version from the Hugging Face Hub.

        Returns:
            str: The latest model version.
        """
        repo_id = "impresso-project/mallet-topic-inferencer"
        files = list_repo_files(repo_id)
        versions = [f for f in files if f.startswith(f"models/tm/tm-{self.language}-all") and f.endswith(".pipe")] # check version of pipe 
        
        # Extract version numbers and find the latest one
        versions.sort(reverse=True)
        # extract the version number from the filename and set self.latest_model to the latest version
        if versions:
            self.latest_model = versions[0].split('-v')[-1].replace('.pipe', '')
        else:
            raise ValueError(f"Could not get latest version for language: {self.language}")

    def language_detection(self, text):
        lang_model = LangIdentPipeline()
        lang_result = lang_model(text)
        self.language = lang_result["language"]
        return self.language
    
    def SPACY(self, text):
        """Uses the appropriate SpaCy model based on language"""
        from impresso_pipelines.mallet.SPACY import SPACY  # Lazy import
        from impresso_pipelines.mallet.config import SUPPORTED_LANGUAGES  # Lazy import

        model_id = SUPPORTED_LANGUAGES[self.language]
        if not model_id:
            raise ValueError(f"No SpaCy model available for {self.language}")

        nlp = SPACY(model_id, self.language, self.latest_model)
        return nlp(text)

    def vectorizer_mallet(self, text, output_file):
        from impresso_pipelines.mallet.mallet_vectorizer_changed import MalletVectorizer  # Lazy import


        # Load the Mallet pipeline
        pipe_file = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=f"models/tm/tm-{self.language}-all-v{self.latest_model}.pipe"
        )


        
        mallet = MalletVectorizer(pipe_file, output_file)
        mallet(text)

    def mallet_inferencer(self):
        lang = self.language  # adjusting calling based on language


        inferencer_pipe = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=f"models/tm/tm-{lang}-all-v{self.latest_model}.pipe"
        )
        
        inferencer_file = hf_hub_download(  
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=f"models/tm/tm-{lang}-all-v{self.latest_model}.inferencer"
        )
      


        args = argparse.Namespace(
            input=self.output_file,  # Use the dynamically created output file
            input_format="jsonl",
            languages=[lang],
            output=os.path.join(self.temp_dir, "tmp_output.jsonl"),
            output_format="jsonl",
            **{
                f"{lang}_inferencer": inferencer_file,
                f"{lang}_pipe": inferencer_pipe,
                f"{lang}_model_id": f"tm-{lang}-all-v{self.latest_model}",
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
