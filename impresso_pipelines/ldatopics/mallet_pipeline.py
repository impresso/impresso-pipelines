from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline
from impresso_pipelines.ldatopics.mallet_topic_inferencer import MalletTopicInferencer
import argparse
import json
import os
import bz2
from huggingface_hub import hf_hub_download, list_repo_files  # Add list_repo_files import
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



class LDATopicsPipeline:
    """
    Pipeline for topic modeling using Mallet and SpaCy.
    Handles language detection, lemmatization, vectorization, and topic inference.
    """

    def __init__(self):
        """
        Initializes the pipeline, sets up temporary directories, and starts the JVM for Mallet.
        """
        self.temp_dir = tempfile.mkdtemp(prefix="mallet_models_")  # Create temp folder for models
        self.temp_output_file = None  # Placeholder for temporary output file
        self.latest_model = None
        self.doc_counter = 0

        # Start JVM if not already running
        if not jpype.isJVMStarted():
            mallet_dir = self.setup_mallet_jars()  # Use Hugging Face caching
            # need to add mallet/lib since thats how it saves from hf_hub_download
            classpath = f"{mallet_dir}/mallet.jar:{mallet_dir}/mallet-deps.jar"
            
            # Start JVM with Mallet's classpath
            jpype.startJVM(jpype.getDefaultJVMPath(), f"-Djava.class.path={classpath}")

    
    def setup_mallet_jars(self):
        """
        Downloads Mallet JAR files from Hugging Face Hub and ensures they are locally available.

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


    def __call__(self, text, language=None, doc_name=None, diagnostics_lemmatization=False, diagnostics_topics=False, min_relevance=0.02):
        """
        Executes the pipeline on the input text.

        Parameters:
            text (str): Input text for processing.
            language (str, optional): Language of the text. Auto-detected if None.
            doc_name (str, optional): Name of the document.
            diagnostics_lemmatization (bool): Whether to include lemmatization diagnostics.
            diagnostics_topics (bool): Whether to include topic diagnostics.
            min_relevance (float): Minimum relevance threshold for topics.

        Returns:
            dict: Processed output with topics and metadata.
        """
        self.min_p = min_relevance
        if self.min_p < 0.02:
            raise ValueError("min_p must be at least 0.02")
       
        self.temp_output_file = tempfile.NamedTemporaryFile(
            prefix="tmp_output_", suffix=".mallet", dir=self.temp_dir, delete=False
        )
        self.output_file = self.temp_output_file.name
       

        # PART 1: Language Identification
        self.language = language
        if self.language is None:
            self.language_detection(text)

        from impresso_pipelines.ldatopics.config import SUPPORTED_LANGUAGES, TOPIC_MODEL_DESCRIPTIONS  # Lazy import
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}. Supported languages are: {SUPPORTED_LANGUAGES.keys()}")

        # Part 1.5: Find the latest model version
        self.find_latest_model_version()

        # PART 2: Lemmatization using SpaCy
        lemma_text = self.SPACY(text)

        # PART 3: Vectorization using Mallet
        self.vectorizer_mallet(lemma_text, self.output_file, doc_name)

        # PART 4: Mallet inferencer and JSONification
        self.mallet_inferencer()

        # PART 5: Return the JSON output
        output = self.json_output(filepath=os.path.join(self.temp_dir, "tmp_output.jsonl"))


        # SOME RENAMINGS AND ADDITIONS _____________________________________________________

        # for each entry in the output list, add key "topic_model_description" with the value from the config file for the language
        for entry in output:
            entry["topic_model_description"] = TOPIC_MODEL_DESCRIPTIONS[self.language]
        
        # rename the key "lg" to "language" in the output list
        output = [self.rename_key_preserve_position(entry, 'lg', 'language') for entry in output]
        
        # rename the key "ci_id" to "uid" in the output list, preserving the original key order
        output = [self.rename_key_preserve_position(entry, 'ci_id', 'uid') for entry in output]

        # rename the key "min_p" to "min_relevance" in the output list, preserving the original key order
        output = [self.rename_key_preserve_position(entry, 'min_p', 'min_relevance') for entry in output]
            
        # for each entry in output, if diagnostics_lemmatization is True, add the key "diagnostics_lemmatization" with the value of lemma_text
        if diagnostics_lemmatization:
            for entry in output:
                entry["diagnostics_lemmatization"] = lemma_text
        
        if diagnostics_topics:
            output = self.add_topic_words_to_output(output)
        
        # Rename 'p' to 'relevance' in the topics list
        for entry in output:
            if "topics" in entry:
                for topic in entry["topics"]:
                    topic["uid"] = topic.pop("t", None)
                    topic["relevance"] = topic.pop("p", None)
                    


        # ____________________________________________________________

        if doc_name is None:
            self.doc_counter += 1  # Increment the document counter for the next call
        return output[0]  # Returns clean lemmatized text without punctuation
    
    def find_latest_model_version(self):
        """
        Finds the latest version of the topic model for the specified language.

        Raises:
            ValueError: If no model version is found.
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
        """
        Detects the language of the input text using LangIdentPipeline.

        Parameters:
            text (str): Input text.

        Returns:
            str: Detected language.
        """
        lang_model = LangIdentPipeline()
        lang_result = lang_model(text)
        self.language = lang_result["language"]
        return self.language
    
    def SPACY(self, text):
        """
        Lemmatizes the input text using SpaCy based on the detected language.

        Parameters:
            text (str): Input text.

        Returns:
            str: Lemmatized text.
        """
        from impresso_pipelines.ldatopics.SPACY import SPACY  # Lazy import
        from impresso_pipelines.ldatopics.config import SUPPORTED_LANGUAGES  # Lazy import

        model_id = SUPPORTED_LANGUAGES[self.language]
        if not model_id:
            raise ValueError(f"No SpaCy model available for {self.language}")

        nlp = SPACY(model_id, self.language, self.latest_model)
        return nlp(text)

    def vectorizer_mallet(self, text, output_file, doc_name):
        """
        Vectorizes the lemmatized text using Mallet.

        Parameters:
            text (str): Lemmatized text.
            output_file (str): Path to the output file.
            doc_name (str): Name of the document.
        """
        from impresso_pipelines.ldatopics.mallet_vectorizer_changed import MalletVectorizer  # Lazy import


        # Load the Mallet pipeline
        pipe_file = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=f"models/tm/tm-{self.language}-all-v{self.latest_model}.pipe"
        )


        
        mallet = MalletVectorizer(pipe_file, output_file)
        if doc_name is not None:
            mallet(text, doc_name)
        else:
            mallet(text, f"doc{self.doc_counter}")

    def mallet_inferencer(self):
        """
        Runs the Mallet topic inferencer on the vectorized text.
        """
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
            min_p=self.min_p,
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
            List[dict]: Parsed JSON objects.
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

    def add_topic_words_to_output(self, output):
        """
        Adds top-10 topic words to the output based on precomputed topic descriptions.

        Parameters:
            output (dict): Processed output.

        Returns:
            dict: Output with added topic diagnostics.
        """
        from impresso_pipelines.ldatopics.config import TOPIC_MODEL_DESCRIPTIONS_HF

         # If the pipeline returned a list of docs, recurse into each one
        if isinstance(output, list):
            return [self.add_topic_words_to_output(item) for item in output]

        # 1) Lookup repo_id & filename from your config
        try:
            repo_id, hf_filename = TOPIC_MODEL_DESCRIPTIONS_HF[self.language]
        except KeyError:
            raise ValueError(f"No HF topic‐description entry for language '{self.language}'")

        # 2) Download the compressed .jsonl.bz2 from HF
        compressed = hf_hub_download(repo_id=repo_id, filename=hf_filename)

        # 3) Unpack into a temp folder
        temp_dir = tempfile.mkdtemp(prefix="topic_desc_")
        try:
            jsonl_path = os.path.join(temp_dir, "topic_model_descriptions.jsonl")
            with bz2.open(compressed, "rb") as f_in, open(jsonl_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            # 4) Build a map: full_topic_id → top-10 words
            topic_to_words = {}
            with open(jsonl_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    data = json.loads(line)
                    # use the JSONL's `id` field, which matches your output['topics'][*]['t']
                    full_id = data["id"]
                    word_probs = data.get("word_probs", [])
                    # sort by prob desc, take the first 10 words
                    top10 = [
                        wp["word"]
                        for wp in sorted(word_probs, key=lambda x: x.get("prob", 0), reverse=True)[:10]
                    ]
                    topic_to_words[full_id] = top10

            # 5) Stitch into output
            diagnostics = {}
            for t in output.get("topics", []):
                key = t.get("t") or t.get("topic_model")
                diagnostics[key] = topic_to_words.get(key, [])

            output["diagnostics_topics"] = diagnostics

        finally:
            shutil.rmtree(temp_dir)

        return output


    def rename_key_preserve_position(self, d: dict, old_key: str, new_key: str) -> dict:
        """
        Renames a key in a dictionary while preserving the original key order.

        Parameters:
            d (dict): Input dictionary.
            old_key (str): Key to be renamed.
            new_key (str): New key name.

        Returns:
            dict: Dictionary with the renamed key.
        """
        new_d = {}
        for k, v in d.items():
            if k == old_key:
                new_d[new_key] = v
            else:
                new_d[k] = v
        return new_d