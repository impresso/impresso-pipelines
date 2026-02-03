"""
LDA topic modeling pipeline using Mallet and SpaCy for multilingual text analysis.

This module provides a complete pipeline for extracting topics from text documents
using Latent Dirichlet Allocation (LDA) via Mallet. It handles language detection,
lemmatization with SpaCy, text vectorization, and topic inference.

Supported languages: French (fr), German (de), Luxembourgish (lb)

Example usage:
    >>> pipeline = LDATopicsPipeline()
    >>> result = pipeline("This is a sample text for topic modeling.")
    >>> print(result['topics'])
    [{'uid': 'tm-fr-all-v2.1.0-t42', 'relevance': 0.85}, ...]
    
    >>> # With diagnostics
    >>> result = pipeline(
    ...     "Sample text",
    ...     language="fr",
    ...     diagnostics_topics=True,
    ...     min_relevance=0.05
    ... )
    >>> print(result['diagnostics_topics'])
"""

from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline
from impresso_pipelines.ldatopics.config import (
    SUPPORTED_LANGUAGES,
    TOPIC_MODEL_DESCRIPTIONS,
    TOPIC_MODEL_DESCRIPTIONS_HF,
)
from impresso_pipelines.ldatopics.mallet_topic_inferencer import MalletTopicInferencer
import argparse
import json
import os
import bz2
from typing import Dict, List, Any, Optional, Union, Tuple
from huggingface_hub import hf_hub_download, list_repo_files
import tempfile
import shutil
import subprocess
import sys
import logging
try:
    import jpype
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jpype1"])
    import jpype

logger = logging.getLogger(__name__)


class LDATopicsPipeline:
    """
    LDA topic modeling pipeline using Mallet and SpaCy.
    
    This pipeline processes text through multiple stages:
    1. Language detection (if not specified)
    2. Lemmatization using SpaCy language models
    3. Text vectorization with Mallet
    4. Topic inference using pre-trained LDA models
    
    The pipeline uses pre-trained topic models from Hugging Face Hub and
    automatically downloads required Mallet JARs and SpaCy models.
    
    Attributes:
        temp_dir (str): Temporary directory for model files and intermediate outputs
        temp_output_file: Temporary file handle for Mallet output
        latest_model (Optional[str]): Version string of the latest topic model
        doc_counter (int): Counter for auto-generated document names
        language (Optional[str]): Detected or specified language code
        
    Example:
        >>> pipeline = LDATopicsPipeline()
        >>> result = pipeline(
        ...     "Le texte français pour l'analyse",
        ...     language="fr",
        ...     min_relevance=0.03
        ... )
        >>> print(f"Language: {result['language']}")
        >>> print(f"Topics: {len(result['topics'])}")
    """

    def __init__(self) -> None:
        """
        Initialize the LDA topics pipeline.
        
        Sets up temporary directories, downloads Mallet JAR files from Hugging Face,
        and initializes the Java Virtual Machine (JVM) with Mallet's classpath.
        
        Raises:
            RuntimeError: If JVM cannot be started or Mallet classes are unavailable
            OSError: If JAVA_HOME is not set and JVM path cannot be determined
        """
        self.temp_dir = tempfile.mkdtemp(prefix="mallet_models_")  # Create temp folder for models
        self.temp_output_file = None  # Placeholder for temporary output file
        self.latest_model = None
        self.doc_counter = 0
        self.lang_identifier = LangIdentPipeline()
        self.supported_languages = SUPPORTED_LANGUAGES
        self.topic_model_descriptions = TOPIC_MODEL_DESCRIPTIONS
        self.topic_model_descriptions_hf = TOPIC_MODEL_DESCRIPTIONS_HF
        self._spacy_pipelines: Dict[Tuple[str, str], Any] = {}

        # Start JVM if not already running
        if not jpype.isJVMStarted():
            mallet_dir = self.setup_mallet_jars()  # Use Hugging Face caching
            # need to add mallet/lib since thats how it saves from hf_hub_download
            classpath = f"{mallet_dir}/mallet.jar:{mallet_dir}/mallet-deps.jar"
            # Start JVM with Mallet's classpath
            # Try to get JVM path, with fallback to JAVA_HOME if default fails
            try:
                jvm_path = jpype.getDefaultJVMPath()
            except Exception as e:
                # If getDefaultJVMPath() fails, try to use JAVA_HOME or system default
                java_home = os.environ.get('JAVA_HOME')
                if java_home:
                    # Try common JVM library locations
                    import platform
                    system = platform.system()
                    if system == 'Darwin':  # macOS
                        jvm_path = os.path.join(java_home, 'lib', 'server', 'libjvm.dylib')
                        if not os.path.exists(jvm_path):
                            jvm_path = os.path.join(java_home, 'lib', 'jli', 'libjli.dylib')
                    elif system == 'Linux':
                        jvm_path = os.path.join(java_home, 'lib', 'server', 'libjvm.so')
                    else:  # Windows
                        jvm_path = os.path.join(java_home, 'bin', 'server', 'jvm.dll')
                    
                    if not os.path.exists(jvm_path):
                        raise RuntimeError(f"Could not find JVM library. Please set JAVA_HOME environment variable. Error: {e}")
                else:
                    raise RuntimeError(f"Could not find JVM. Please install Java and/or set JAVA_HOME environment variable. Error: {e}")
            
            jpype.startJVM(jvm_path, f"-Djava.class.path={classpath}")
        else:
            # JVM already started, check if Mallet classes are available
            try:
                from cc.mallet.classify.tui import Csv2Vectors
            except ImportError as e:
                logger.error("JVM is already started but Mallet classes are not available in the classpath.")
                logger.error("This usually happens if another library started the JVM without Mallet jars.")
                raise RuntimeError("JVM started without Mallet jars. Please ensure no other code starts the JVM before LDATopicsPipeline.") from e

    
    def setup_mallet_jars(self) -> str:
        """
        Download Mallet JAR files from Hugging Face Hub.
        
        Downloads mallet.jar and mallet-deps.jar from the impresso-project repository
        and caches them locally using Hugging Face's download mechanism.

        Returns:
            Path to the directory containing the downloaded Mallet JAR files.
            
        Note:
            Files are cached by Hugging Face Hub, so subsequent calls won't re-download.
        """
        jar_files = ["mallet.jar", "mallet-deps.jar"]
        jar_paths = []

        for jar_name in jar_files:
            logger.info("Downloading %s from Hugging Face Hub...", jar_name)
            jar_path = hf_hub_download(
                repo_id="impresso-project/mallet-topic-inferencer",
                filename=f"mallet/lib/{jar_name}"
            )
            jar_paths.append(jar_path)

        # Return the directory containing the first JAR file (all files are in the same directory)
        return os.path.dirname(jar_paths[0])


    def __call__(
        self, 
        text: str, 
        language: Optional[str] = None, 
        doc_name: Optional[str] = None, 
        diagnostics_lemmatization: bool = False, 
        diagnostics_topics: bool = False, 
        min_relevance: float = 0.02
    ) -> Dict[str, Any]:
        """
        Execute the complete topic modeling pipeline on input text.
        
        Processes text through language detection, lemmatization, vectorization,
        and topic inference. Returns identified topics with relevance scores.

        Args:
            text: Input text to process for topic modeling
            language: Language code ('fr', 'de', 'lb'). Auto-detected if None.
            doc_name: Document identifier. Auto-generated if None.
            diagnostics_lemmatization: If True, includes lemmatized text in output
            diagnostics_topics: If True, includes top-10 words for each topic
            min_relevance: Minimum topic relevance threshold (must be >= 0.02)

        Returns:
            Dictionary containing:
                - uid (str): Document identifier
                - language (str): Language code
                - topic_model_description (str): Model version info
                - topics (List[Dict]): List of topics with 'uid' and 'relevance'
                - min_relevance (float): Applied threshold
                - diagnostics_lemmatization (str): Only if diagnostics_lemmatization=True
                - diagnostics_topics (Dict): Only if diagnostics_topics=True

        Raises:
            ValueError: If min_relevance < 0.02 or language is not supported
            
        Example:
            >>> pipeline = LDATopicsPipeline()
            >>> result = pipeline(
            ...     "Le gouvernement a annoncé de nouvelles mesures.",
            ...     language="fr",
            ...     min_relevance=0.05
            ... )
            >>> for topic in result['topics']:
            ...     print(f"Topic {topic['uid']}: {topic['relevance']:.3f}")
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

        if self.language not in self.supported_languages:
            raise ValueError(
                f"Unsupported language: {self.language}. Supported languages are: {self.supported_languages.keys()}"
            )

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

        # for each entry in the output list, add key "topic_model_description" with the value from the config file for the language
        for entry in output:
            entry["topic_model_description"] = self.topic_model_descriptions[self.language]
        
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
                    

        if doc_name is None:
            self.doc_counter += 1  # Increment the document counter for the next call
        return output[0]  # Returns clean lemmatized text without punctuation
    
    def find_latest_model_version(self) -> None:
        """
        Find and set the latest topic model version for the current language.
        
        Queries Hugging Face Hub for available model versions and selects the
        most recent one based on version numbering in filenames.

        Raises:
            ValueError: If no model version is found for the specified language
            
        Side effects:
            Sets self.latest_model to the version string (e.g., "2.1.0")
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

    def language_detection(self, text: str) -> str:
        """
        Detect the language of input text using LangIdentPipeline.

        Args:
            text: Input text for language detection

        Returns:
            Detected language code (e.g., 'fr', 'de', 'lb')
            
        Side effects:
            Sets self.language to the detected language code
        """
        lang_result = self.lang_identifier(text)
        self.language = lang_result["language"]
        return self.language
    
    def SPACY(self, text: str) -> str:
        """
        Lemmatize input text using language-specific SpaCy models.
        
        Downloads and uses the appropriate SpaCy model based on self.language.
        The model is configured for the specific topic model version being used.

        Args:
            text: Input text to lemmatize

        Returns:
            Lemmatized text with tokens joined by spaces
            
        Raises:
            ValueError: If no SpaCy model is available for the current language
            
        Note:
            SpaCy models are downloaded automatically if not already present.
        """
        from impresso_pipelines.ldatopics.SPACY import SPACY as SpacyPipeline  # Lazy import

        model_id = self.supported_languages[self.language]
        if not model_id:
            raise ValueError(f"No SpaCy model available for {self.language}")

        cache_key = (self.language, self.latest_model)
        nlp = self._spacy_pipelines.get(cache_key)
        if nlp is None:
            nlp = SpacyPipeline(model_id, self.language, self.latest_model)
            self._spacy_pipelines[cache_key] = nlp
        return nlp(text)

    def vectorizer_mallet(self, text: str, output_file: str, doc_name: str) -> None:
        """
        Vectorize lemmatized text using Mallet's pipeline.
        
        Loads the appropriate Mallet pipeline file for the current language and
        version, then converts text to Mallet's vector format.

        Args:
            text: Lemmatized text to vectorize
            output_file: Path where Mallet output will be written
            doc_name: Document identifier for tracking
            
        Side effects:
            Writes vectorized output to output_file
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

    def mallet_inferencer(self) -> None:
        """
        Run Mallet topic inference on vectorized text.
        
        Downloads pre-trained topic model files (inferencer and pipe) from Hugging Face,
        configures the MalletTopicInferencer with appropriate parameters, and executes
        topic inference.
        
        Side effects:
            Writes inference results to temporary JSONL file in self.temp_dir
            
        Note:
            Uses self.language, self.latest_model, and self.min_p to configure inference.
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

    
    def json_output(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Read and parse JSONL output file from Mallet inference.
        
        Reads the inference results file line by line, parsing each as JSON.
        Skips empty lines and logs warnings for malformed JSON.

        Args:
            filepath: Path to the JSONL file to read

        Returns:
            List of parsed JSON objects (dictionaries) from the file
            
        Side effects:
            Deletes the filepath after reading
            
        Note:
            Handles malformed JSON gracefully by logging warnings and continuing.
        """
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid line: {line}\nError: {e}")

        # delete the file after reading
        os.remove(filepath)

        return data

    def add_topic_words_to_output(self, output: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Add top-10 topic words to output for diagnostic purposes.
        
        Downloads pre-computed topic descriptions from Hugging Face, extracts the
        top 10 words for each topic, and adds them to the output under 'diagnostics_topics'.

        Args:
            output: Single result dictionary or list of result dictionaries

        Returns:
            Output with added 'diagnostics_topics' field containing top words for each topic
            
        Raises:
            ValueError: If no topic description file is configured for the current language
            
        Example output structure:
            {
                ...,
                "diagnostics_topics": {
                    "tm-fr-all-v2.1.0-t42": ["word1", "word2", ...],
                    "tm-fr-all-v2.1.0-t15": ["word3", "word4", ...]
                }
            }
        """
         # If the pipeline returned a list of docs, recurse into each one
        if isinstance(output, list):
            return [self.add_topic_words_to_output(item) for item in output]

        # 1) Lookup repo_id & filename from your config
        try:
            repo_id, hf_filename = self.topic_model_descriptions_hf[self.language]
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
