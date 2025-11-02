"""
Text normalization pipeline using Apache Lucene analyzers for multilingual processing.

This module provides a pipeline for normalizing text across multiple languages using
Apache Lucene's analysis framework. It handles language detection, tokenization,
stopword removal, stemming, and other text normalization tasks.

Supported languages: German (de), English (en), Spanish (es), French (fr), 
Italian (it), Dutch (nl), Portuguese (pt).

Example usage:
    >>> with SolrNormalizationPipeline() as pipeline:
    ...     result = pipeline("Dies ist ein deutscher Text.")
    ...     print(result['tokens'])
    ['deutsch', 'text']
    
    >>> # With explicit language specification
    >>> with SolrNormalizationPipeline(lucene_version="9.3.0") as pipeline:
    ...     result = pipeline("This is English text.", lang="en")
    ...     print(result)
    {'language': 'en', 'tokens': ['english', 'text']}
"""

import jpype
import jpype.imports
from jpype.types import JString
import os
import urllib.request
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import tempfile
import shutil
import logging
from ..langident import LangIdentPipeline
import importlib.resources
from .lang_configs import LANG_CONFIGS

logger = logging.getLogger(__name__)


class SolrNormalizationPipeline:
    """
    Text normalization pipeline using Apache Lucene analyzers.
    
    This pipeline normalizes text for search and analysis by performing language-specific
    processing including tokenization, lowercasing, stopword removal, stemming, and
    elision handling (for languages like French).
    
    The pipeline can operate in two modes:
    1. Auto-download mode: Automatically downloads required Lucene JARs
    2. External JAR mode: Uses pre-existing Lucene JAR files
    
    Attributes:
        lucene_version (str): Version of Apache Lucene being used
        temp_dir (str): Temporary directory for downloaded files and stopwords
        lib_dir (str): Directory containing Lucene JAR files
        stopwords (Dict[str, str]): Mapping of language codes to stopword file paths
        
    Example:
        >>> # Basic usage with auto-download
        >>> with SolrNormalizationPipeline() as pipeline:
        ...     result = pipeline("Der Wald ist schön")
        ...     print(result['language'], result['tokens'])
        de ['wald', 'schon']
        
        >>> # Using external Lucene JARs
        >>> pipeline = SolrNormalizationPipeline(lucene_dir="./lucene_jars")
        >>> result = pipeline("Le forêt est belle", lang="fr")
        >>> pipeline.cleanup()
    """

    def __init__(self, lucene_dir: Optional[str] = None, lucene_version: str = "9.3.0") -> None:
        """
        Initialize the normalization pipeline.
        
        Sets up temporary directories, downloads Lucene dependencies (if needed),
        and prepares stopword files for all supported languages.
        
        Args:
            lucene_dir: Optional path to directory containing Lucene JAR files.
                       If not provided, JARs will be auto-downloaded to a temp directory.
            lucene_version: Apache Lucene version to use (default: "9.3.0").
                           Only used when lucene_dir is not specified.
                           
        Raises:
            urllib.error.URLError: If JAR download fails
            OSError: If temporary directory creation fails
        """
        self._external_lucene_dir = lucene_dir
        self.lucene_version = lucene_version
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="solrnorm_")
        self.lib_dir = os.path.join(self.temp_dir, "lib")
        self.stopwords = {
            lang: os.path.join(self.temp_dir, f"stopwords_{lang}.txt")
            for lang in LANG_CONFIGS
        }
        self.jar_urls = {
            "lucene-core": f"https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/{self.lucene_version}/lucene-core-{self.lucene_version}.jar",
            "lucene-analysis-common": f"https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-common/{self.lucene_version}/lucene-analysis-common-{self.lucene_version}.jar"
        }
        self._setup_environment()
        if not self._external_lucene_dir:
            self._download_dependencies()
        self._create_stopwords()
        self._analyzers = {}
        self._lang_detector = None

    def __enter__(self) -> 'SolrNormalizationPipeline':
        """
        Enter the context manager.
        
        Returns:
            SolrNormalizationPipeline: Self instance for use within a with statement.
            
        Example:
            >>> with SolrNormalizationPipeline() as pipeline:
            ...     result = pipeline("Text to normalize")
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager and perform cleanup.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.cleanup()


    def cleanup(self) -> None:
        """
        Clean up temporary directories and release resources.
        
        Closes all open Lucene analyzers and removes temporary files/directories
        created during pipeline initialization. Safe to call multiple times.
        """
        try:
            if hasattr(self, '_analyzers'):
                # Close any open analyzers
                for analyzer in self._analyzers.values():
                    try:
                        analyzer.close()
                    except:
                        pass
                self._analyzers.clear()
            
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def _load_snowball_stopwords(self, filepath: Union[str, Path]) -> List[str]:
        """
        Load stopwords from a Snowball-formatted file.
        
        Snowball stopword files use pipes (|) as comment markers. This method
        extracts only the stopword tokens, ignoring comments.
        
        Args:
            filepath: Path to the stopword file (can be Path object or string)
            
        Returns:
            List of stopword strings
            
        Example file format:
            word1 | comment
            word2
            | full line comment
        """
        stopwords = []
        # Support both Path and str
        if hasattr(filepath, "open"):
            f = filepath.open("r", encoding="utf-8")
        else:
            f = open(filepath, encoding="utf-8")
        with f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('|'):
                    continue
                word = line.split('|')[0].strip()
                if word:
                    stopwords.append(word)
        return stopwords

    def __del__(self) -> None:
        """
        Destructor to ensure cleanup if context manager is not used.
        
        Note: Relying on __del__ for cleanup is not recommended. Use the
        context manager (with statement) or call cleanup() explicitly.
        """
        self.cleanup()

    def _setup_environment(self) -> None:
        """
        Create necessary directory structure for storing Lucene dependencies.
        
        Creates the lib_dir where Lucene JAR files will be stored (if auto-downloading).
        """
        os.makedirs(self.lib_dir, exist_ok=True)

    def _download_dependencies(self) -> None:
        """
        Download required Apache Lucene JAR files from Maven Central.
        
        Downloads lucene-core and lucene-analysis-common JARs if they don't
        already exist in the lib_dir. Uses the version specified in lucene_version.
        
        Downloads are cached - if a JAR already exists, it won't be re-downloaded.
        
        Raises:
            urllib.error.URLError: If download fails due to network issues
        """
        for name, url in self.jar_urls.items():
            dest = os.path.join(self.lib_dir, os.path.basename(url))
            if not os.path.isfile(dest):
                logger.info(f"Downloading {name}...")
                urllib.request.urlretrieve(url, dest)
            else:
                logger.debug(f"{name} already exists.")

    def _create_stopwords(self) -> None:
        """
        Generate stopword files for all supported languages.
        
        Loads stopwords from package resources and writes them to temporary files
        that can be used by Lucene analyzers. Only creates files for languages
        that have stopword configurations in LANG_CONFIGS.
        """
        stopwords = {}
        for lang, config in LANG_CONFIGS.items():
            stopwords_file = config.get("stopwords_file")
            if stopwords_file:
                stopwords[lang] = self._load_snowball_stopwords(
                    importlib.resources.files(__package__).joinpath(stopwords_file)
                )
        for lang, words in stopwords.items():
            if lang in self.stopwords:
                if not os.path.isfile(self.stopwords[lang]):
                    with open(self.stopwords[lang], "w", encoding="utf8") as f:
                        f.write("\n".join(words))

    def _start_jvm(self) -> None:
        """
        Initialize the Java Virtual Machine with Lucene classpath.
        
        Starts JPype JVM if not already running, configuring the classpath to include
        all required Lucene JAR files. If the JVM is already running, verifies that
        Lucene classes are available.
        
        Raises:
            RuntimeError: If JVM is already started without Lucene JARs in classpath
            ImportError: If required Lucene classes cannot be imported
        """
        if not jpype.isJVMStarted():
            if self._external_lucene_dir:
                import glob
                jar_paths = glob.glob(os.path.join(self._external_lucene_dir, "*.jar"))
                logger.info("Starting JVM with external lucene_dir classpath:")
                for j in jar_paths:
                    logger.debug(f"  {j}")
            else:
                jar_paths = [os.path.join(self.lib_dir, os.path.basename(url)) 
                             for url in self.jar_urls.values()]
                logger.info("Starting JVM with downloaded classpath:")
                for j in jar_paths:
                    logger.debug(f"  {j}")
            jpype.startJVM(classpath=jar_paths)
        else:
            # JVM already started, check if Lucene classes are available
            try:
                from org.apache.lucene.analysis.standard import StandardAnalyzer
                from org.apache.lucene.analysis.custom import CustomAnalyzer
            except ImportError as e:
                logger.error("JVM is already started but Lucene classes are not available in the classpath.")
                logger.error("This usually happens if another library started the JVM without Lucene jars.")
                raise RuntimeError("JVM started without Lucene jars. Please ensure no other code starts the JVM before SolrNormalizationPipeline.") from e

    def _build_analyzer(self, lang: str) -> Any:
        """
        Build a custom Lucene analyzer for a specific language.
        
        Constructs a Lucene CustomAnalyzer with language-specific processing pipeline
        defined in LANG_CONFIGS. The pipeline typically includes tokenization,
        lowercasing, stopword removal, and stemming.
        
        Args:
            lang: Language code (e.g., 'de', 'fr', 'en')
        
        Returns:
            Lucene CustomAnalyzer instance configured for the specified language
        
        Raises:
            ValueError: If the language is not supported (not in LANG_CONFIGS)
            
        Example:
            >>> analyzer = pipeline._build_analyzer('de')
            >>> # Analyzer is configured with German tokenization and stemming
        """
        from org.apache.lucene.analysis.custom import CustomAnalyzer
        from java.nio.file import Paths
        from java.util import HashMap

        if lang not in LANG_CONFIGS:
            raise ValueError(f"Unsupported language: {lang}")

        config = LANG_CONFIGS[lang]
        builder = CustomAnalyzer.builder(Paths.get("."))

        # Track if stop or elision params are needed
        for step in config["analyzer_pipeline"]:
            if step["type"] == "tokenizer":
                builder = builder.withTokenizer(step["name"])
            elif step["type"] == "tokenfilter":
                if step["name"] == "stop":
                    stop_params = HashMap()
                    for k, v in config.get("stop_params", {}).items():
                        stop_params.put(k, v)
                    stop_params.put("words", self.stopwords[lang])
                    builder = builder.addTokenFilter("stop", stop_params)
                elif step["name"] == "elision":
                    elision_params = HashMap()
                    for k, v in config.get("elision_params", {}).items():
                        elision_params.put(k, v)
                    # For French, articles param is the stopword file
                    elision_params.put("articles", self.stopwords[lang])
                    builder = builder.addTokenFilter("elision", elision_params)
                else:
                    builder = builder.addTokenFilter(step["name"])
        return builder.build()

    def _analyze_text(self, analyzer: Any, text: str) -> List[str]:
        """
        Tokenize and normalize text using a Lucene analyzer.
        
        Processes the input text through the analyzer's token stream, extracting
        normalized tokens. Properly handles stream lifecycle (reset, process, end, close).
        
        Args:
            analyzer: Lucene analyzer instance (e.g., CustomAnalyzer)
            text: Input text to process
        
        Returns:
            List of normalized token strings
            
        Example:
            >>> tokens = pipeline._analyze_text(analyzer, "Der schöne Wald")
            >>> print(tokens)
            ['schon', 'wald']
        """
        from java.io import StringReader
        from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
        tokens = []
        stream = analyzer.tokenStream("field", StringReader(text))
        try:
            termAttr = stream.addAttribute(CharTermAttribute.class_)
            stream.reset()
            while stream.incrementToken():
                tokens.append(termAttr.toString())
            stream.end()
            return tokens
        finally:
            stream.close()

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of input text using LangIdentPipeline.
        
        Uses a language identification model to determine the language. If the detected
        language is not supported by this pipeline, raises an error. Falls back to
        "general" mode if confidence is too low.
        
        Args:
            text: Input text for language detection
            
        Returns:
            Language code from LANG_CONFIGS (e.g., 'de', 'fr', 'en')
            
        Raises:
            ValueError: If detected language is not in LANG_CONFIGS
            
        Note:
            The language detector is initialized lazily on first use and cached
            for subsequent calls.
        """
        if self._lang_detector is None:
            self._lang_detector = LangIdentPipeline()
        
        result = self._lang_detector(text)
        detected_lang = result['language']
        confidence = result['score']
        
        if detected_lang not in LANG_CONFIGS:
            raise ValueError(f"Detected language '{detected_lang}' is not supported. Supported: {list(LANG_CONFIGS.keys())}")
        
        if confidence < 0.5:
            detected_lang = "general"
            logger.warning(f"Low confidence ({confidence}) in detected language '{detected_lang}'. Switching to general case. Otherwise, consider providing a specific language code.")

        return detected_lang
    


    def __call__(
        self, 
        text: str, 
        lang: Optional[str] = None, 
        diagnostics: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Process text through the normalization pipeline.
        
        Main entry point for text normalization. Detects language (if not provided),
        builds appropriate analyzer, and returns normalized tokens.
        
        Args:
            text: Input text to normalize
            lang: Optional language code (e.g., 'de', 'fr'). If None, language is auto-detected.
            diagnostics: If True, includes additional debugging information in output
            
        Returns:
            Dictionary containing:
                - language (str): Detected or specified language code
                - tokens (List[str]): Normalized tokens
                - stopwords_detected (List[str]): Only if diagnostics=True
                - analyzer_pipeline (List[Dict]): Only if diagnostics=True
        
        Raises:
            ValueError: If language (specified or detected) is not supported
            
        Example:
            >>> with SolrNormalizationPipeline() as pipeline:
            ...     result = pipeline("Der schöne Wald")
            ...     print(result)
            {'language': 'de', 'tokens': ['schon', 'wald']}
            
            >>> # With diagnostics
            >>> result = pipeline("Le forêt", lang="fr", diagnostics=True)
            >>> print(result['analyzer_pipeline'])
            [{'type': 'tokenizer', 'name': 'standard'}, ...]
        """
        # Detect language if not specified
        detected_lang = self._detect_language(text) if lang is None else lang

        if detected_lang not in LANG_CONFIGS:
            raise ValueError(f"Unsupported language: '{detected_lang}'. Supported: {', '.join(LANG_CONFIGS.keys())}")

        self._start_jvm()

        if detected_lang not in self._analyzers:
            self._analyzers[detected_lang] = self._build_analyzer(detected_lang)

        tokens = self._analyze_text(self._analyzers[detected_lang], text)

        if diagnostics:
            stopword_file = LANG_CONFIGS[detected_lang].get("stopwords_file")
            if stopword_file is None:
                stopwords_set = set()
            else:
                stopwords_set = set(self._load_snowball_stopwords(
                    importlib.resources.files(__package__).joinpath(stopword_file)
                ))
            text_tokens = set(word.lower() for word in text.split())
            detected = [sw for sw in stopwords_set if sw.lower() in text_tokens]
            return {
                "language": detected_lang,
                "tokens": tokens,
                "stopwords_detected": detected,
                "analyzer_pipeline": LANG_CONFIGS[detected_lang].get("analyzer_pipeline", [])
            }

        return {
            "language": detected_lang,
            "tokens": tokens
        }
