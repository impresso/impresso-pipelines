import jpype
import jpype.imports
from jpype.types import JString
import os
import urllib.request
from typing import List, Dict, Optional
import tempfile
import shutil
from ..langident import LangIdentPipeline
import importlib.resources
from .lang_configs import LANG_CONFIGS

class SolrNormalizationPipeline:
    """
    Pipeline for text normalization using Apache Lucene analyzers.
    Handles language detection, tokenization, and normalization for supported languages ('de', 'fr', 'el', 'ru').
    """

    LUCENE_VERSION = "9.3.0"

    def __init__(self):
        """
        Initialize the pipeline, setting up temporary directories, downloading dependencies, and preparing stopwords.
        Supports all languages defined in LANG_CONFIGS.
        """
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="solrnorm_")
        self.lib_dir = os.path.join(self.temp_dir, "lib")
        self.stopwords = {
            lang: os.path.join(self.temp_dir, f"stopwords_{lang}.txt")
            for lang in LANG_CONFIGS
        }
        self.jar_urls = {
            "lucene-core": f"https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/{self.LUCENE_VERSION}/lucene-core-{self.LUCENE_VERSION}.jar",
            "lucene-analysis-common": f"https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-common/{self.LUCENE_VERSION}/lucene-analysis-common-{self.LUCENE_VERSION}.jar"
        }
        
        self._setup_environment()
        self._download_dependencies()
        self._create_stopwords()
        self._analyzers = {}
        self._lang_detector = None

    def __enter__(self):
        """
        Enter context manager.
        Returns:
            Self instance for use within a context.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager and clean up resources.
        """
        self.cleanup()


    def cleanup(self):
        """
        Clean up temporary directories and resources.
        Ensures analyzers are closed and temporary files are deleted.
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
            print(f"Warning: Cleanup failed: {e}")

    def _load_snowball_stopwords(self, filepath):
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

    def __del__(self):
        """
        Destructor to ensure cleanup happens if context manager is not used.
        """
        self.cleanup()

    def _setup_environment(self):
        """
        Create necessary directories for storing dependencies.
        """
        os.makedirs(self.lib_dir, exist_ok=True)

    def _download_dependencies(self):
        """
        Download required Lucene JAR files if not already present.
        """
        for name, url in self.jar_urls.items():
            dest = os.path.join(self.lib_dir, os.path.basename(url))
            if not os.path.isfile(dest):
                print(f"⬇️ Downloading {name}...")
                urllib.request.urlretrieve(url, dest)
            else:
                print(f"✔️ {name} already exists.")

    def _create_stopwords(self):
        """
        Generate stopword files for supported languages.
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

    def _start_jvm(self):
        """
        Start the JVM with the required classpath for Lucene libraries.
        """
        if not jpype.isJVMStarted():
            jar_paths = [os.path.join(self.lib_dir, os.path.basename(url)) 
                        for url in self.jar_urls.values()]
            print("📦 Starting JVM with classpath:")
            for j in jar_paths:
                print(" -", j)
            jpype.startJVM(classpath=jar_paths)

    def _build_analyzer(self, lang: str):
        """
        Build a custom Lucene analyzer for the specified language using LANG_CONFIGS.
        
        Args:
            lang (str): Language code ('de' or 'fr').
        
        Returns:
            CustomAnalyzer instance configured for the language.
        
        Raises:
            ValueError: If the language is unsupported.
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

    def _analyze_text(self, analyzer, text: str) -> List[str]:
        """
        Tokenize and normalize text using the provided Lucene analyzer.
        
        Args:
            analyzer: Lucene analyzer instance.
            text (str): Input text to process.
        
        Returns:
            List of normalized tokens.
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
        Detect the language of the input text using LangIdentPipeline.
        Returns a language code supported in LANG_CONFIGS.
        """
        if self._lang_detector is None:
            self._lang_detector = LangIdentPipeline()
        
        result = self._lang_detector(text)
        detected_lang = result['language']
        
        if detected_lang not in LANG_CONFIGS:
            raise ValueError(f"Detected language '{detected_lang}' is not supported. Supported: {list(LANG_CONFIGS.keys())}")
        return detected_lang
    


    def __call__(self, text: str, lang: Optional[str] = None, diagnostics: Optional[bool] = False) -> Dict[str, List[str]]:
        """
        Process text through the normalization pipeline.
        Supports all languages defined in LANG_CONFIGS.
        
        Args:
            text (str): Input text to normalize.
            lang (str, optional): Language code ('de' or 'fr'). If not provided, language is detected automatically.
            diagnostics (bool, optional): If True, returns additional diagnostic information (not implemented).
        
        Returns:
            Dict containing normalized tokens and detected language.
        
        Raises:
            ValueError: If the language (specified or detected) is unsupported.
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
