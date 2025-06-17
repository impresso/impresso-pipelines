import jpype
import jpype.imports
from jpype.types import JString
import os
import urllib.request
from typing import List, Dict, Optional
import tempfile
import shutil

class SolrNormalizationPipeline:
    LUCENE_VERSION = "9.3.0"
    
    def __init__(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="solrnorm_")
        self.lib_dir = os.path.join(self.temp_dir, "lib")
        self.stopwords = {
            "de": os.path.join(self.temp_dir, "stopwords_de.txt"),
            "fr": os.path.join(self.temp_dir, "stopwords_fr.txt")
        }
        self.jar_urls = {
            "lucene-core": f"https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/{self.LUCENE_VERSION}/lucene-core-{self.LUCENE_VERSION}.jar",
            "lucene-analysis-common": f"https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-common/{self.LUCENE_VERSION}/lucene-analysis-common-{self.LUCENE_VERSION}.jar"
        }
        
        self._setup_environment()
        self._download_dependencies()
        self._create_stopwords()
        self._analyzers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up temporary directory and resources"""
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

    def __del__(self):
        """Ensure cleanup happens even if context manager is not used"""
        self.cleanup()

    def _setup_environment(self):
        os.makedirs(self.lib_dir, exist_ok=True)

    def _download_dependencies(self):
        for name, url in self.jar_urls.items():
            dest = os.path.join(self.lib_dir, os.path.basename(url))
            if not os.path.isfile(dest):
                print(f"⬇️ Downloading {name}...")
                urllib.request.urlretrieve(url, dest)
            else:
                print(f"✔️ {name} already exists.")

    def _create_stopwords(self):
        stopwords = {
            "de": ["und", "oder", "aber", "der", "die", "das", "über", "den"],
            "fr": ["le", "la", "les", "des", "et", "mais", "ou", "donc", "or", "ni", "car"]
        }
        for lang, words in stopwords.items():
            if not os.path.isfile(self.stopwords[lang]):
                with open(self.stopwords[lang], "w", encoding="utf8") as f:
                    f.write("\n".join(words))

    def _start_jvm(self):
        if not jpype.isJVMStarted():
            jar_paths = [os.path.join(self.lib_dir, os.path.basename(url)) 
                        for url in self.jar_urls.values()]
            print("📦 Starting JVM with classpath:")
            for j in jar_paths:
                print(" -", j)
            jpype.startJVM(classpath=jar_paths)

    def _build_analyzer(self, lang: str):
        from org.apache.lucene.analysis.custom import CustomAnalyzer
        from java.nio.file import Paths
        from java.util import HashMap

        stop_params = HashMap()
        stop_params.put("ignoreCase", "true")
        stop_params.put("words", self.stopwords[lang])  # Updated to use instance stopwords path
        stop_params.put("format", "wordset")

        builder = CustomAnalyzer.builder(Paths.get("."))

        if lang == "de":
            return (builder
                .withTokenizer("standard")
                .addTokenFilter("lowercase")
                .addTokenFilter("stop", stop_params)
                .addTokenFilter("germanNormalization")
                .addTokenFilter("asciifolding")
                .addTokenFilter("germanMinimalStem")
                .build()
            )
        elif lang == "fr":
            elision_params = HashMap()
            elision_params.put("ignoreCase", "true")
            elision_params.put("articles", self.stopwords["fr"])

            return (builder
                .withTokenizer("standard")
                .addTokenFilter("elision", elision_params)
                .addTokenFilter("lowercase")
                .addTokenFilter("stop", stop_params)
                .addTokenFilter("asciifolding")
                .addTokenFilter("frenchMinimalStem")
                .build()
            )
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def _analyze_text(self, analyzer, text: str) -> List[str]:
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

    def __call__(self, text: str, lang: str = "de") -> Dict[str, List[str]]:
        """
        Process text through the Solr normalization pipeline.
        
        Args:
            text (str): The input text to normalize
            lang (str): Language code ('de' or 'fr')
            
        Returns:
            Dict containing normalized tokens
        """
        if lang not in self.stopwords:
            raise ValueError(f"Unsupported language: {lang}")

        self._start_jvm()
        
        if lang not in self._analyzers:
            self._analyzers[lang] = self._build_analyzer(lang)
            
        tokens = self._analyze_text(self._analyzers[lang], text)
        
        return {
            "language": lang,
            "tokens": tokens
        }
