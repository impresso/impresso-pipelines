from typing import List, Dict, Union, Optional
import unicodedata
from huggingface_hub import hf_hub_download, list_repo_files
from pybloomfilter import BloomFilter
import re
from functools import lru_cache

from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline


@lru_cache(maxsize=1)
def cached_list_repo_files(repo_id: str):
    """
    Cache the list of files in a Hugging Face repository.

    Args:
        repo_id (str): The repository ID.

    Returns:
        List[str]: List of file names in the repository.
    """
    return list_repo_files(repo_id)


def get_bloomfilter(model_id: str, filename: str):
    """
    Load a BloomFilter from the Hugging Face Hub.

    Args:
        model_id (str): The repository ID.
        filename (str): The file name of the BloomFilter.

    Returns:
        BloomFilter: The loaded BloomFilter instance.
    """
    return BloomFilter.open(hf_hub_download(repo_id=model_id, filename=filename))

class OCRQAPipeline:
    """
    Pipeline for OCR Quality Assessment using BloomFilters.

    Attributes:
        SUPPORTED_LANGUAGES (set): Set of supported languages.
        lang_model (LangIdentPipeline): Language identification model.
        bloomfilters (dict): Cache for BloomFilter instances.
    """
    def __init__(self):
        """
        Initialize the pipeline by loading supported languages and setting up caches.
        """
        self.SUPPORTED_LANGUAGES = self.get_supported_languages()
        self.lang_model = LangIdentPipeline()  # Initialize LangIdentPipeline here
        self.bloomfilters = {}  # Cache for BloomFilter instances

    def get_supported_languages(self) -> set:
        """
        Retrieve the set of supported languages from the repository.

        Returns:
            set: Supported language codes.
        """
        repo_files = cached_list_repo_files("impresso-project/OCR-quality-assessment-unigram")
        languages = {file.split('-')[-1].split('.')[0] for file in repo_files if file.startswith("ocrqa-wp_v")}
        return languages

    def __call__(self, text: str, language: Optional[str] = None, version: Optional[str] = None, 
                 diagnostics: bool = False, model_id: bool = False, supported_languages: bool = False) -> Dict[str, Union[str, float, Dict]]:
        """
        Process the input text and assess its quality using BloomFilters.

        Args:
            text (str): Input text to process.
            language (Optional[str]): Language code of the text.
            version (Optional[str]): Version of the BloomFilter to use.
            diagnostics (bool): Whether to include diagnostics in the output.
            model_id (bool): Whether to include model ID in the output.
            supported_languages (bool): Whether to include supported languages in the output.

        Returns:
            Dict[str, Union[str, float, Dict]]: Output containing language, score, and optional diagnostics.
        """
        self.language = language
        self.version = version
        self.diagnostics = diagnostics
        self.model_id = model_id
        self.supported_languages = supported_languages
        
        if self.language is None:
            lang_result = self.lang_model(text)  # Use the initialized LangIdentPipeline
            self.language = lang_result["language"]

        if self.language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}")

        if self.version is None:
            repo_files = cached_list_repo_files("impresso-project/OCR-quality-assessment-unigram")
            versions = [
                re.search(r"_v(\d+\.\d+\.\d+)", file).group(1)
                for file in repo_files
                if file.startswith("ocrqa-wp_v") and file.endswith(f"-{self.language}.bloom")
            ]
            self.version = max(versions, key=lambda v: list(map(int, v.split('.'))))

        # Check if BloomFilter for the language and version is already cached
        bloomfilter_key = f"{self.language}_{self.version}"
        if bloomfilter_key not in self.bloomfilters:
            self.bloomfilters[bloomfilter_key] = get_bloomfilter(
                "impresso-project/OCR-quality-assessment-unigram", 
                f"ocrqa-wp_v{self.version}-{self.language}.bloom"
            )
        bf = self.bloomfilters[bloomfilter_key]

        output = self.filter_text(text, bf)

        if self.supported_languages:
            output["supported_languages"] = list(self.SUPPORTED_LANGUAGES)

        return output

    # Define normalization table
    QUOTES_PUNCT = "„•<>!\"#%&'’"
    ASCII_PUNCT = "()*,./:;?"
    BRACKETS_SPECIAL = "[]\\~_{}"
    UNICODE_PUNCT = "\xa1\xab\xb7\xbb\xbf"
    DASH_CARET = "—^`"
    SPECIAL_SYMBOLS = "¦§£="
    HYPHEN = "-"
    DIGITS = "0123456789"

    NORMALIZATION_TABLE = str.maketrans(
        {
            char: " "
            for char in (
                QUOTES_PUNCT
                + ASCII_PUNCT
                + BRACKETS_SPECIAL
                + UNICODE_PUNCT
                + DASH_CARET
                + SPECIAL_SYMBOLS
                + HYPHEN
            )
        }
        | {char: "0" for char in DIGITS}
    )


    def normalize_text(self, s: str, unicode_normalize: Optional[str] = "NFKC") -> str:
        """
        Normalize text by replacing punctuation with spaces and digits with '0'.

        Args:
            s (str): Input text to normalize.
            unicode_normalize (Optional[str]): Unicode normalization form.

        Returns:
            str: Normalized text.
        """
        if unicode_normalize:
            s = unicodedata.normalize(unicode_normalize, s).lower()
        return s.translate(self.NORMALIZATION_TABLE)


    def filter(self, text: str, bloom_filter: BloomFilter):
        """
        Check tokens in the text against the BloomFilter and print diagnostics.

        Args:
            text (str): Input text to filter.
            bloom_filter (BloomFilter): BloomFilter instance to use.
        """
        # Normalize and tokenize text
        normalized_text = self.normalize_text(text)
        tokens = normalized_text.split()

        # Check tokens against the bloom filter
        for token in tokens:
            if self.diagnostics:
                if token in bloom_filter:
                    print(f"'{token}' is in the bloom filter.")
                else:
                    print(f"'{token}' is NOT in the bloom filter.")


    def filter_text(self, DE_TEXT: str, bloom_filter: BloomFilter) -> Dict[str, Union[str, float, Dict[str, Union[List[str], str]]]]:
        """
        Filter the text using the BloomFilter and compute a quality score.

        Args:
            DE_TEXT (str): Input text to filter.
            bloom_filter (BloomFilter): BloomFilter instance to use.

        Returns:
            Dict[str, Union[str, float, Dict[str, Union[List[str], str]]]]: Output containing language, score, and optional diagnostics.
        """
        knowns = set()
        unknowns = set()

        # Normalize and tokenize text
        normalized_text = self.normalize_text(DE_TEXT)
        tokens = normalized_text.split()

        # Check tokens against the bloom filter
        for token in tokens:
            if token in bloom_filter:
                knowns.add(token)
            else:
                unknowns.add(token)

        # Compute the score
        score = len(knowns) / (len(knowns) + len(unknowns)) if (len(knowns) + len(unknowns)) > 0 else 0
        score = round(score, 1)

        output = {"language": self.language, "score": score}

        if self.diagnostics:
            output["diagnostics"] = {
                "known_tokens": list(knowns),
                "unknown_tokens": list(unknowns),
                "model_id": f"ocrqa-wp_v{self.version}-{self.language}"
            }
        elif self.model_id:
            output["model_id"] = f"ocrqa-wp_v{self.version}-{self.language}"

        return output
