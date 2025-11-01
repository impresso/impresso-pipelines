from typing import List, Dict, Union, Optional, Set
import unicodedata
from huggingface_hub import hf_hub_download, list_repo_files
from pybloomfilter import BloomFilter
import re

from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline


def get_bloomfilter(model_id: str, filename: str, revision: str = "main") -> BloomFilter:
    """
    Load a BloomFilter from the Hugging Face Hub.

    Args:
        model_id (str): The repository ID.
        filename (str): The file name of the BloomFilter.
        revision (str): The repository revision (branch, tag, or commit). Defaults to "main".

    Returns:
        BloomFilter: The loaded BloomFilter instance.
    """
    return BloomFilter.open(hf_hub_download(repo_id=model_id, filename=filename, revision=revision))

class OCRQAPipeline:
    """
    Pipeline for OCR Quality Assessment using BloomFilters.

    Attributes:
        repo_id (str): The Hugging Face repository ID for OCR quality assessment models.
        revision (str): The repository revision (branch, tag, or commit).
        SUPPORTED_LANGUAGES (set): Set of supported languages.
        lang_model (LangIdentPipeline): Language identification model.
        bloomfilters (dict): Cache for BloomFilter instances.
        repo_files (list): List of files in the repository (cached at initialization).
    """
    
    DEFAULT_REPO_ID: str = "impresso-project/OCR-quality-assessment-unigram"
    DEFAULT_REVISION: str = "main"
    
    def __init__(self, repo_id: Optional[str] = None, revision: str = "main") -> None:
        """
        Initialize the pipeline by loading supported languages and setting up caches.
        
        Args:
            repo_id (Optional[str]): The Hugging Face repository ID. 
                                    Defaults to "impresso-project/OCR-quality-assessment-unigram".
            revision (str): The repository revision (branch, tag, or commit). Defaults to "main".
        """
        self.repo_id: str = repo_id or self.DEFAULT_REPO_ID
        self.revision: str = revision
        
        self.repo_files: List[str] = list_repo_files(self.repo_id, revision=self.revision)
        self.SUPPORTED_LANGUAGES: Set[str] = self._get_supported_languages()
        self.lang_model: LangIdentPipeline = LangIdentPipeline()
        self.bloomfilters: Dict[str, BloomFilter] = {}

    def _is_bloomfilter_file(self, filename: str) -> bool:
        """
        Check if a filename matches the BloomFilter naming pattern.
        
        This method can be overridden in subclasses to support different file naming conventions.

        Args:
            filename (str): The filename to check.

        Returns:
            bool: True if the filename matches the BloomFilter pattern.
        """
        return filename.startswith("ocrqa-wp_v") and filename.endswith(".bloom")

    def _extract_language_from_filename(self, filename: str) -> str:
        """
        Extract the language code from a BloomFilter filename.
        
        This method can be overridden in subclasses to support different file naming conventions.

        Args:
            filename (str): The filename to parse.

        Returns:
            str: The language code.
        """
        return filename.split('-')[-1].split('.')[0]

    def _extract_version_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract the version string from a BloomFilter filename.
        
        This method can be overridden in subclasses to support different file naming conventions.

        Args:
            filename (str): The filename to parse.

        Returns:
            Optional[str]: The version string (e.g., "1.0.5"), or None if not found.
        """
        match = re.search(r"_v(\d+\.\d+\.\d+)", filename)
        return match.group(1) if match else None

    def _get_supported_languages(self) -> Set[str]:
        """
        Retrieve the set of supported languages from the repository files.

        Returns:
            Set[str]: Supported language codes.
        """
        languages: Set[str] = {
            self._extract_language_from_filename(file)
            for file in self.repo_files 
            if self._is_bloomfilter_file(file)
        }
        return languages

    def _get_available_versions(self, language: str) -> List[str]:
        """
        Get all available BloomFilter versions for a specific language.
        
        This method can be overridden in subclasses to support different version retrieval strategies.

        Args:
            language (str): The language code.

        Returns:
            List[str]: List of available version strings.
        """
        versions: List[str] = []
        for file in self.repo_files:
            if self._is_bloomfilter_file(file) and file.endswith(f"-{language}.bloom"):
                version = self._extract_version_from_filename(file)
                if version:
                    versions.append(version)
        return versions

    def _select_latest_version(self, versions: List[str]) -> str:
        """
        Select the latest version from a list of version strings.
        
        This method can be overridden in subclasses to implement different version selection logic.

        Args:
            versions (List[str]): List of version strings (e.g., ["1.0.0", "1.0.5", "2.0.0"]).

        Returns:
            str: The selected version string.
        
        Raises:
            ValueError: If the versions list is empty.
        """
        if not versions:
            raise ValueError("No versions available")
        return max(versions, key=lambda v: list(map(int, v.split('.'))))

    def _build_bloomfilter_filename(self, version: str, language: str) -> str:
        """
        Build the BloomFilter filename for a given version and language.
        
        This method can be overridden in subclasses to support different file naming conventions.

        Args:
            version (str): The version string.
            language (str): The language code.

        Returns:
            str: The BloomFilter filename.
        """
        return f"ocrqa-wp_v{version}-{language}.bloom"

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
        
        Raises:
            ValueError: If the language is not supported.
            Exception: If there are issues downloading or loading the BloomFilter.
        """
        # Use local variables instead of instance variables to avoid state pollution
        detected_language: Optional[str] = language
        selected_version: Optional[str] = version
        
        try:
            # Detect language if not provided
            if detected_language is None:
                lang_result: Dict[str, str] = self.lang_model(text)
                detected_language = lang_result["language"]

            if detected_language not in self.SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {detected_language}. Supported languages: {sorted(self.SUPPORTED_LANGUAGES)}")

            # Determine version if not provided
            if selected_version is None:
                try:
                    versions: List[str] = self._get_available_versions(detected_language)
                    if not versions:
                        raise ValueError(f"No BloomFilter versions found for language: {detected_language}")
                    selected_version = self._select_latest_version(versions)
                except Exception as e:
                    raise Exception(f"Failed to retrieve BloomFilter versions: {str(e)}")

            # Check if BloomFilter for the language and version is already cached
            bloomfilter_key: str = f"{detected_language}_{selected_version}"
            if bloomfilter_key not in self.bloomfilters:
                try:
                    bloomfilter_filename: str = self._build_bloomfilter_filename(selected_version, detected_language)
                    self.bloomfilters[bloomfilter_key] = get_bloomfilter(
                        self.repo_id, 
                        bloomfilter_filename,
                        self.revision
                    )
                except Exception as e:
                    raise Exception(f"Failed to download or load BloomFilter for {detected_language} v{selected_version}: {str(e)}")
            
            bf: BloomFilter = self.bloomfilters[bloomfilter_key]

            output: Dict[str, Union[str, float, List[str]]] = self.filter_text(text, bf, detected_language, selected_version, diagnostics, model_id)

            if supported_languages:
                output["supported_languages"] = sorted(self.SUPPORTED_LANGUAGES)

            return output
        
        except ValueError:
            # Re-raise ValueError as-is (user input errors)
            raise
        except Exception as e:
            # Wrap other exceptions with more context
            raise Exception(f"OCR quality assessment failed: {str(e)}")

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


    def filter(self, text: str, bloom_filter: BloomFilter) -> None:
        """
        Check tokens in the text against the BloomFilter and print diagnostics.

        Args:
            text (str): Input text to filter.
            bloom_filter (BloomFilter): BloomFilter instance to use.
        """
        # Normalize and tokenize text
        normalized_text: str = self.normalize_text(text)
        tokens: List[str] = normalized_text.split()

        # Check tokens against the bloom filter
        for token in tokens:
            if self.diagnostics:
                if token in bloom_filter:
                    print(f"'{token}' is in the bloom filter.")
                else:
                    print(f"'{token}' is NOT in the bloom filter.")


    def filter_text(self, text: str, bloom_filter: BloomFilter, language: str, version: str, 
                    include_diagnostics: bool, include_model_id: bool) -> Dict[str, Union[str, float, Dict[str, Union[List[str], str]]]]:
        """
        Filter the text using the BloomFilter and compute a quality score.

        Args:
            text (str): Input text to filter.
            bloom_filter (BloomFilter): BloomFilter instance to use.
            language (str): Language code of the text.
            version (str): Version of the BloomFilter being used.
            include_diagnostics (bool): Whether to include diagnostics in the output.
            include_model_id (bool): Whether to include model ID in the output.

        Returns:
            Dict[str, Union[str, float, Dict[str, Union[List[str], str]]]]: Output containing language, score, and optional diagnostics.
        """
        knowns: Set[str] = set()
        unknowns: Set[str] = set()

        # Normalize and tokenize text
        normalized_text: str = self.normalize_text(text)
        tokens: List[str] = normalized_text.split()

        # Check tokens against the bloom filter
        for token in tokens:
            if token in bloom_filter:
                knowns.add(token)
            else:
                unknowns.add(token)

        # Compute the score
        score: float = len(knowns) / (len(knowns) + len(unknowns)) if (len(knowns) + len(unknowns)) > 0 else 0
        score = round(score, 1)

        output: Dict[str, Union[str, float, Dict[str, Union[List[str], str]]]] = {"language": language, "score": score}

        if include_diagnostics:
            output["diagnostics"] = {
                "known_tokens": sorted(knowns),
                "unknown_tokens": sorted(unknowns),
                "model_id": f"ocrqa-wp_v{version}-{language}"
            }
        elif include_model_id:
            output["model_id"] = f"ocrqa-wp_v{version}-{language}"

        return output
