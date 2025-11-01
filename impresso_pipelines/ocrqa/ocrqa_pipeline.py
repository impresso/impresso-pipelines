from typing import List, Dict, Union, Optional, Set
import unicodedata
from huggingface_hub import hf_hub_download, list_repo_files
from pybloomfilter import BloomFilter
import re

from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline


# ===== Normalization Tables for Different BloomFilter Versions =====

# v1.x.x normalization constants
_V1_QUOTES_PUNCT = "„•<>!\"#%&''"
_V1_ASCII_PUNCT = "()*,./:;?"
_V1_BRACKETS_SPECIAL = "[]\\~_{}"
_V1_UNICODE_PUNCT = "\xa1\xab\xb7\xbb\xbf"
_V1_DASH_CARET = "—^`"
_V1_SPECIAL_SYMBOLS = "¦§£="
_V1_HYPHEN = "-"
_V1_DIGITS = "0123456789"

# v1.x.x normalization table
_V1_NORMALIZATION_TABLE = str.maketrans(
    {
        char: " "
        for char in (
            _V1_QUOTES_PUNCT
            + _V1_ASCII_PUNCT
            + _V1_BRACKETS_SPECIAL
            + _V1_UNICODE_PUNCT
            + _V1_DASH_CARET
            + _V1_SPECIAL_SYMBOLS
            + _V1_HYPHEN
        )
    }
    | {char: "0" for char in _V1_DIGITS}
)

# v2.x.x normalization constants
_V2_PRIVATE_CHAR = "\ue000"  # Private-use Unicode character for Luxembourgish apostrophes
_V2_APOSTROPHES = "''`′""ʻ"
_V2_QUOTES_PUNCT = '„•<>!"%&'
_V2_ASCII_PUNCT = "()*,./:;?"
_V2_CHARS_TO_SPACE = _V2_APOSTROPHES + _V2_QUOTES_PUNCT + _V2_ASCII_PUNCT
_V2_BRACKETS_SPECIAL = "[]\\~_{}"
_V2_UNICODE_PUNCT = "\xa1\xab\xb7\xbb\xbf"
_V2_DASH_CARET = "—^"
_V2_SPECIAL_SYMBOLS = "¦§£=|#•■+"
_V2_HYPHEN = "-"
_V2_OCR_ARTIFACTS = _V2_BRACKETS_SPECIAL + _V2_UNICODE_PUNCT + _V2_DASH_CARET + _V2_SPECIAL_SYMBOLS + _V2_HYPHEN
_V2_DIGITS = "0123456789"

# v2.x.x normalization table (OCR artifacts are handled separately via regex)
_V2_NORMALIZATION_TABLE = str.maketrans(
    {char: " " for char in _V2_CHARS_TO_SPACE}
    | {char: "0" for char in _V2_DIGITS}
)


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
    DEFAULT_SCORE_PRECISION: int = 2
    
    def __init__(self, repo_id: Optional[str] = None, revision: str = "main", score_precision: int = 2) -> None:
        """
        Initialize the pipeline by loading supported languages and setting up caches.
        
        Args:
            repo_id (Optional[str]): The Hugging Face repository ID. 
                                    Defaults to "impresso-project/OCR-quality-assessment-unigram".
            revision (str): The repository revision (branch, tag, or commit). Defaults to "main".
            score_precision (int): Number of decimal places for rounding the score. Defaults to 2.
        """
        self.repo_id: str = repo_id or self.DEFAULT_REPO_ID
        self.revision: str = revision
        self.score_precision: int = score_precision
        
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

            if selected_version is None:
                try:
                    versions: List[str] = self._get_available_versions(detected_language)
                    if not versions:
                        raise ValueError(f"No BloomFilter versions found for language: {detected_language}")
                    selected_version = self._select_latest_version(versions)
                except Exception as e:
                    raise Exception(f"Failed to retrieve BloomFilter versions: {str(e)}")

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

            output: Dict[str, Union[str, float, List[str]]] = self.filter_text(
                text, bf, detected_language, selected_version, diagnostics, model_id
            )

            if supported_languages:
                output["supported_languages"] = sorted(self.SUPPORTED_LANGUAGES)

            return output
        
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"OCR quality assessment failed: {str(e)}")

    def _get_normalization_table(self, major_version: int) -> dict:
        """
        Get the normalization table for a specific BloomFilter major version.
        
        This method can be overridden in subclasses to add support for additional versions
        or modify existing normalization tables.

        Args:
            major_version (int): The major version number of the BloomFilter.

        Returns:
            dict: The normalization translation table for str.translate().
        
        Raises:
            ValueError: If the major version is not supported.
        """
        if major_version == 1:
            return _V1_NORMALIZATION_TABLE
        elif major_version == 2:
            return _V2_NORMALIZATION_TABLE
        else:
            raise ValueError(f"Unsupported BloomFilter major version: {major_version}. "
                           f"Supported versions: 1, 2.")

    def _extract_major_version(self, version: str) -> int:
        """
        Extract the major version number from a version string.

        Args:
            version (str): Version string (e.g., "1.0.5", "2.1.3").

        Returns:
            int: The major version number.
        """
        return int(version.split('.')[0])

    def normalize_text(self, s: str, version: str, language: Optional[str] = None, 
                      unicode_normalize: Optional[str] = "NFKC", lowercase: bool = True) -> str:
        """
        Normalize text using the appropriate normalization table for the BloomFilter version.

        Args:
            s (str): Input text to normalize.
            version (str): BloomFilter version string (e.g., "1.0.5", "2.0.0").
            language (Optional[str]): Language code (e.g., "lb" for Luxembourgish). Used for v2+ special handling.
            unicode_normalize (Optional[str]): Unicode normalization form.
            lowercase (bool): Whether to apply lowercasing. Defaults to True.

        Returns:
            str: Normalized text.
        """
        major_version: int = self._extract_major_version(version)
        
        # Apply Unicode normalization first
        if unicode_normalize:
            s = unicodedata.normalize(unicode_normalize, s)
        
        # Apply lowercasing if requested (v1 does it here, v2 does it in subtokens)
        if lowercase and major_version == 1:
            s = s.lower()
        
        # v2+ Luxembourgish-specific handling: preserve word-internal apostrophes
        if major_version >= 2 and language == "lb":
            s = re.sub(rf"(?<=[oe])[{_V2_APOSTROPHES}](?=\S)", _V2_PRIVATE_CHAR, s)
        
        # v2+ OCR artifact preservation: add spaces around artifacts
        if major_version >= 2:
            for char in _V2_OCR_ARTIFACTS:
                escaped_char = re.escape(char)
                s = re.sub(rf"{escaped_char}+", lambda m: f" {m.group()} ", s)
        
        # Apply version-specific normalization table
        normalization_table: dict = self._get_normalization_table(major_version)
        s = s.translate(normalization_table)
        
        # v2+ Luxembourgish post-processing: restore apostrophes
        if major_version >= 2 and language == "lb":
            s = s.replace(_V2_PRIVATE_CHAR, "'")
        
        return s

    def subtokens(self, text: str, version: str, language: Optional[str] = None,
                  unicode_normalize: Optional[str] = "NFKC", 
                  min_length: int = 1, lowercase: bool = True) -> List[str]:
        """
        Normalize and tokenize text into subtokens.
        
        OCR artifact characters become separate tokens, allowing them to be
        flagged as errors when they don't appear in the lexicon.

        Args:
            text (str): Input text to tokenize.
            version (str): BloomFilter version string (e.g., "1.0.5", "2.0.0").
            language (Optional[str]): Language code (e.g., "lb" for Luxembourgish).
            unicode_normalize (Optional[str]): Unicode normalization form (default: 'NFKC').
            min_length (int): Minimum token length to include (default: 1).
            lowercase (bool): Apply lowercasing as first step (default: True).

        Returns:
            List[str]: List of normalized tokens.
            
        Examples:
            >>> pipeline = OCRQAPipeline()
            >>> pipeline.subtokens("hello~world", version="2.0.0")
            ['hello', '~', 'world']
            >>> pipeline.subtokens("Price: £100", version="2.0.0")
            ['price', '£', '000']
            >>> pipeline.subtokens("ge'nt", version="2.0.0", language="lb")
            ["ge'nt"]
        """
        major_version: int = self._extract_major_version(version)
        
        # v2+ applies lowercasing before normalization for better consistency
        if lowercase and major_version >= 2:
            text = text.lower()
        
        # Normalize the text (v1 lowercases here, v2 already lowercased above)
        normalized = self.normalize_text(
            text, version, language, unicode_normalize, 
            lowercase=(lowercase if major_version == 1 else False)
        )
        
        # Split into tokens
        tokens = normalized.split()
        
        # Filter by minimum length if requested
        if min_length <= 1:
            return tokens
        return [tok for tok in tokens if len(tok) >= min_length]


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

        # Use subtokens() for proper v2-compatible tokenization
        tokens: List[str] = self.subtokens(text, version, language)

        for token in tokens:
            if token in bloom_filter:
                knowns.add(token)
            else:
                unknowns.add(token)

        score: float = len(knowns) / (len(knowns) + len(unknowns)) if (len(knowns) + len(unknowns)) > 0 else 0
        score = round(score, self.score_precision)

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
