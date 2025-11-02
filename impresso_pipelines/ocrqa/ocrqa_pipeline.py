"""
OCR Quality Assessment pipeline using BloomFilter-based lexicon matching.

This module provides a complete pipeline for assessing the quality of OCR-processed text
by comparing tokens against language-specific lexicons stored as BloomFilters. It supports
multiple languages and different normalization strategies across BloomFilter versions.

Key features:
- Automatic language detection using floret models
- Version-aware text normalization (v1.x.x and v2.x.x BloomFilters)
- Luxembourgish-specific apostrophe handling (v2+)
- OCR artifact detection and isolation (v2+)
- Quality scoring based on lexicon coverage

Supported languages: Multiple European languages including German, French, English, 
Italian, Luxembourgish, and others (language support depends on available BloomFilters).

Example usage:
    >>> pipeline = OCRQAPipeline()
    >>> result = pipeline("This is some OCR text to analyze")
    >>> print(f"Quality score: {result['score']}")
    Quality score: 0.95
    
    >>> # With diagnostics
    >>> result = pipeline(
    ...     "Text with ~artifacts# and errors",
    ...     language="en",
    ...     diagnostics=True
    ... )
    >>> print(result['diagnostics']['unknown_tokens'])
    ['~', '#', 'artifacts']

Version differences:
- v1.x.x: Basic normalization with punctuation and digit handling
- v2.x.x: Enhanced OCR artifact preservation, Luxembourgish apostrophe support

For detailed normalization behavior, see the normalize_text() and subtokens() functions.
"""

from typing import List, Dict, Union, Optional, Set
import unicodedata
from huggingface_hub import hf_hub_download, list_repo_files
from pybloomfilter import BloomFilter
import re
import logging

from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline

logger = logging.getLogger(__name__)


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


# ===== Module-Level Utility Functions =====

def _extract_major_version(version: str) -> int:
    """
    Extract the major version number from a semantic version string.

    Args:
        version: Version string in semantic versioning format (e.g., "1.0.5", "2.1.3")

    Returns:
        The major version number as an integer
        
    Example:
        >>> _extract_major_version("2.1.3")
        2
        >>> _extract_major_version("1.0.5")
        1
    """
    return int(version.split('.')[0])


def _get_normalization_table(major_version: int) -> Dict[int, str]:
    """
    Get the character normalization translation table for a BloomFilter version.
    
    Returns a str.translate()-compatible dictionary mapping Unicode codepoints
    to their normalized forms. Different versions use different normalization rules.

    Args:
        major_version: The major version number of the BloomFilter (1 or 2)

    Returns:
        Translation table mapping character codepoints to replacement strings
    
    Raises:
        ValueError: If the major version is not supported (not 1 or 2)
        
    Example:
        >>> table = _get_normalization_table(2)
        >>> "hello!".translate(table)
        'hello '
    """
    if major_version == 1:
        return _V1_NORMALIZATION_TABLE
    elif major_version == 2:
        return _V2_NORMALIZATION_TABLE
    else:
        raise ValueError(f"Unsupported BloomFilter major version: {major_version}. "
                       f"Supported versions: 1, 2.")


def normalize_text(s: str, version: str, language: Optional[str] = None, 
                   unicode_normalize: Optional[str] = "NFKC") -> str:
    """
    Normalize text using version-specific rules for BloomFilter comparison.
    
    Applies Unicode normalization, handles OCR artifacts, and performs character
    substitution based on the BloomFilter version. For v2+, special handling is
    applied for Luxembourgish apostrophes and OCR artifacts are isolated with spaces.
    
    **Important**: This function does NOT lowercase. Lowercasing should be done
    before calling this function (e.g., in subtokens()) for consistency across versions.

    Args:
        s: Input text to normalize
        version: BloomFilter version string (e.g., "1.0.5", "2.0.0")
        language: Language code (e.g., "lb" for Luxembourgish). Used for v2+ special handling.
        unicode_normalize: Unicode normalization form ('NFKC', 'NFC', 'NFD', 'NFKD', or None)

    Returns:
        Normalized text with punctuation converted to spaces, digits to '0', 
        and (v2+) OCR artifacts isolated with surrounding spaces
        
    Example:
        >>> normalize_text("Hello, world!", version="2.0.0")
        'Hello  world '
        >>> normalize_text("Price: £100", version="2.0.0")
        'Price   £  000'
        >>> normalize_text("ge'nt", version="2.0.0", language="lb")
        "ge'nt"
    """
    major_version: int = _extract_major_version(version)
    
    # Apply Unicode normalization first
    if unicode_normalize:
        s = unicodedata.normalize(unicode_normalize, s)
    
    # v2+ Luxembourgish-specific handling: preserve word-internal apostrophes
    if major_version >= 2 and language == "lb":
        s = re.sub(rf"(?<=[oe])[{_V2_APOSTROPHES}](?=\S)", _V2_PRIVATE_CHAR, s)
    
    # v2+ OCR artifact preservation: add spaces around artifacts
    if major_version >= 2:
        for char in _V2_OCR_ARTIFACTS:
            escaped_char = re.escape(char)
            s = re.sub(rf"{escaped_char}+", lambda m: f" {m.group()} ", s)
    
    # Apply version-specific normalization table
    normalization_table: Dict[int, str] = _get_normalization_table(major_version)
    s = s.translate(normalization_table)
    
    # v2+ Luxembourgish post-processing: restore apostrophes
    if major_version >= 2 and language == "lb":
        s = s.replace(_V2_PRIVATE_CHAR, "'")
    
    return s


def subtokens(text: str, version: str, language: Optional[str] = None,
              unicode_normalize: Optional[str] = "NFKC", 
              min_length: int = 1, lowercase: bool = True) -> List[str]:
    """
    Normalize and tokenize text into subtokens for BloomFilter lookup.
    
    Applies lowercasing (optional), normalization, and whitespace-based tokenization.
    OCR artifact characters become separate tokens (v2+), allowing them to be
    flagged as errors when they don't appear in the lexicon.

    Args:
        text: Input text to tokenize
        version: BloomFilter version string (e.g., "1.0.5", "2.0.0")
        language: Language code (e.g., "lb" for Luxembourgish)
        unicode_normalize: Unicode normalization form (default: 'NFKC')
        min_length: Minimum token length to include (default: 1)
        lowercase: Apply lowercasing as first step (default: True)

    Returns:
        List of normalized tokens (whitespace-separated after normalization)
        
    Examples:
        >>> subtokens("Hello~World", version="2.0.0")
        ['hello', '~', 'world']
        >>> subtokens("Price: £100", version="2.0.0")
        ['price', '£', '000']
        >>> subtokens("ge'nt", version="2.0.0", language="lb")
        ["ge'nt"]
        >>> subtokens("hi", version="2.0.0", min_length=3)
        []
    """
    # Apply lowercasing before normalization (consistent for both v1 and v2)
    if lowercase:
        text = text.lower()
    
    # Normalize and tokenize
    tokens = normalize_text(text, version, language, unicode_normalize).split()
    
    # Filter by minimum length if needed
    if min_length > 1:
        tokens = [tok for tok in tokens if len(tok) >= min_length]
    
    return tokens


def get_bloomfilter(model_id: str, filename: str, revision: str = "main") -> BloomFilter:
    """
    Download and load a BloomFilter from the Hugging Face Hub.
    
    Uses Hugging Face's caching mechanism to avoid redundant downloads.

    Args:
        model_id: The Hugging Face repository ID (e.g., "impresso-project/OCR-quality-assessment-unigram")
        filename: The BloomFilter filename to download (e.g., "ocrqa-wp_v2.0.0-en.bloom")
        revision: The repository revision - branch, tag, or commit hash (default: "main")

    Returns:
        Loaded BloomFilter instance ready for membership testing
        
    Raises:
        Exception: If download or loading fails
        
    Example:
        >>> bf = get_bloomfilter(
        ...     "impresso-project/OCR-quality-assessment-unigram",
        ...     "ocrqa-wp_v2.0.0-en.bloom"
        ... )
        >>> "hello" in bf
        True
    """
    return BloomFilter.open(hf_hub_download(repo_id=model_id, filename=filename, revision=revision))


class OCRQAPipeline:
    """
    OCR Quality Assessment pipeline using BloomFilter-based lexicon matching.
    
    This pipeline evaluates OCR text quality by comparing normalized tokens against
    language-specific lexicons stored as BloomFilters. It automatically detects language,
    selects appropriate BloomFilter versions, and computes quality scores based on
    the proportion of recognized tokens.
    
    The pipeline supports multiple BloomFilter versions with different normalization
    strategies, and automatically caches loaded BloomFilters for efficient reuse.

    Attributes:
        repo_id: Hugging Face repository ID for OCR quality assessment models
        revision: Repository revision (branch, tag, or commit)
        score_precision: Number of decimal places for quality score rounding
        repo_files: List of files in the repository (cached at initialization)
        SUPPORTED_LANGUAGES: Set of available language codes
        lang_model: Language identification pipeline
        bloomfilters: Cache of loaded BloomFilter instances
        
    Example:
        >>> pipeline = OCRQAPipeline()
        >>> result = pipeline("This is good quality OCR text")
        >>> print(f"{result['language']}: {result['score']}")
        en: 0.95
        
        >>> # With specific version and diagnostics
        >>> result = pipeline(
        ...     "Text with errors",
        ...     language="en",
        ...     version="2.0.0",
        ...     diagnostics=True
        ... )
        >>> print(result['diagnostics']['unknown_tokens'])
    """
    
    DEFAULT_REPO_ID: str = "impresso-project/OCR-quality-assessment-unigram"
    DEFAULT_REVISION: str = "main"
    DEFAULT_SCORE_PRECISION: int = 2
    
    def __init__(self, repo_id: Optional[str] = None, revision: str = "main", score_precision: int = 2) -> None:
        """
        Initialize the OCR Quality Assessment pipeline.
        
        Connects to Hugging Face Hub, retrieves available BloomFilter files,
        determines supported languages, and initializes the language detection model.
        
        Args:
            repo_id: Hugging Face repository ID. If None, uses DEFAULT_REPO_ID 
                    ("impresso-project/OCR-quality-assessment-unigram")
            revision: Repository revision - branch, tag, or commit hash (default: "main")
            score_precision: Number of decimal places for score rounding (default: 2)
            
        Raises:
            Exception: If repository access or language detection initialization fails
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
        
        Default pattern: files starting with "ocrqa-wp_v" and ending with ".bloom"
        Override this method in subclasses for custom naming conventions.

        Args:
            filename: The filename to check

        Returns:
            True if the filename matches the BloomFilter pattern, False otherwise
            
        Example:
            >>> pipeline._is_bloomfilter_file("ocrqa-wp_v2.0.0-en.bloom")
            True
            >>> pipeline._is_bloomfilter_file("other-file.txt")
            False
        """
        return filename.startswith("ocrqa-wp_v") and filename.endswith(".bloom")

    def _extract_language_from_filename(self, filename: str) -> str:
        """
        Extract the language code from a BloomFilter filename.
        
        Default pattern: extracts the last component before ".bloom" extension.
        Override this method in subclasses for custom naming conventions.

        Args:
            filename: The filename to parse (e.g., "ocrqa-wp_v2.0.0-en.bloom")

        Returns:
            The extracted language code (e.g., "en")
            
        Example:
            >>> pipeline._extract_language_from_filename("ocrqa-wp_v2.0.0-en.bloom")
            'en'
        """
        return filename.split('-')[-1].split('.')[0]

    def _extract_version_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract the semantic version string from a BloomFilter filename.
        
        Default pattern: extracts version matching "_vX.Y.Z" format.
        Override this method in subclasses for custom naming conventions.

        Args:
            filename: The filename to parse (e.g., "ocrqa-wp_v2.0.0-en.bloom")

        Returns:
            The version string (e.g., "2.0.0"), or None if not found
            
        Example:
            >>> pipeline._extract_version_from_filename("ocrqa-wp_v2.0.0-en.bloom")
            '2.0.0'
            >>> pipeline._extract_version_from_filename("invalid-name.bloom")
            None
        """
        match = re.search(r"_v(\d+\.\d+\.\d+)", filename)
        return match.group(1) if match else None

    def _get_supported_languages(self) -> Set[str]:
        """
        Retrieve the set of supported languages from repository BloomFilter files.
        
        Scans all files in the repository, identifies BloomFilter files, and extracts
        their language codes to build the set of supported languages.

        Returns:
            Set of language codes (e.g., {'en', 'de', 'fr', 'it'})
            
        Note:
            This method is called during initialization to populate SUPPORTED_LANGUAGES.
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
        
        Scans repository files to find all BloomFilter versions available for the
        given language. Override this method in subclasses for custom version discovery.

        Args:
            language: The language code (e.g., "en", "de", "fr")

        Returns:
            List of version strings (e.g., ["1.0.0", "1.0.5", "2.0.0"])
            
        Example:
            >>> pipeline._get_available_versions("en")
            ['1.0.0', '1.0.5', '2.0.0']
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
        Select the latest version from a list of semantic version strings.
        
        Compares versions numerically (e.g., "2.0.0" > "1.0.5") and returns the highest.
        Override this method in subclasses for custom version selection logic.

        Args:
            versions: List of semantic version strings (e.g., ["1.0.0", "1.0.5", "2.0.0"])

        Returns:
            The selected (latest) version string
        
        Raises:
            ValueError: If the versions list is empty
            
        Example:
            >>> pipeline._select_latest_version(["1.0.0", "2.0.0", "1.0.5"])
            '2.0.0'
        """
        if not versions:
            raise ValueError("No versions available")
        return max(versions, key=lambda v: list(map(int, v.split('.'))))

    def _build_bloomfilter_filename(self, version: str, language: str) -> str:
        """
        Build the BloomFilter filename for a given version and language.
        
        Default format: "ocrqa-wp_vX.Y.Z-LANG.bloom"
        Override this method in subclasses for custom naming conventions.

        Args:
            version: The semantic version string (e.g., "2.0.0")
            language: The language code (e.g., "en")

        Returns:
            The complete BloomFilter filename
            
        Example:
            >>> pipeline._build_bloomfilter_filename("2.0.0", "en")
            'ocrqa-wp_v2.0.0-en.bloom'
        """
        return f"ocrqa-wp_v{version}-{language}.bloom"

    def __call__(self, text: str, language: Optional[str] = None, version: Optional[str] = None, 
                 diagnostics: bool = False, model_id: bool = False, supported_languages: bool = False) -> Dict[str, Union[str, float, List[str], Dict]]:
        """
        Assess OCR quality of input text using BloomFilter lexicon matching.
        
        Main entry point for the pipeline. Detects language if not specified, selects
        appropriate BloomFilter version, and computes quality score based on token
        recognition rate.

        Args:
            text: Input text to assess
            language: Language code (e.g., "en"). Auto-detected if None.
            version: BloomFilter version (e.g., "2.0.0"). Latest version used if None.
            diagnostics: If True, includes known/unknown tokens in output
            model_id: If True, includes BloomFilter model ID in output
            supported_languages: If True, includes list of supported languages in output

        Returns:
            Dictionary containing:
                - language (str): Detected or specified language code
                - score (float): Quality score (0.0-1.0, proportion of recognized tokens)
                - diagnostics (Dict): Only if diagnostics=True, contains:
                    - known_tokens (List[str]): Tokens found in lexicon
                    - unknown_tokens (List[str]): Tokens not found in lexicon
                    - model_id (str): BloomFilter model identifier
                - model_id (str): Only if model_id=True (and diagnostics=False)
                - supported_languages (List[str]): Only if supported_languages=True
        
        Raises:
            ValueError: If language is not supported or no BloomFilter versions found
            Exception: If BloomFilter download/loading fails or processing errors occur
            
        Example:
            >>> pipeline = OCRQAPipeline()
            >>> result = pipeline("Good quality text")
            >>> print(f"Score: {result['score']}")
            Score: 0.95
            
            >>> # With diagnostics
            >>> result = pipeline("text with ~errors#", diagnostics=True)
            >>> print(result['diagnostics']['unknown_tokens'])
            ['~', '#']
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


    def filter_text(self, text: str, bloom_filter: BloomFilter, language: str, version: str, 
                    include_diagnostics: bool, include_model_id: bool) -> Dict[str, Union[str, float, List[str], Dict[str, Union[List[str], str]]]]:
        """
        Filter text tokens through BloomFilter and compute quality score.
        
        Tokenizes text using version-appropriate normalization, checks each token
        against the BloomFilter, and calculates the proportion of recognized tokens.

        Args:
            text: Input text to filter
            bloom_filter: Loaded BloomFilter instance for lexicon lookup
            language: Language code of the text
            version: BloomFilter version string for proper normalization
            include_diagnostics: Whether to include token lists in output
            include_model_id: Whether to include model identifier in output

        Returns:
            Dictionary containing:
                - language (str): The language code
                - score (float): Quality score (0.0-1.0)
                - diagnostics (Dict): Only if include_diagnostics=True
                - model_id (str): Only if include_model_id=True (and not diagnostics)
                
        Note:
            Uses module-level subtokens() function for version-compatible tokenization.
        """
        knowns: Set[str] = set()
        unknowns: Set[str] = set()

        # Use module-level subtokens() for proper v2-compatible tokenization
        tokens: List[str] = subtokens(text, version, language)

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
