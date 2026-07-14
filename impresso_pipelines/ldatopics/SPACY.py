import spacy
import subprocess
import json
import bz2
import gzip
import os
import re
import tarfile
import tempfile
import requests
import shutil  # Add this import for moving directories
import logging
from functools import lru_cache
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


VALID_CORE_RE = re.compile(r"^[a-z]+$")
DEFAULT_BOUNDARY_CHARS = (
    " \t\n\r"
    ".,;:!?()[]{}"
    "\"'"
    "-_\\/|~^=+*@#$%&§°£€¥¢©®™"
    "•■□▲►▼★♦✓†‡¶"
)


def count_ascii_letters(text: str) -> int:
    return sum("a" <= ch <= "z" for ch in text)


def load_translation_table(path: str) -> dict[int, str | None]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_table = data.get("char_normalization")
    if not isinstance(raw_table, dict):
        raise ValueError(
            f"Normalization JSON must contain a char_normalization object: {path}"
        )

    table: dict[int, str | None] = {}
    for source, target in raw_table.items():
        if len(source) != 1:
            raise ValueError(f"Source key must be one character: {source!r}")
        if target is not None and not isinstance(target, str):
            raise ValueError(
                f"Replacement for {source!r} must be string or null, got {target!r}"
            )
        table[ord(source)] = target
    return table


class LemmaNormalizer:
    def __init__(
        self,
        translation_table: dict[int, str | None],
        boundary_chars: str = DEFAULT_BOUNDARY_CHARS,
        min_alpha: int = 3,
        min_alpha_ratio: float = 0.75,
        cache_size: int = 200_000,
    ) -> None:
        self.translation_table = translation_table
        self.boundary_chars = boundary_chars
        self.min_alpha = min_alpha
        self.min_alpha_ratio = min_alpha_ratio
        self.normalize = lru_cache(maxsize=cache_size)(self._normalize_uncached)

    def normalize_chars(self, lemma: str) -> str:
        return lemma.lower().translate(self.translation_table)

    def _normalize_uncached(self, lemma: str) -> str | None:
        base = self.normalize_chars(lemma).strip()
        if not base or any(ch.isdigit() for ch in base):
            return None

        candidate = base.strip(self.boundary_chars)
        if not candidate:
            return None
        candidate = candidate.replace(".", "")
        candidate = candidate.replace("-", "")
        candidate = candidate.replace("'", "")
        if not candidate or VALID_CORE_RE.fullmatch(candidate) is None:
            return None

        alpha_len = count_ascii_letters(candidate)
        if alpha_len < self.min_alpha:
            return None
        if alpha_len / len(base) < self.min_alpha_ratio:
            return None

        return candidate


def load_vocab(path: str) -> set[str]:
    vocab = set()
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            vocab.add(line.rstrip("\n").split("\t", 1)[0])
    return vocab


class SPACY:
    def __init__(self, model_id, language, latest_version, model_config=None):
        self.language = language
        self.latest_version = latest_version
        # load spcay file
        from impresso_pipelines.ldatopics.config import MODEL_URLS  # Lazy import
        model_url = MODEL_URLS[model_id]
        if not model_url:
            raise ValueError(f"No SpaCy model available for {model_id}")
        
        path_to_model = self.download_and_extract_model(model_url)
        self.nlp = spacy.load(path_to_model, disable=["parser", "ner"])

        self.config = model_config or self.load_legacy_config(language, latest_version)
        self.topic_model_id = self.config.get(
            "model_id", f"tm-{language}-all-v{latest_version}"
        )
        preprocessing = self.config.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            preprocessing = {}
        self.preprocessing_mode = preprocessing.get(
            "mode", self.config.get("preprocessing_mode", "v2.0-legacy")
        )
        self.upos_filter = set(
            preprocessing.get(
                "upos_filter",
                self.config.get("upos_filter", self.config.get("uposFilter", [])),
            )
        )
        self.lemmatization_dict = {}
        self.vocab = set()
        self.normalizer = None

        if self.preprocessing_mode == "normalized-lemma-vocab-v1":
            self.load_v3_vocab()
        else:
            self.load_legacy_lemmatization_file(language, latest_version)

    def load_legacy_config(self, language, latest_version):
        config_file = hf_hub_download(
            repo_id="impresso-project/lb-spacy-pos",
            filename=f"tm-{language}-all-v{latest_version}.config.json"
        )
        with open(config_file, "r") as f:
            return json.load(f)

    def load_legacy_lemmatization_file(self, language, latest_version):
        lemmatization_file = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=f"models/tm/tm-{language}-all-v{latest_version}.vocab.lemmatization.tsv.gz"
        )
        with gzip.open(lemmatization_file, "rt", encoding="utf-8") as f:
            for line in f:
                lemma = line.strip().split("\t")
                if len(lemma) > 2:
                    self.lemmatization_dict[lemma[0].lower()] = lemma[2]

    def load_v3_vocab(self):
        artifacts = self.config.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}

        vocab_filename = artifacts.get("vocab", f"{self.topic_model_id}.vocab.tsv.bz2")
        normalization_filename = artifacts.get(
            "char_normalization", f"{self.topic_model_id}.char-normalization.json"
        )
        vocab_path = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=self.model_artifact_path(vocab_filename),
        )
        normalization_path = hf_hub_download(
            repo_id="impresso-project/mallet-topic-inferencer",
            filename=self.model_artifact_path(normalization_filename),
        )

        self.vocab = load_vocab(vocab_path)
        min_lemma_length = int(
            self.config.get("preprocessing", {}).get(
                "min_lemma_length", self.config.get("min_lemma_length", 3)
            )
        )
        self.normalizer = LemmaNormalizer(
            load_translation_table(normalization_path),
            min_alpha=min_lemma_length,
        )

    def model_artifact_path(self, filename: str) -> str:
        return filename if "/" in filename else f"models/tm/{filename}"

    def download_model(self, model_id):
        """Ensures the SpaCy model is installed before use."""
        try:
            spacy.load(model_id)
        except OSError:
            logger.info("Downloading SpaCy model: %s...", model_id)
            subprocess.run(["python", "-m", "spacy", "download", model_id], check=True)

    def download_and_extract_model(self, model_url):
        """Downloads and extracts the SpaCy model tar file to a cache directory."""
        cache_dir = os.path.expanduser("~/.cache/spacy_models")
        os.makedirs(cache_dir, exist_ok=True)

        # Generate a unique filename for the model based on its URL
        model_filename = os.path.basename(model_url)
        cached_model_path = os.path.join(cache_dir, model_filename)

        # Check if the model is already cached
        if os.path.exists(cached_model_path):
            # print(f"Using cached SpaCy model from: {cached_model_path}")
            logger.info("Using cached SpaCy model...")
        else:
            # Download the tar file
            logger.info("Downloading SpaCy model from: %s...", model_url)
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(cached_model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Extract the tar file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        # print(f"Extracting SpaCy model to: {temp_dir}...")
        with tarfile.open(cached_model_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Locate the directory containing the config.cfg file
        for root, dirs, files in os.walk(temp_dir):
            if "config.cfg" in files:
                # Move the model directory to a new location and return its path
                model_dir = root
                final_model_dir = os.path.join(temp_dir, "model")
                shutil.move(model_dir, final_model_dir)
                return final_model_dir

        raise IOError("Could not find config.cfg in the extracted model directory.")

    def __call__(self, text):
        doc = self.nlp(text)

        lemmatized_text = []
        for token in doc:
            pos_tag = self.token_pos(token)
            if pos_tag in self.upos_filter:
                if self.preprocessing_mode == "normalized-lemma-vocab-v1":
                    lemma = token.lemma_ or token.text
                    normalized = self.normalizer.normalize(lemma)
                    if normalized and normalized in self.vocab:
                        lemmatized_text.append(normalized)
                else:
                    lemmatized_text.append(
                        self.lemmatization_dict.get(token.text.lower(), token.lemma_.lower())
                    )


        return lemmatized_text

    def token_pos(self, token):
        if self.language == "lb":
            return token.pos_ or self.map_tag_to_pos(token.tag_)
        return token.pos_

    def map_tag_to_pos(self, tag):
        # Map the fine-grained tags used by your Luxembourgish model to Universal POS tags
        tag_map = {
            "$": "PUNCT",
            "ADJ": "ADJ",
            "AV": "ADV",
            "APPR": "ADP",
            "APPRART": "ADP",
            "D": "DET",
            "KO": "CONJ",
            "N": "NOUN",
            "P": "ADV",
            "TRUNC": "X",
            "AUX": "AUX",
            "V": "VERB",
            "MV": "VERB",
            "PTK": "PART",
            "INTER": "PART",
            "NUM": "NUM",
            "_SP": "SPACE",
        }
        return tag_map.get(tag, "")
