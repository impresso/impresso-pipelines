#!/usr/bin/env python3
"""
This module provides classes for reading input documents from various file formats.
It defines an abstract base class `InputReader` and its concrete implementations
for reading JSONL and CSV files.

Classes:
    InputReader: Abstract base class for input readers.
    JsonlInputReader: Reads input from a JSONL file.
    ImpressoLinguisticProcessingJsonlInputReader: Reads input from an impresso
        linguistic processing JSONL file.
    CsvInputReader: Reads input from a CSV file in Mallet's format.
"""
import collections
import json
from typing import Generator, Tuple, List, Set, Dict
import logging
import csv
from abc import ABC, abstractmethod
from smart_open import open

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)  # Suppress warnings and unnecessary logs


class InputReader(ABC):
    """
    Abstract base class for input readers.
    Subclasses should implement the `read_documents` method to yield documents.
    """

    @abstractmethod
    def read_documents(self) -> Generator[Tuple[str, str], None, None]:
        """
        Yields a tuple of (document_id, text).
        Each implementation should handle its specific input format.
        """
        pass


class JsonlInputReader(InputReader):
    """
    Reads input from a JSONL file, where each line contains a JSON object
    with at least "id" and "text" fields.

    Args:
        input_file (str): Path to the input JSONL file.
        text_key (str): Key for the text field in the JSON objects.
        language_key (str): Key for the language field in the JSON objects.
    """

    def __init__(
        self,
        input_file: str,
        text_key: str = "text",
        language_key: str = "lg",
        docid_key: str = "id",
    ) -> None:
        self.input_file = input_file
        self.text_key = text_key
        self.language_key = language_key
        self.docid_key = docid_key

    def read_documents(self) -> Generator[Tuple[str, str], None, None]:
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                document_id = data[self.docid_key]
                text = data[self.text_key]
                language = data.get(self.language_key, "und")
                yield document_id, language, text


class ImpressoLinguisticProcessingJsonlInputReader(InputReader):
    """
      Reads input from an impresso linguistic processing JSONL file, where each line
      contains a JSON object with the tokenized and PoS-tagged text.

      Args:
          input_file (str): Path to the input JSONL file.
          lang_lemmatization_dict (dict): Dictionary mapping languages to their
            respective lemmatization dictionaries.

      Input Example (omitting "l" if equal to "t"):
      ```
      {
    "ts": "2024-4-9T03:08:05",
    "id": "onsjongen-1947-01-31-a-i0035",
    "sents": [
      {
        "lg": "de",
        "tok": [
          {
            "t": "Glück",
            "p": "NOUN",
            "o": 0
          },
          {
            "t": "um",
            "p": "ADP",
            "o": 6
          },
          ...
          ```

    """

    def __init__(
        self,
        input_file: str,
        lang_lemmatization_dict: dict,
        language_configs: dict,
        ci_id_key: str = "id",
        language_key: str = "lg",
        ci_ids: List[str] | None = None,
    ) -> None:
        self.input_file = input_file
        self.lang_lemmatization_dict = lang_lemmatization_dict
        self.ci_id_key = ci_id_key
        self.language_key = language_key
        self.ci_ids: Set[str] | None = set(ci_ids) if ci_ids else None
        self.language_pos_filter_dict: Dict[str, Set[str]] = {
            lang: set(language_configs[lang].get("upos_filter", []))
            for lang in language_configs
        }
        self.language_configs = language_configs
        log.info(
            "%s",
            self,
        )
        self.stats = collections.Counter()

    def __repr__(self):
        return (
            f"ImpressoLinguisticProcessingJsonlInputReader({self.input_file},"
            f" language_key: {self.language_key},  LemmatizationDictSize:"
            f" {len(self.lang_lemmatization_dict)}, ci_id_key: {self.ci_id_key},"
            f" {self.language_pos_filter_dict})"
        )

    def read_documents(
        self, lemmatization_strategy: str = "v2.0-legacy"
    ) -> Generator[Tuple[str, str], None, None]:
        log.warning("LOG Reading documents from %s", self.input_file)

        if self.input_file.startswith("s3://"):
            from impresso_pipelines.mallet.s3_to_local_stamps import get_s3_client  # Lazy import
            tranport_params = {"client": get_s3_client()}
        else:
            tranport_params = {}
        with open(
            self.input_file, "r", encoding="utf-8", transport_params=tranport_params
        ) as f:
            for line in f:
                data = json.loads(line)
                document_id = data[self.ci_id_key]
                if self.ci_ids and document_id not in self.ci_ids:
                    continue
                sents = data["sents"]
                if not sents:
                    continue
                language = sents[0][self.language_key]

                if language not in self.lang_lemmatization_dict:
                    self.stats[f"unsupported_language: {language}"] += 1
                    continue

                self.stats[f"supported_language: {language}"] += 1
                lowercase_token: bool = self.language_configs[language].get(
                    "lowercase_token", False
                )
                min_lemmas = self.language_configs[language].get("min_lemmas", 10)
                lemma_lookup = self.lang_lemmatization_dict[language]

                posfilter = self.language_pos_filter_dict[language]
                lemmatized_text = []
                # currently only lemmatization v2.0-legacy is supported
                # @TODO support a more flexible case-insensitive lemmatization
                for sent in sents:
                    for token in sent["tok"]:
                        # if posfilter is set, only include tokens with specified pos
                        if posfilter and token["p"] not in posfilter:
                            continue

                        # sometimes the lemma is missing or set to "", then ignore it!
                        if token.get("l") == "":
                            del token["l"]

                        token = token.get("t")

                        # note that the freq_filter.py script used to use the lemma as
                        # the key for the lookup, but this is not correct to do so, but
                        # had limited effects. in the old spacy pipelines the lemma and
                        # token was mostly the same and the loookup was actually done
                        # with the lemma. But with better spacy lemmatization this does
                        # not work anymore! so we use the token as the key for the
                        # lookup
                        if lowercase_token:
                            token = token.lower()
                        lemma = lemma_lookup.get(token)
                        if lemma:
                            lemmatized_text.append(lemma)

                log.debug(
                    "Document %s in language %s has %d lemmas",
                    document_id,
                    language,
                    len(lemmatized_text),
                )
                if len(lemmatized_text) > min_lemmas:
                    self.stats[
                        f"INCLUDED: lang {language}: at_least_{min_lemmas}_lemmas"
                    ] += 1
                else:
                    self.stats[
                        f"EXCLUDED: lang {language}: less_than_{min_lemmas}_lemmas"
                    ] += 1
                    continue

                yield document_id, language, " ".join(lemmatized_text)
        for key, value in sorted(self.stats.items()):
            log.info(f"STATS: {key}: {value}")


class CsvInputReader(InputReader):
    """
    Reads input from a CSV file in Mallet's format (document ID, dummy class, text).
    Assumes that the CSV has three columns: "id", "dummyclass", and "text".
    """

    def __init__(self, input_file: str) -> None:
        self.input_file = input_file

    def read_documents(self) -> Generator[Tuple[str, str], None, None]:
        with open(self.input_file, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row in csv_reader:
                if len(row) < 3:
                    continue
                document_id, lang, text = row[0], row[1], row[2]

                yield document_id, lang.lower(), text
