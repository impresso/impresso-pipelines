#!/usr/bin/python3

"""
This script performs vectorization and topic inference using Mallet models. It accepts a
raw JSONL file, identifies the language of the text, and applies the corresponding
Mallet model for topic inference. It also supports other input formats through a
flexible InputReader abstraction (e.g., CSV, JSONL).

Key Features:
- Handles multiple languages in a single run without calling Mallet multiple times.
- Supports various input formats (JSONL, CSV).
- Outputs results in JSONL or CSV format.

Classes:
- MalletVectorizer: Handles text-to-Mallet vectorization.
- LanguageInferencer: Performs topic inference using a Mallet inferencer and the vectorizer.
- InputReader (abstract class): Defines the interface for reading input documents.
- JsonlInputReader: Reads input from JSONL files.
- CsvInputReader: Reads input from CSV files (Mallet format).
- MalletTopicInferencer: Coordinates the process, identifies language, and manages inference.

Usage: python mallet_topic_inferencer.py -h
"""

import collections

import jpype
import jpype.imports
from dotenv import load_dotenv

import os
import logging
import argparse
import json
import csv
import tempfile

from typing import List, Dict, Generator, Optional, Set, Iterable, Any

import impresso_pipelines.ldatopics.language_inferencer as language_inferencer
# Remove direct imports to avoid circular dependencies
# import impresso_pipelines.mallet.s3_to_local_stamps
# import impresso_pipelines.mallet.mallet2topic_assignment_jsonl as m2taj
from smart_open import open


from impresso_pipelines.ldatopics.language_inferencer import LanguageInferencer

from impresso_pipelines.ldatopics.input_reader import (
    InputReader,
    JsonlInputReader,
    CsvInputReader,
    ImpressoLinguisticProcessingJsonlInputReader,
)


log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)  # Suppress warnings and unnecessary logs
load_dotenv()


def save_text_as_csv(text: str) -> str:
    """
    Save the given text as a temporary CSV file with an arbitrary ID and return the file
    name.

    Args:
        text (str): The text to be saved in the CSV file.

    Returns:
        str: The name of the temporary CSV file.
    """
    # Create a temporary file with .csv suffix
    temp_csv_file = tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".csv", newline="", encoding="utf-8"
    )

    # Write the text to the CSV file with an arbitrary ID
    csv_writer = csv.writer(temp_csv_file, delimiter="\t")
    csv_writer.writerow(["ID", "DUMMYCLASS", "TEXT"])  # Header
    csv_writer.writerow(["USERINPUT-2024-10-24-a-i0042", "dummy_class", text])

    # Close the file to ensure all data is written
    temp_csv_file.close()

    return temp_csv_file.name


class MalletTopicInferencer:
    """
    MalletTopicInferencer class coordinates the process of reading input documents,
    identifying their language, and performing topic inference using Mallet models.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.languages = set(args.languages)
        self.language_inferencers: Optional[Dict[str, LanguageInferencer]] = None
        self.language_lemmatizations: Optional[Dict[str, Dict[str, str]]] = None
        self.language_ma2ta_converters: Optional[Dict[str, Generator]] = None
        self.language_configs: Optional[Dict[str, Dict[str, str]]] = None
        self.input_reader = None
        self.inference_results: List[Dict[str, str]] = []
        self.language_dict: Dict[str, str] = {}
        self.seen_languages: Set[str] = set()
        self.stats = collections.Counter()
        self.initialized = False  # To check if the inferencer is initialized
        self.jvm_started = False  # To track if JVM is started by this instance
        self.output_path_base = args.output_path_base
        self.keep_tmp_files = args.keep_tmp_files
        self.include_lid_path = args.include_lid_path  # New argument
        self.inferencer_random_seed: int = args.inferencer_random_seed

        # Start the JVM and initialize inferencers
        self.start_jvm()
        self.initialize()

        # Initialize S3 client if input or language file is in S3
        from impresso_pipelines.ldatopics.s3_to_local_stamps import get_s3_client, s3_file_exists  # Lazy import
        self.S3_CLIENT = (
            get_s3_client()
            if self.args.input.startswith("s3://")
            or self.args.s3_output_path
            else None
        )

        # Check if the output file already exists in S3 and avoid lengthy processing
        if self.args.quit_if_s3_output_exists and (s3out := self.args.s3_output_path):
            if s3_file_exists(self.S3_CLIENT, s3out):
                log.warning(
                    "%s exists. Exiting without processing %s", s3out, self.args.input
                )
                exit(3)
            else:
                log.info("%s does not exist. Proceeding with processing.", s3out)
        self.git_version = (
            self.args.git_version
            if self.args.git_version
            else os.environ.get("GIT_VERSION", "unknown")
        )

        self.model_versions: Dict[str, str] = {}

        if args.keep_timestamp_only:
            self.keep_timestamp_only = True
        else:
            self.keep_timestamp_only = False

    def __del__(self):
        # Shut down the JVM if it was started by this instance
        if self.jvm_started and jpype.isJVMStarted():
            jpype.shutdownJVM()
            logging.info("JVM shut down.")

    # Optionally, implement context manager methods for better resource handling
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Shut down the JVM if it was started by this instance
        if self.jvm_started and jpype.isJVMStarted():
            jpype.shutdownJVM()
            logging.info("JVM shut down.")

        # Handle exceptions if needed
        if exc_type:
            logging.error(f"Exception: {exc_value}")
            log.error("Traceback: %s", traceback.format_exc())
            return False  # Re-raise exception
        return True  # Suppress exception if any

    def initialize(self):
        """Initialize the inferencers after JVM startup."""

        if not self.initialized:
            self.language_configs = self.init_language_configs(self.args)
            self.language_inferencers = self.init_language_inferencers(self.args)
            self.language_lemmatizations = self.init_language_lemmatizations(self.args)

            if self.args.output_format == "jsonl":
                self.language_ma2ta_converters = self.init_ma2ta_converters(self.args)
            if self.args.language_file:
                self.language_dict = self.read_language_file(self.args.language_file)
            if self.args.input:
                self.input_reader = self.build_input_reader(self.args)
            self.initialized = True  # Mark as initialized

    def start_jvm(self) -> None:
        """Start the Java Virtual Machine if not already started."""

        if not jpype.isJVMStarted():
            current_dir = os.getcwd()
            source_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct classpath relative to the current directory
            classpath = [
                os.path.join(current_dir, "mallet/lib/mallet-deps.jar"),
                os.path.join(current_dir, "mallet/lib/mallet.jar"),
            ]

            # Check if the files exist in the current directory
            if not all(os.path.exists(path) for path in classpath):
                # If not, construct classpath relative to the source directory
                classpath = [
                    os.path.join(source_dir, "mallet/lib/mallet-deps.jar"),
                    os.path.join(source_dir, "mallet/lib/mallet.jar"),
                ]

            jpype.startJVM(classpath=classpath)
        else:
            pass  # Suppress "JVM already running" warning

    def run(self) -> None:
        """Main execution method. Either processing an input file or waiting for
        interactive use."""

        if self.args.input:
            self.process_input_file()

        if self.args.s3_output_path and not self.args.s3_output_dry_run:
            from impresso_pipelines.mallet.s3_to_local_stamps import upload_file_to_s3  # Lazy import
            upload_file_to_s3(
                self.S3_CLIENT, self.args.output, self.args.s3_output_path
            )
        for key, value in sorted(self.stats.items()):
            log.info(f"STATS: {key}: {value}")

    def read_language_file(self, language_file: str) -> Dict[str, str]:
        """Read the language file (JSONL) and return a dictionary of document_id ->
        language."""

        language_dict = {}
        with open(language_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                doc_id = data.get("doc_id")
                language = data.get("language")
                if doc_id and language:
                    language_dict[doc_id] = language
        return language_dict

    @staticmethod
    def load_lemmatization_file(
        lemmatization_file_path: str,
        bidi: bool = False,
        lowercase: bool = False,
        ignore_pos: bool = True,
    ) -> Dict[str, str]:
        """
        Load lemmatization data from the file.
        :param lemmatization_file_path: Path to the lemmatization file.
        :return: A dictionary mapping tokens to their corresponding lemmas.
        """

        token2lemma = {}
        n = 0
        logging.info(
            "Reading lemmatization entries from %s with setting: lowercase=%s"
            " ignore_pos=%s ",
            lemmatization_file_path,
            lowercase,
            ignore_pos,
        )
        with open(lemmatization_file_path, "r", "utf-8") as file:
            for line in file:
                token, _, lemma = line.strip().split("\t")
                if lowercase:
                    token2lemma[token.lower()] = lemma.lower()
                else:
                    token2lemma[token] = lemma
                n += 1

        logging.info(
            "Read %d lemmatization entries from %s", n, lemmatization_file_path
        )
        return token2lemma

    def init_language_configs(
        self, args: argparse.Namespace
    ) -> Dict[str, Dict[str, str]]:
        """Build a mapping of languages to their respective Mallet configurations."""

        language_configs = {}
        for language in args.languages:
            config_key = f"{language}_config"
            if getattr(args, config_key, None):
                config_file = getattr(args, config_key)
                language_configs[language] = self.load_config_file(config_file)
                log.info(
                    "Loaded configuration for language: %s : %s : %s",
                    language,
                    config_file,
                    language_configs[language],
                )
            else:
                log.info(
                    "Configuration file for language: %s not provided by"
                    " arguments. Skipping.",
                    language,
                )
        return language_configs

    def load_config_file(self, config_file_path: str) -> Dict[str, Any]:
        """
        Load JSON configuration data from the file.

        :param config_file_path: Path to the configuration file.
        :return: A dictionary containing the configuration data.
        """

        try:
            with open(config_file_path, "r", encoding="utf-8") as file:
                config_data = json.load(file)
                log.debug(
                    f"Loaded configuration data from {config_file_path}: {config_data}"
                )
                return config_data
        except json.JSONDecodeError as e:
            log.error(
                f"JSON decode error in configuration file {config_file_path}: {e}"
            )
        except Exception as e:
            log.error(f"Error reading configuration file {config_file_path}: {e}")
        log.info("Continuing with empty configuration")
        return {}

    def init_language_lemmatizations(
        self, args: argparse.Namespace
    ) -> Dict[str, Dict[str, str]]:
        """Build a mapping of languages to their respective lemmatization
        dictionaries."""

        language_lemmatizations: Dict[str, Dict[str, str]] = {}
        for language in args.languages:
            lemmatization_key = f"{language}_lemmatization"
            if getattr(args, lemmatization_key, None):
                lemmatization_file = getattr(args, lemmatization_key)
                language_lemmatizations[language] = self.load_lemmatization_file(
                    lemmatization_file
                )
            else:
                log.info(
                    f"Lemmatization file for language: {language} not provided by"
                    " arguments. Skipping."
                )
        return language_lemmatizations

    def init_ma2ta_converters(self, args: argparse.Namespace) -> Dict[str, Generator]:
        from impresso_pipelines.ldatopics.mallet2topic_assignment_jsonl import Mallet2TopicAssignment  # Lazy import

        """
        Build a mapping of languages to their respective Mallet2TopicAssignment
        converters.

        Args:
            args (argparse.Namespace): The arguments namespace containing the
            configuration for initializing the converters. It should include:
            - languages (List[str]): List of languages to initialize converters for.
            - <language>_model_id (str): Model ID for each language.
            - <language>_topic_count (int): Topic count for each language.
            - min_p (float): Minimum probability threshold.
            - lingproc_run_id (Optional[str]): Linguistic processing run ID.
            - git_version (Optional[str]): Git version.
            - impresso_model_id (Optional[str]): Impresso model ID.

        Returns:
            Dict[str, Generator]: A dictionary mapping each language to its respective
            Mallet2TopicAssignment converter generator.
        """

        ma2ta_converters = {}
        for language in args.languages:
            logging.info(
                "Initializing Mallet2TopicAssignment converter for %s", language
            )
            topic_model_id = getattr(args, f"{language}_model_id")
            if "{lang}" in topic_model_id:
                topic_model_id.format(lang=language)
            ma2ta_args = [
                "--output",
                "<generator>",
                "--topic_model",
                topic_model_id,
                "--topic_count",
                str(getattr(args, f"{language}_topic_count")),
                "--lg",
                language,
                "--min-p",
                str(args.min_p),
            ]
            if self.args.lingproc_run_id:
                ma2ta_args.extend(["--lingproc-run_id", self.args.lingproc_run_id])
            if self.args.git_version:
                ma2ta_args.extend(["--git-version", self.args.git_version])
            if self.args.impresso_model_id:
                ma2ta_args.extend(["--impresso-model-id", self.args.impresso_model_id])
            ma2ta_converters[language] = Mallet2TopicAssignment.main(
                ma2ta_args
            ).run()
        return ma2ta_converters

    def identify_language(self, document_id: str, text: str) -> str:
        """Identify the language of the text using the language file or a dummy
        method."""

        # Check if the document ID is in the language dictionary
        if document_id in self.language_dict:
            return self.language_dict[document_id]
        # Placeholder: Assume German ("de") for now if not found in the dictionary
        return "de"

    def init_language_inferencers(
        self, args: argparse.Namespace
    ) -> Dict[str, LanguageInferencer]:
        """Build a mapping of languages to their respective inferencers

        Includes the vectorizer pipe for each language as well.
        """

        language_inferencers: Dict[str, LanguageInferencer] = {}
        for language in args.languages:
            inferencer_key = f"{language}_inferencer"
            pipe_key = f"{language}_pipe"
            if getattr(args, inferencer_key, None) and getattr(args, pipe_key, None):
                language_inferencers[language] = LanguageInferencer(
                    language=language,
                    inferencer_file=getattr(args, inferencer_key),
                    pipe_file=getattr(args, pipe_key),
                    keep_tmp_files=args.keep_tmp_files,
                    random_seed=self.inferencer_random_seed,
                )
            else:
                log.info(
                    f"Inferencer or pipe file for language: {language} not provided by"
                    " arguments. Skipping."
                )
        return language_inferencers

    def build_input_reader(self, args: argparse.Namespace) -> InputReader:
        """Select the appropriate input reader based on the input format."""

        if args.input_format == "jsonl":
            return JsonlInputReader(args.input)
        elif args.input_format == "csv":
            return CsvInputReader(args.input)
        elif args.input_format == "impresso":
            return ImpressoLinguisticProcessingJsonlInputReader(
                args.input,
                self.language_lemmatizations,
                self.language_configs,
                ci_ids=self.args.ci_ids,
            )
        else:
            raise ValueError(f"Unsupported input format: {args.input_format}")

    def process_input_file(self) -> None:
        """
        Process the input .mallet file and perform topic inference.
        """
        logging.info("Processing input file: %s", self.args.input)

        # Validate the input .mallet file
        if not os.path.exists(self.args.input):
            raise FileNotFoundError(
                f"Input file not found: {self.args.input}. Ensure the file exists and is in the correct format."
            )

        # Check if the file is a valid Mallet InstanceList
        try:
            from jpype import JClass  # Import JClass to create Java objects
            java_file = JClass("java.io.File")(self.args.input)  # Convert to java.io.File
            from cc.mallet.types import InstanceList  # Import Mallet's InstanceList
            InstanceList.load(java_file)  # Use the java.io.File object
        except Exception as e:
            raise ValueError(
                f"Invalid Mallet file format: {self.args.input}. Ensure the file was generated using Csv2Vectors. Error: {e}"
            )

        # Run topic inference
        doctopics_file = self.run_topic_inference(self.args.input)


        if self.args.output_format == "csv":
            self.merge_inference_results({self.args.languages[0]: doctopics_file})
        elif self.args.output_format == "jsonl":
            self.merge_inference_results_jsonl({self.args.languages[0]: doctopics_file})

    def infer_texts(self, texts: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Infer topics for multiple texts provided as an iterable of dicts.

        Each dict should have keys: 'text', 'language', and 'id'.

        Returns a list of result dicts, each containing 'doc_id', 'language', and
        'topic_distribution'.
        """

        if not self.initialized:
            self.initialize()

        # Group texts by language
        texts_by_language = collections.defaultdict(list)
        for item in texts:
            text = item["text"]
            language = item["language"]
            doc_id = item.get("id", "doc1")
            if language not in self.languages:
                logging.warning(
                    f"Language '{language}' not supported. Skipping document {doc_id}."
                )
                continue
            lemmas = self.analyze_text(text, language)
            if not lemmas:
                logging.warning("No lemmas found in document %s: %s", doc_id, text)
            texts_by_language[language].append((doc_id, " ".join(lemmas)))

        results = []

        # For each language, process the texts
        for language, docs in texts_by_language.items():
            # Create a temporary CSV file with the input texts
            with tempfile.NamedTemporaryFile(
                delete=False,
                mode="w",
                suffix=f".{language}.csv",
                newline="",
                encoding="utf-8",
            ) as temp_csv_file:
                csv_writer = csv.writer(
                    temp_csv_file,
                    delimiter="\t",
                    escapechar=None,
                    quoting=csv.QUOTE_NONE,
                )

                for doc_id, text in docs:
                    csv_writer.writerow([doc_id, language, text])
                csv_file_path = temp_csv_file.name
                

            # Run topic inference
            inferencer = self.language_inferencers[language]
            doctopics_file = inferencer.run_csv2topics(csv_file_path)

            # Read the results from the doctopics file
            with open(doctopics_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    # The format is: <doc_index> <doc_id> <topic_0_weight> <topic_1_weight> ...
                    # doc_index = parts[0]
                    doc_id = parts[1]
                    topic_weights = parts[2:]
                    topic_distribution = [float(x) for x in topic_weights]

                    result = {
                        "doc_id": doc_id,
                        "language": language,
                        "topic_distribution": topic_distribution,
                    }
                    results.append(result)

            if logging.getLogger().getEffectiveLevel() != logging.DEBUG:
                logging.info("Deleting temporary file: %s", csv_file_path + ".mallet")
                os.remove(csv_file_path + ".mallet")
        

        return results

    def infer_text(
        self, text: str, language: str, doc_id: str = "doc1"
    ) -> Dict[str, Any]:
        """
        Infer topics for a single text input.

        Returns a result dict containing 'doc_id', 'language', and 'topic_distribution'.
        """

        items = [{"text": text, "language": language, "id": doc_id}]
        results = self.infer_texts(items)
        if results:
            return results[0]
        else:
            return {}

    def merge_inference_results_jsonl(self, doctopics_files_by_language):
        """
        Merges inference results from multiple JSONL files into a single output file.
        Args:
            doctopics_files_by_language (dict): A dictionary where keys are language
              codes and values are paths to the corresponding doctopics files.
        Returns:
            None

        This method processes the given doctopics files by language, converts them using
        the Mallet2TopicAssignment tool, and writes the merged results to the output
        file specified in self.args.output. It also updates the content items statistics
        and deletes the temporary files if the logging level is not DEBUG.
        """

        m2ta_converters = {}
        for lang, doctopics_file in doctopics_files_by_language.items():
            args = ["--output", "<generator>"]
            topic_model_id = self.args.__dict__[f"{lang}_model_id"]
            if "{lang}" in topic_model_id:
                topic_model_id.format(lang=lang)
            args += [
                "--git-version",
                self.args.git_version,
                "--topic_model",
                topic_model_id,
                "--topic_count",
                str(self.args.__dict__[f"{lang}_topic_count"]),
                "--lang",
                lang,
                "--min-p",
                str(self.args.min_p),
                doctopics_file,  # input comes last!
            ]
            if self.args.lingproc_run_id:
                args.extend(["--lingproc_run_id", self.args.lingproc_run_id])
            if self.args.git_version:
                args.extend(["--git-version", self.args.git_version])
            if self.args.impresso_model_id:
                args.extend(["--impresso-model-id", self.args.impresso_model_id])
            from impresso_pipelines.ldatopics.mallet2topic_assignment_jsonl import Mallet2TopicAssignment  # Lazy import
            m2ta_converters[lang] = Mallet2TopicAssignment.main(args).run()

        with open(self.args.output, "w", encoding="utf-8") as out_f:
            for lang, m2ta_converter in m2ta_converters.items():
                for row in m2ta_converter:
                    self.stats["content_items"] += 1
                    if self.include_lid_path:
                        row["lid_path"] = self.args.language_file
                    print(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                        file=out_f,
                    )
        if not self.keep_tmp_files:
            for doctopics_file in doctopics_files_by_language.values():
                logging.debug("Deleting temporary file: %s", doctopics_file)
                os.remove(doctopics_file)

    def merge_inference_results(
        self, doctopics_files_by_language: Dict[str, str]
    ) -> None:
        """
        Merges topic inference results from multiple languages into a single CSV file.
        Args:
            doctopics_files_by_language (Dict[str, str]): A dictionary where keys are
              language codes and values are file paths to the topic distribution files
              for each language.
        Returns:
            None

        The method reads topic distribution files for each language, appends the
        language code to the document ID, and writes the merged results into a single
        output file specified by `self.args.output`.
        """
        logging.info(
            "Saving CSV inference results into file %s from multiple languages: %s",
            self.args.output,
            doctopics_files_by_language,
        )
        with open(self.args.output, "w", encoding="utf-8") as out_f:
            logging.info("Writing merged inference results to: %s", self.args.output)
            for language, doctopics_file in doctopics_files_by_language.items():
                with open(doctopics_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        doc_id, topic_dist = line.strip().split("\t", 1)
                        print(
                            doc_id + "__" + language,
                            topic_dist,
                            sep="\t",
                            end="\n",
                            file=out_f,
                        )
        if not self.keep_tmp_files:
            for doctopics_file in doctopics_files_by_language.values():
                logging.info("Deleting temporary file: %s", doctopics_file)
                os.remove(doctopics_file)

    def write_language_specific_csv_files(self) -> Dict[str, str]:
        """Read documents and write to language-specific temporary files"""
        tsv_files_by_language = {}

        for document_id, lang, text in self.input_reader.read_documents():
            if lang in self.languages:
                language_code = lang
            else:
                language_code = self.identify_language(document_id, text)
            self.stats["LANGUAGE: " + language_code] += 1
            if language_code not in self.languages:
                continue

            if language_code not in tsv_files_by_language:
                if self.output_path_base:
                    temp_file_path = f"{self.output_path_base}.{language_code}.tsv"
                    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                    tsv_files_by_language[language_code] = open(
                        temp_file_path, "w", encoding="utf-8"
                    )
                else:
                    tsv_files_by_language[language_code] = tempfile.NamedTemporaryFile(
                        delete=False,
                        mode="w",
                        suffix=f".{language_code}.tsv",
                        encoding="utf-8",
                    )
                logging.debug(
                    "Writing documents for language: %s in temp file: %s",
                    language_code,
                    tsv_files_by_language[language_code].name,
                )

            print(
                document_id,
                language_code,
                text,
                sep="\t",
                end="\n",
                file=tsv_files_by_language[language_code],
            )

        # Close all temporary files
        for temp_file in tsv_files_by_language.values():
            temp_file.close()

        # noinspection PyShadowingNames
        result = {
            lang: temp_file.name for lang, temp_file in tsv_files_by_language.items()
        }
        return result

    def run_topic_inference(self, mallet_file: str) -> str:
        """
        Run topic inference on the provided .mallet file.
        """
        inferencer = self.language_inferencers.get(self.args.languages[0])
        if not inferencer:
            log.error(f"No inferencer found for language: {self.args.languages[0]}")
            exit(1)

        doctopics_file = inferencer.run_csv2topics(
            mallet_file, delete_mallet_file_after=not self.keep_tmp_files
        )
        logging.debug("Resulting doctopic file: %s", doctopics_file)
        return doctopics_file

    def write_results_to_output(self) -> None:
        """Write the final merged inference results to the output file."""
        with open(self.args.output, "w", encoding="utf-8") as out_file:
            for result in self.inference_results:
                out_file.write(json.dumps(result) + "\n")
        log.info(f"All inferences merged and written to {self.args.output}")


if __name__ == "__main__":
    languages = ["de", "fr", "lb"]  # You can add more languages as needed
    parser = argparse.ArgumentParser(description="Mallet Topic Inference in Python")

    parser.add_argument("--logfile", help="Path to log file", default=None)
    parser.add_argument(
        "--level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level $(default)s",
    )
    parser.add_argument(
        "--input",
        help=(
            "Path to input file. If omitted, no file processing is done but you can use"
            " infert_topics() method for more interactive use."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["impresso", "jsonl", "csv"],
        default="jsonl",
        help=(
            "Format of the input file (default: %(default)s). 'impresso' is a JSONL"
            " file containing linguistic processing data. 'jsonl' is a JSONL file with"
            " doc_id, text and language. 'csv' is a mallet three-column CSV file"
            " (DOCID, CLASS, LEMMAS)."
        ),
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=languages,
        help="List of languages to support (%(default)s)",
    )
    parser.add_argument(
        "--ci_ids", nargs="+", help="List of ci_ids to process", required=False
    )
    parser.add_argument(
        "--output",
        help="Path to final output file. (%(default)s)",
        default="impresso_pipelines/mallet/tmp_output.jsonl",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "csv"],
        help=(
            "Format of the output file: csv: raw Mallet output with docids patched into"
            " numericID-LANG, jsonl: impresso JSONL format"
        ),
    )
    parser.add_argument(
        "--lemmatization_mode",
        choices=["v2.0-legacy"],
        default="v2.0-legacy",
        help=(
            "Lemmatization mode to use (%(default)s). v2.0-legacy: Use the"
            " lemmatization strategy from v2.0: case-sensitive lookup of token then"
            " lookup of spacy lemma in the additional lemmatization UPOS matches"
            " (except for lb)"
        ),
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.02,
        help=(
            "Minimum probability threshold to include the topic in the output (Default:"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--quit-if-s3-output-exists",
        action="store_true",
        help=(
            "Exit with code 3 if the output file already exists in the specified S3"
            " bucket"
        ),
    )
    parser.add_argument(
        "--s3-output-dry-run",
        action="store_true",
        help=(
            "Never upload anything to s3 even if s3-output-path is provided. Useful for"
            " local testing."
        ),
    )
    parser.add_argument(
        "--s3-output-path",
        help=(
            "S3 path to upload the output file after processing or check if it already"
            " exists"
        ),
    )
    parser.add_argument(
        "--git-version",
        help="Specify the git version to use",
    )
    parser.add_argument(
        "--lingproc-run_id",
        help=(
            "Add the impresso linguistic processing run id as property"
            " 'lingproc_run_id' to the output for data traceability."
        ),
    )
    parser.add_argument(
        "--keep-timestamp-only",
        action="store_true",
        help="Keep only the timestamp in the output",
    )
    parser.add_argument(
        "--log-file",
        help="Path to the log file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not log to console, only to the log file (if specified).",
    )
    for lang in languages:
        parser.add_argument(
            f"--{lang}_config",
            help=(
                "Configuration file of topic modeling for language residing in the"
                " model_dir. If provided,--model_dir (will be derived as directory of"
                f" configuration file) --{lang}_topic_count  options are oveset from"
                " configuration file . "
            ),
        )
    parser.add_argument(
        "--language-file",
        help="Path to JSONL containing document_id to language mappings",
        required=False,
    )
    parser.add_argument(
        "--keep-tmp-files",
        "--keep_tmp_files",
        action="store_true",
        help="Keep temporary files (Default: %(default)s)",
    )
    parser.add_argument("--model_dir", help="Path to model directory")
    parser.add_argument(
        "--output_path_base",
        help=(
            "Base path for temporary files. If not specified, uses system temporary"
            " files. And default to removing intermediate files."
        ),
        required=False,
    )
    parser.add_argument(
        "--include-lid-path",
        action="store_true",
        help="Include the LID file path in the output JSON for traceability",
    )
    parser.add_argument(
        "--inferencer-random-seed",
        type=int,
        default=42,
        help="Set the random seed for the inferencer (Default: %(default)s)",
    )
    parser.add_argument(
        "--impresso-model-id",
        help="The s3 model id stored as 'model_id' in the output.",
    )
    # Dynamically generate arguments for each language's inferencer and pipe files
    for lang in languages:
        parser.add_argument(
            f"--{lang}_inferencer",
            help=f"Path to {lang} inferencer file",
        )
        parser.add_argument(f"--{lang}_pipe", help=f"Path to {lang} pipe file")
        parser.add_argument(
            f"--{lang}_lemmatization", help=f"Path to {lang} lemmatization file"
        )
    # Dynamically generate arguments for each language's inferencer and pipe files
    for lang in languages:
        parser.add_argument(
            f"--{lang}_model_id",
            help="Model ID can take a {lang} format placeholder (%(default)s)",
        )
    for lang in languages:
        parser.add_argument(
            f"--{lang}_topic_count",
            help="Number of topics of model (%(default)s). ",
        )

    args = parser.parse_args()

    # Check if the output file already exists on S3 and avoid any processing
    if args.quit_if_s3_output_exists and (s3out := args.s3_output_path):
        from impresso_pipelines.mallet.s3_to_local_stamps import s3_file_exists, get_s3_client  # Lazy import
        if s3_file_exists(get_s3_client(), s3out):
            logging.warning(
                "S3 file exists: %s Exiting without processing %s", s3out, args.input
            )
            exit(3)
    # Configure logging
    if args.quiet:
        log_handlers = []
    else:
        log_handlers = [logging.StreamHandler()]
    if args.log_file:

        class SmartFileHandler(logging.FileHandler):
            def _open(self):
                return open(self.baseFilename, self.mode, encoding="utf-8")

        log_handlers.append(SmartFileHandler(args.log_file, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s",
        handlers=log_handlers,
        force=True,
    )
    log.info("Script called with args: %s", args)

    logging.info("Setting up MalletTopicInferencer")
    # Automatically construct file paths if not explicitly specified
    for lang in args.languages:
        if config_path := getattr(args, f"{lang}_config"):
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            config = json.load(open(config_path, "r", encoding="utf-8"))

            model_dir = os.path.dirname(config_path)
            model_id = config.get("model_id", f"tm-{lang}-all-v2.0")
            setattr(args, f"{lang}_model_id", model_id)
            setattr(args, f"{lang}_topic_count", config["topic_count"])
            logging.info(
                "Config's %s topic count for language %s used: %s",
                config_path,
                lang,
                config["topic_count"],
            )

        else:
            model_id = getattr(args, f"{lang}_model_id")
            if not args.model_dir:
                logging.error(
                    "Model directory option --model-dir  not provided. Please provide a"
                    " model directory or specify the config json path."
                )
                exit(1)
            model_dir = args.model_dir
            config_path = os.path.join(model_dir, f"{model_id}.config.json")
            if os.path.exists(config_path):
                logging.info(
                    "Automatically setting config json path to %s", config_path
                )
                setattr(args, f"{lang}_config", config_path)

        pipe_path = os.path.join(model_dir, f"{model_id}.pipe")
        inferencer_path = os.path.join(model_dir, f"{model_id}.inferencer")
        lemmatization_path = os.path.join(
            model_dir, f"{model_id}.vocab.lemmatization.tsv.gz"
        )

        if not getattr(args, f"{lang}_pipe") and os.path.exists(pipe_path):
            logging.info("Automatically setting pipe path to %s", pipe_path)
            setattr(args, f"{lang}_pipe", pipe_path)
        if not getattr(args, f"{lang}_inferencer") and os.path.exists(inferencer_path):
            logging.info("Automatically setting inferencer path to %s", inferencer_path)
            setattr(args, f"{lang}_inferencer", inferencer_path)
        if not getattr(args, f"{lang}_lemmatization") and os.path.exists(
            lemmatization_path
        ):
            logging.info(
                "Automatically setting lemmatization path to %s", lemmatization_path
            )
            setattr(args, f"{lang}_lemmatization", lemmatization_path)
    if args.output_path_base:
        args.keep_tmp_files = True
        if args.output == "impresso_pipelines/mallet/tmp_output.jsonl":  # the default should be overwritten
            if args.output_format == "jsonl":
                args.output = args.output_path_base + ".jsonl"
            elif args.output_format == "csv":
                args.output = args.output_path_base + ".csv"
            else:
                logging.error("Unsupported output format: %s", args.output_format)
                exit(1)
    if not args.output_format:
        if "jsonl" in args.output:
            args.output_format = "jsonl"
        else:
            args.output_format = "csv"
        logging.warning("Unspecified output format set to %s", args.output_format)
    for lang in args.languages:
        if not getattr(args, f"{lang}_inferencer") or not getattr(args, f"{lang}_pipe"):
            logging.warning(
                "Inferencer or pipe file not provided for language: %s. Ignoring"
                " content items for this language.",
                lang,
            )
            args.languages.remove(lang)
    logging.info(
        "Performing monolingual topic inference for the following languages: %s",
        args.languages,
    )
    logging.info("MalletTopicInferencer setup finished.")
    logging.info("MalletTopicInferencer Class Arguments: %s", args)
    app = MalletTopicInferencer(args)
    app.run()
