#!/usr/bin/env python3

"""
This script processes Mallet topic modeling output and generates impresso JSON topic assignment format.

It can be used in two ways:

 - As a standalone script that reads Mallet doctopics files and writes JSON topic assignments.
 - As a module that can be imported and used to generate JSON topic assignments from Mallet doctopics files.

The script reads Mallet doctopics files and generates JSON topic assignments in the following format:

Typical input format:
DOC_SERIAL_NUMBER DOC_NAME TOPIC0_PROPORTION  TOPIC2_PROPORTION   ...

Typical output of the script (pretty printed):

{
    "topic_count": 100,
    "lg": "de",
    "topics": [
        {"t": "tm-de-all-v2.0_tp02_de", "p": 0.027},
        {"t": "tm-de-all-v2.0_tp11_de", "p": 0.119},
        {"t": "tm-de-all-v2.0_tp26_de", "p": 0.045}
    ],
    "min_p": 0.02,
    "ts": "2024.08.29",
    "id": "actionfem-1927-12-15-a-i0001",
    "model_id": "tm-de-all-v2.0"
}
"""

import argparse
import json
import math
import re
import logging
import traceback
import collections
import jsonschema
from jsonschema import Draft7Validator
from typing import Generator, List, Dict, Any, Optional
from smart_open import open
import impresso_pipelines.ldatopics.s3_to_local_stamps


SCHEMA_BASE_URI = "https://impresso.github.io/impresso-schemas/json/topic_model/"

IMPRESSO_SCHEMA = "topic_assignment.v2.schema.json"


# Regular expression to extract the CI_ID from the document path with unix path separators
# Pattern breakdown:
# ^(.+?/)? - Matches the beginning of the string and optionally captures any leading path segments ending with a slash
# ([^/]+?-\d{4}-\d{2}-\d{2}-\w-i\d{4}) - Captures the CI_ID which includes a non-slash sequence followed by a date and an identifier
# [^/]*$ - Matches any trailing characters that are not slashes until the end of the string

CI_ID_REGEX = re.compile(r"^(.+?/)?([^/]+?-\d{4}-\d{2}-\d{2}-\w-i\d{4})[^/]*$")


def initialize_validator(
    schema_base_uri=SCHEMA_BASE_URI, schema=IMPRESSO_SCHEMA
) -> Draft7Validator:
    schema_path = schema_base_uri + schema
    with open(schema_path, "r") as f:
        schema = json.load(f)
    # Directly create the validator without a registry or a resolver
    validator = Draft7Validator(schema)
    return validator


class Mallet2TopicAssignment:
    """
    A class to convert Mallet topic assignments to JSON format.

    Attributes:
        min_p (float): The topic assignment threshold.
        lang (str): The language code.
        topic_model (str): The topic model identifier.
        numeric_topic_ids (bool): Whether to use numeric topic IDs.
        input_format_type (str): The format type of the input files.
        topic_count (int): The number of topics.
        output (str): The output file path.
        input_files (Optional[List[str]]): The list of input files.
        precision (int): The precision for rounding probabilities.
        padding_length (int): The padding length for topic IDs.
        topic_id_format (str): The format string for topic IDs.
        last_timestamp (str): The timestamp of the last modification.

    Methods:
        __init__(self, min_p: float, lang: str, topic_model: str, numeric_topic_ids: bool, input_format_type: str, topic_count: int, output: str, input_files: Optional[List[str]] = None) -> None:
            Initializes the Mallet2TopicAssignment instance with the provided parameters.

        validate_options(self) -> None:
            Validates the options provided during initialization.

        read_tsv_files(self, filenames: List[str]) -> Generator[List[str], None, None]:

        read_tsv_file(self, filename: str) -> Generator[List[str], None, None]:
            Reads a single TSV file and yields its contents line by line.

        convert_matrix_row(self, row: List[str]) -> Dict[str, Any]:
            Converts a row from a matrix format TSV file to a dictionary.

        convert_sparse_row(self, row: List[str]) -> Dict[str, Any]:
            Converts a row from a sparse format TSV file to a dictionary.

        convert_doctopics_files(self, filenames: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]:
            Yields JSON topic assignments for Mallet doctopics files.

        run(self, input_files: Optional[List[str]] = None, mode: str = "file") -> Optional[Generator[Dict[str, Any], None, None]]:
            Processes the input files based on the initialization and returns a generator if output is set to '<generator>', otherwise writes to a file.

        setup_logging(logging_level: str = "INFO", logfile: Optional[str] = None, format: str = "%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s") -> None:
            Sets up logging configuration based on command line options.

        main(args: Optional[List[str]] = None) -> "Mallet2TopicAssignment":
            Static method serving as the CLI entry point of the script and returns a configured instance of the
            application.
    """

    def __init__(
        self,
        min_p: float,
        lang: str,
        topic_model: str,
        numeric_topic_ids: bool,
        input_format_type: str,
        topic_count: int,
        output: str,
        input_files: Optional[List[str]] = None,
        git_version: Optional[str] = None,
        lingproc_run_id: Optional[str] = None,
        impresso_model_id: Optional[str] = None,
        no_jsonschema_validation: bool = False,
    ) -> None:
        self.min_p = min_p
        self.lang = lang
        self.topic_model = topic_model
        self.numeric_topic_ids = numeric_topic_ids
        self.input_format_type = input_format_type.lower()
        self.topic_count = topic_count
        self.output = output
        self.input_files = input_files if input_files else None
        self.git_version = git_version
        self.lingproc_run_id = lingproc_run_id
        self.impresso_model_id = impresso_model_id
        self.schema_validator = (
            None if no_jsonschema_validation else initialize_validator()
        )
        self.validate_options()

        self.precision = math.ceil(abs(math.log10(self.min_p))) + 1
        self.padding_length = math.ceil(math.log10(self.topic_count))
        self.topic_id_format = (
            f"{self.topic_model}_tp{{t:0{self.padding_length}d}}_{self.lang}"
        )
        self.last_timestamp = impresso_pipelines.ldatopics.s3_to_local_stamps.get_timestamp()

    def validate_options(self) -> None:
        if self.min_p <= 0 or self.min_p >= 1:
            raise ValueError("min_p must be between 0 and 1.")
        if self.input_format_type == "sparse" and not self.topic_count:
            raise ValueError(
                "The --topic_count option is required when using the 'sparse' format."
            )

    def read_tsv_files(self, filenames: List[str]) -> Generator[List[str], None, None]:
        """
        Reads multiple TSV files and yields their contents line by line.

        Args:
            filenames (List[str]): A list of file paths to the TSV files.

        Yields:
            Generator[List[str], None, None]: A generator that yields lists of strings,
            each representing a line from the TSV files.
        """

        for filename in filenames:
            yield from self.read_tsv_file(filename)

    def read_tsv_file(self, filename: str) -> Generator[List[str], None, None]:
        line_count = 0
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line_count += 1
                if not line.startswith("#"):
                    yield line.strip().split("\t")
                if line_count % 1000 == 0:
                    logging.info("Processed lines: %s", line_count)

    def convert_matrix_row(self, row: List[str]) -> Dict[str, Any]:
        ci_id = re.sub(CI_ID_REGEX, r"\2", row[1])
        topics = row[2:]
        topic_count = len(topics)
        if self.numeric_topic_ids:
            topics = [
                {"t": t, "p": round(fp, self.precision)}
                for t, p in enumerate(topics)
                if (fp := float(p)) >= self.min_p
            ]
        else:
            topics = [
                {
                    "t": self.topic_id_format.format(t=t),
                    "p": round(p, self.precision),
                }
                for t, p in sorted(
                    enumerate(float(p) for p in topics),
                    key=lambda x: x[1],
                    reverse=True,
                )
                if p >= self.min_p
            ]
        result = {
            "ci_id": ci_id,
            "ts": self.last_timestamp,
            "lg": self.lang,
            "topic_count": topic_count,
            "topics": topics,
            "min_p": self.min_p,
            "topic_model_id": self.topic_model,
        }
        if self.git_version:
            result["topics_git"] = self.git_version
        if self.lingproc_run_id:
            result["lingproc_run_id"] = self.lingproc_run_id
        if self.impresso_model_id:
            result["model_id"] = self.impresso_model_id
        return result

    def convert_sparse_row(self, row: List[str]) -> Dict[str, Any]:
        ci_id = re.sub(CI_ID_REGEX, r"\2", row[1])
        topic_pairs = row[2:]
        topics = []
        for i in range(0, len(topic_pairs), 2):
            t = int(topic_pairs[i])
            p = float(topic_pairs[i + 1])
            if p >= self.min_p:
                if self.numeric_topic_ids:
                    topics.append(
                        {
                            "t": t,
                            "p": round(p, math.ceil(abs(math.log10(self.min_p))) + 1),
                        }
                    )
                else:
                    topics.append(
                        {
                            "t": self.topic_id_format.format(t=t),
                            "p": round(p, math.ceil(abs(math.log10(self.min_p))) + 1),
                        }
                    )

        result = {
            "ci_id": ci_id,
            "model_id": self.topic_model,
            "ts": self.last_timestamp,
            "lg": self.lang,
            "topic_count": self.topic_count,
            "topics": topics,
            "min_p": self.min_p,
            "topic_model_id": self.topic_model,
        }
        if self.git_version:
            result["git_version"] = self.git_version
        if self.lingproc_run_id:
            result["lingproc_run_id"] = self.lingproc_run_id
        if self.impresso_model_id:
            result["model_id"] = self.impresso_model_id
        return result

    def convert_doctopics_files(
        self, filenames: Optional[List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield JSON topic assignments for Mallet doctopics files.

        Args:
            filenames (List[str]): List of paths to the input files.

        Yields:
            Dict[str, Any]: Parsed topic assignment from each line in the input files.
        """

        if filenames is None:
            filenames = self.input_files

        ci_id_stats = collections.Counter()
        if self.input_format_type == "sparse":
            convert_row = self.convert_sparse_row
        elif self.input_format_type == "matrix":
            convert_row = self.convert_matrix_row
        else:
            raise ValueError(f"Invalid format type: {self.input_format_type}")

        for row in self.read_tsv_files(filenames):
            ci_id = re.sub(CI_ID_REGEX, r"\2", row[1])
            if ci_id in ci_id_stats:
                ci_id_stats["DUPLICATE_COUNT"] += 1
                continue
            ci_id_stats[ci_id] = 1

            doc_json = convert_row(row)
            if self.schema_validator:
                if not self.validate_document(doc_json):
                    continue
            yield doc_json

        logging.info("DUPLICATE-COUNT: %d", ci_id_stats["DUPLICATE_COUNT"])

    def validate_document(self, document: Dict[str, Any]) -> bool:
        """
        Validates a document against the schema.

        Args:
            document (Dict[str, Any]): The document to validate.

        Returns:
            bool: True if the document is valid, False otherwise.
        """
        try:
            self.schema_validator.validate(document)
            logging.debug("Document %s is valid", document["ci_id"])
            return True
        except jsonschema.ValidationError as e:
            logging.error("Validation error: %s", e)
            return False
        except jsonschema.SchemaError as e:
            logging.error("Schema error: %s", e)
            return False

    def run(
        self, input_files: Optional[List[str]] = None, mode: str = "file"
    ) -> Optional[Generator[Dict[str, Any], None, None]]:
        """
        Generic method to process the input files based on the initialization.
        Returns a generator if output is set to '<generator>', otherwise writes to a file.

        Args:
            input_files (List[str]): List of input files to process.
            mode (str): The mode of operation. Can be 'file' or 'generator'. Default is 'file'.

        Returns:
            Optional[Generator[Dict[str, Any], None, None]]: A generator for topic assignments
            if output is set to '<generator>', otherwise None.
        """

        if input_files is None:
            input_files = self.input_files
        if mode == "generator" or self.output == "<generator>":
            return self.convert_doctopics_files(input_files)
        elif mode == "file":
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")
        try:
            with open(self.output, "w", encoding="utf-8") as out_file:
                for topic_assignment in self.convert_doctopics_files(input_files):
                    out_file.write(
                        json.dumps(
                            topic_assignment, ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error("Traceback: %s", traceback.format_exc())
            exit(1)

    @staticmethod
    def setup_logging(
        logging_level: str = "INFO",
        logfile: Optional[str] = None,
        log_format: str = "%(asctime)-15s %(filename)s:%(lineno)d %(levelname)s: %(message)s",
    ) -> None:
        """
        Set up logging configuration based on command line options.
        """
        logging_level = getattr(logging, logging_level.upper(), logging.INFO)
        logging.basicConfig(
            level=logging_level,
            filename=logfile if logfile else None,
            force=True,
            format=log_format,
        )

    @staticmethod
    def main(
        args: Optional[List[str]] = None,
        set_logging: bool = False,
    ) -> "Mallet2TopicAssignment":
        """
        Static method serving as the cli entry point of the script and returns a configured instance of the application.
        Use the run() method to start processing the input files.
        Or use the convert_doctopics_files() method to process the input files and get a generator.

        Args:
            args (Optional[List[str]]): Command line arguments.

        Returns:
            Mallet2TopicAssignment: An instance of the application
        """
        parser = argparse.ArgumentParser(
            usage="%(prog)s [OPTIONS] INPUT [INPUT ...]",
            description=(
                "Return impresso JSON topic assignments from mallet textual doctopics"
                " output."
            ),
            epilog="See https://github.com/impresso for more information.",
        )

        parser.add_argument("--version", action="version", version="2024.11.01")
        parser.add_argument(
            "-l", "--logfile", help="Write log information to FILE", metavar="FILE"
        )
        parser.add_argument(
            "-L",
            "--lang",
            "--lg",
            "--language",
            default="und",
            help="ISO 639 language code two-letter or 'und' for undefined",
        )
        parser.add_argument(
            "-M",
            "--topic_model",
            default="tm000",
            help=(
                "Topic model identifier, which is used as a prefix for topic IDs:"
                " TOPIC_MODEL_tpTOPIC_ID_LANG It appears as 'mallet_model_id' in the"
                " output."
            ),
        )
        parser.add_argument(
            "-N",
            "--numeric_topic_ids",
            "--numeric-topic-ids",
            action="store_true",
            help="Use numeric topic IDs in the topic assignment",
        )
        parser.add_argument(
            "-T",
            "--min-p",
            "--topic-assignment-threshold",
            type=float,
            dest="min_p",
            default=0.02,
            help="Minimum probability for inclusion in the output (%(default)s)",
        )
        parser.add_argument(
            "-F",
            "--input_format_type",
            "--input-format-type",
            choices=["matrix", "sparse"],
            default="matrix",
            help=(
                "Format of the input file: 'matrix' or 'sparse' (Default"
                " `%(default)s`). This depends on the mallet output format option used."
                " In mallet --doc-topics-threshold N can be used to filter out topics"
                " with a probability less than N."
            ),
        )
        parser.add_argument(
            "-C",
            "--topic_count",
            "--topic-count",
            type=int,
            help="Needed for formatting. Either set this option or the config file.",
        )
        parser.add_argument(
            "-o",
            "--output",
            help=(
                "Path to the output file (%(default)s). If set to '<generator>' it will"
                " return a generator that can be used to enumerate all results in a"
                " flexible way. "
            ),
            default="/dev/stdout",
        )
        parser.add_argument(
            "--impresso-model-id",
            help="The s3 model id stored as 'model_id' in the output.",
        )
        parser.add_argument(
            "--lingproc-run_id",
            "--lingproc_run_id",
            help=(
                "Add the impresso linguistic processing run id as property"
                " 'lingproc_run_id' to the output for data traceability."
            ),
        )
        parser.add_argument(
            "--git-version",
            "--git_version",
            help=(
                "Add the git version as property 'git_version' to the output for data"
                " traceability."
            ),
        )
        parser.add_argument(
            "--no-jsonschema-validation",
            action="store_true",
            help=(
                "Do not validate the output agains the schema"
                f" {SCHEMA_BASE_URI + IMPRESSO_SCHEMA}."
            ),
        )
        parser.add_argument(
            "--level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level. Default: %(default)s",
        )
        parser.add_argument(
            "INPUT_FILES", nargs="*", help="Preset input files to process."
        )

        options = parser.parse_args(args=args)

        # Configure logging: Only do it if the script is run as a standalone script
        # The main script is responsible for setting up logging.
        if set_logging:
            Mallet2TopicAssignment.setup_logging(
                logging_level=options.level, logfile=options.logfile
            )
        logging.info("Mallet2TopicAssignment Options: %s", options)

        # Create the application instance
        app = Mallet2TopicAssignment(
            input_files=options.INPUT_FILES,
            min_p=options.min_p,
            lang=options.lang,
            topic_model=options.topic_model,
            numeric_topic_ids=options.numeric_topic_ids,
            input_format_type=options.input_format_type,
            topic_count=options.topic_count,
            output=options.output,
            git_version=options.git_version,
            lingproc_run_id=options.lingproc_run_id,
            impresso_model_id=options.impresso_model_id,
        )
        return app


if __name__ == "__main__":
    Mallet2TopicAssignment.main(set_logging=True).run()
