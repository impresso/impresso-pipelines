#!/usr/bin/env python3
"""

This module provides the LanguageInferencer class, which manages Mallet topic inference
for a specific language. It loads the inferencer and pipe file during initialization and
provides functionality to perform topic inference on input files.

Classes:
    LanguageInferencer: A class to manage Mallet inferencing for a specific language.

Usage example:
    inferencer = LanguageInferencer(language='en', inferencer_file='path/to/inferencer',
        pipe_file='path/to/pipe')
    topics = inferencer.run_csv2topics(csv_file='path/to/csv')


    Attributes:
        language (str): The language for which to perform topic inference.
        inferencer_file (str): Path to the Mallet inferencer file.
        inferencer (InferTopics): Instance of Mallet's InferTopics class.
        pipe_file (str): Path to the Mallet pipe file.
        vectorizer (MalletVectorizer): Instance of MalletVectorizer for vectorizing
            input files.
        keep_tmp_files (bool): Flag to indicate whether to keep temporary files.

    Methods:
        run_csv2topics(csv_file: str, delete_mallet_file_after: bool = True) -> Dict[str, str]:
            Perform topic inference on a single input file and return a dictionary of
            document_id -> topic distributions.
"""

import os
import logging
import shutil
import tempfile
from typing import Dict
from impresso_pipelines.mallet.mallet_vectorizer import MalletVectorizer


class LanguageInferencer:
    """
    A class to manage Mallet inferencing for a specific language.
    Loads the inferencer and pipe file during initialization.
    """

    def __init__(
        self,
        language: str,
        inferencer_file: str,
        pipe_file: str,
        keep_tmp_files: bool = False,
        random_seed: int = 42,
    ) -> None:
        # Import after JVM is started, so that the classes are available
        # noinspection PyUnresolvedReferences
        from cc.mallet.topics.tui import InferTopics  # type: ignore

        self.language = language
        self.inferencer_file = inferencer_file
        self.inferencer = InferTopics()
        self.pipe_file = pipe_file
        self.vectorizer = MalletVectorizer(
            language=language, pipe_file=self.pipe_file, keep_tmp_file=keep_tmp_files
        )
        self.keep_tmp_files = keep_tmp_files
        self.random_seed = random_seed

        if not os.path.exists(self.inferencer_file):
            raise FileNotFoundError(
                f"Inferencer file not found: {self.inferencer_file}"
            )

    def run_csv2topics(
        self, csv_file: str, delete_mallet_file_after: bool = True
    ) -> Dict[str, str]:
        """
        Perform topic inference on a single input file.
        The input file should be in the format expected by Mallet.
        Returns a dictionary of document_id -> topic distributions.
        """

        # Retrieve the Mallet file directly from the repository
        mallet_file = os.path.join(os.path.dirname(__file__), "output.mallet")
        if not os.path.exists(mallet_file):
            raise FileNotFoundError(f"Mallet file not found: {mallet_file}")

        topics_file = mallet_file + ".doctopics"

        arguments = [
            "--input",
            mallet_file,
            "--inferencer",
            self.inferencer_file,
            "--output-doc-topics",
            topics_file,
            "--random-seed",
            str(self.random_seed),
        ]

        logging.info("Calling mallet InferTopics: %s", arguments)

        self.inferencer.main(arguments)
        logging.debug("InferTopics call finished.")

        if (
            logging.getLogger().getEffectiveLevel() != logging.DEBUG
            and delete_mallet_file_after
            and not self.keep_tmp_files
        ):
            logging.debug("Skipping deletion of mallet file: %s", mallet_file)

        return topics_file
