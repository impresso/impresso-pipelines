#!/usr/bin/env python3
"""
This module provides the MalletVectorizer class for vectorizing documents into a format usable by Mallet.

Classes:
    MalletVectorizer: Vectorizes documents using Mallet's Csv2Vectors.

Usage example:
    vectorizer = MalletVectorizer(language="en", pipe_file="path/to/pipe/file")
    vectorized_file = vectorizer.run_csv2vectors(input_file="path/to/input.csv")

Attributes:
    language (str): Language of the documents.
    pipe_file (str): Path to the pipe file used by Mallet.
    keep_tmp_file (bool): Whether to keep the temporary input file after vectorization.
"""

import os
import logging
from typing import Optional


class MalletVectorizer:
    """
    Handles the vectorization of multiple documents into a format usable by Mallet
    using the pipe file from the model.
    """

    def __init__(
        self, language: str, pipe_file: str, keep_tmp_file: bool = False
    ) -> None:

        # noinspection PyUnresolvedReferences
        from cc.mallet.classify.tui import Csv2Vectors  # type: ignore # Import after JVM is started

        self.vectorizer = Csv2Vectors()

        self.pipe_file = pipe_file
        self.language = language
        self.keep_tmp_file = keep_tmp_file

    def run_csv2vectors(
        self,
        input_file: str,
        output_file: Optional[str] = None,
    ) -> str:
        """
        Run Csv2Vectors to vectorize the input file.

        Simple java-internal command line interface to the Csv2Vectors class in Mallet.

        Args:
            input_file: Path to the csv input file to be vectorized.
            output_file: Path where the output .mallet file should be saved.
        """

        if not output_file:
            output_file = input_file + ".mallet"

        # Arguments for Csv2Vectors java main class
        arguments = [
            "--input",
            input_file,
            "--output",
            output_file,
            "--keep-sequence",  # Keep sequence for feature extraction
            # "--encoding",
            # "UTF-8",
            "--use-pipe-from",
            self.pipe_file,
        ]

        logging.info("Calling mallet Csv2Vector: %s", arguments)
        self.vectorizer.main(arguments)

        logging.debug("Csv2Vector call finished.")
        if not self.keep_tmp_file:
            os.remove(input_file)
            logging.info("Deleting temporary file: %s", input_file)

        return output_file