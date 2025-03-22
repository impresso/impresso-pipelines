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
from typing import Optional, List


class MalletVectorizer:
    """
    Handles the vectorization of a lemmatized list using Mallet.
    """

    def __init__(self, pipe_file: str, keep_tmp_file: bool = False) -> None:
        # Import after JVM is started
        from cc.mallet.classify.tui import Csv2Vectors  # type: ignore

        self.vectorizer = Csv2Vectors()
        self.pipe_file = pipe_file
        self.keep_tmp_file = keep_tmp_file

    def run_csv2vectors(
        self,
        lemma_list: List[str],
        output_file: Optional[str] = None,
    ) -> str:
        """
        Vectorize the lemmatized list using Mallet.

        Args:
            lemma_list: List of lemmatized words.
            output_file: Path where the output .mallet file should be saved.

        Returns:
            Path to the generated .mallet file.
        """
        if not output_file:
            output_file = "output.mallet"

        # Create a temporary input file for Mallet
        with open("temp_input.csv", "w", encoding="utf-8") as temp_file:
            temp_file.write("id\tclass\ttext\n")
            temp_file.write(f"1\tdummy\t{' '.join(lemma_list)}\n")

        # Arguments for Csv2Vectors
        arguments = [
            "--input",
            "temp_input.csv",
            "--output",
            output_file,
            "--keep-sequence",
            "--use-pipe-from",
            self.pipe_file,
        ]

        logging.info("Calling Mallet Csv2Vectors with arguments: %s", arguments)
        self.vectorizer.main(arguments)
        logging.debug("Csv2Vectors call finished.")

        if not self.keep_tmp_file:
            os.remove("temp_input.csv")
            logging.info("Deleted temporary input file: temp_input.csv")

        return output_file
