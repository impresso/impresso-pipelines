import os
import tempfile
import logging
from typing import Optional
from cc.mallet.classify.tui import Csv2Vectors  # Import after JVM is started

class MalletVectorizer:
    """
    Handles the vectorization of a text string using Mallet without requiring input files.
    """

    def __init__(self, pipe_file: str, keep_tmp_file: bool = False) -> None:
        self.vectorizer = Csv2Vectors()
        self.pipe_file = pipe_file
        self.keep_tmp_file = keep_tmp_file

    def __call__(self, text: str) -> str:
        """
        Processes a given text string, vectorizing it using Mallet.

        Args:
            text (str): The input text to be vectorized.

        Returns:
            str: The Mallet-formatted string.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as temp_file:
            input_file = temp_file.name
            temp_file.write(f"id,text\n1,{text}")  # Mallet expects a CSV-like format

        output_file = input_file + ".mallet"

        # Arguments for Csv2Vectors
        arguments = [
            "--input",
            input_file,
            "--output",
            output_file,
            "--keep-sequence",
            "--use-pipe-from",
            self.pipe_file,
        ]

        logging.info(f"Calling Mallet Csv2Vector with arguments: {arguments}")
        self.vectorizer.main(arguments)
        logging.debug("Csv2Vector processing completed.")

        # Read the generated Mallet file and return its content as a string
        with open(output_file, "r", encoding="utf-8") as mallet_file:
            mallet_output = mallet_file.read()

        if not self.keep_tmp_file:
            os.remove(input_file)
            os.remove(output_file)
            logging.info(f"Temporary files deleted: {input_file}, {output_file}")

        return mallet_output  # Return vectorized text as a string
