import subprocess
import sys

# Ensure jpype1 is installed
try:
    import jpype
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jpype1"])
    import jpype

import jpype.imports
import os
import logging
import tempfile
import urllib.request
from typing import List
from huggingface_hub import hf_hub_download

# from impresso_pipelines.mallet.mallet_setup import initialize_jvm

# Start JVM if not already running
# initialize_jvm()


# Import Mallet Java class
from cc.mallet.classify.tui import Csv2Vectors  # Import after JVM starts

class MalletVectorizer:
    """
    Handles the vectorization of a list of lemmatized words using Mallet without requiring input files.
    """

    def __init__(self, pipe_file: str, output_file: str, keep_tmp_file: bool = False) -> None:
        self.vectorizer = Csv2Vectors()
        self.pipe_file = pipe_file
        self.output_file = os.path.join(os.path.dirname(__file__), output_file)  # Save in the same folder
        self.keep_tmp_file = keep_tmp_file

    def __call__(self, lemmatized_words: List[str], doc_name) -> str:
        """
        Processes a given list of lemmatized words, vectorizing it using Mallet and returns the output file path.

        Args:
            lemmatized_words (list): The input list of lemmatized words to be vectorized.
        
        Returns:
            str: Path to the generated .mallet file.
        """
        # Create a temporary input file for Mallet
        temp_input_file = tempfile.NamedTemporaryFile(
            prefix="temp_input_", suffix=".csv", dir=os.path.dirname(self.output_file), delete=False
        )
        with open(temp_input_file.name, "w", encoding="utf-8") as temp_file:
            # temp_file.write("id\tclass\ttext\n")
            temp_file.write(f"{doc_name}\tdummy\t{' '.join(lemmatized_words)}\n")
            # temp_file.write(f"USERINPUT-001\tdummy\t{' '.join(lemmatized_words)}\n")


        # Arguments for Csv2Vectors
        arguments = [
            "--input", temp_input_file.name,
            "--output", self.output_file,
            "--keep-sequence",
            "--use-pipe-from", self.pipe_file,
        ]

        logging.info("Calling Mallet Csv2Vectors with arguments: %s", arguments)
        self.vectorizer.main(arguments)
        logging.debug("Csv2Vectors call finished.")

        if not self.keep_tmp_file:
            os.remove(temp_input_file.name)
            logging.info("Deleted temporary input file: %s", temp_input_file.name)

        return self.output_file