import jpype
import jpype.imports
import os
import logging
import tempfile
from typing import List

# Start JVM if not already running
if not jpype.isJVMStarted():
    mallet_path = "/opt/mallet"  # Adjust this path if necessary
    classpath = f"{mallet_path}/class:{mallet_path}/lib/mallet-deps.jar"
    
    # Start JVM with Mallet's classpath
    jpype.startJVM(jpype.getDefaultJVMPath(), f"-Djava.class.path={classpath}")

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

    def __call__(self, lemmatized_words: List[str]) -> str:
        """
        Processes a given list of lemmatized words, vectorizing it using Mallet and returns the output file path.

        Args:
            lemmatized_words (list): The input list of lemmatized words to be vectorized.
        
        Returns:
            str: Path to the generated .mallet file.
        """
        # Create a temporary input file for Mallet
        temp_input_file = "temp_input.csv"
        with open(temp_input_file, "w", encoding="utf-8") as temp_file:
            temp_file.write("id\tclass\ttext\n")
            temp_file.write(f"1\tdummy\t{' '.join(lemmatized_words)}\n")

        # Arguments for Csv2Vectors
        arguments = [
            "--input", temp_input_file,
            "--output", self.output_file,
            "--keep-sequence",
            "--use-pipe-from", self.pipe_file,
        ]

        logging.info("Calling Mallet Csv2Vectors with arguments: %s", arguments)
        self.vectorizer.main(arguments)
        logging.debug("Csv2Vectors call finished.")

        if not self.keep_tmp_file:
            os.remove(temp_input_file)
            logging.info("Deleted temporary input file: %s", temp_input_file)

        return self.output_file