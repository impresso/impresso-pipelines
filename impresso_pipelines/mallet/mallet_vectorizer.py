import jpype
import jpype.imports
import os
import sys
import tempfile  # Add this import
import logging  # Add this import

# Start JVM if not already running
if not jpype.isJVMStarted():
    mallet_path = "/opt/mallet"  # Adjust this path if necessary
    classpath = f"{mallet_path}/class:{mallet_path}/lib/mallet-deps.jar"
    
    # Start JVM with Mallet's classpath
    jpype.startJVM(jpype.getDefaultJVMPath(), f"-Djava.class.path={classpath}")

# Now, import Mallet Java classes
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

    def __call__(self, lemmatized_words: list) -> None:
        """
        Processes a given list of lemmatized words, vectorizing it using Mallet and saves the output to a file.

        Args:
            lemmatized_words (list): The input list of lemmatized words to be vectorized.
        """
        text = " ".join(lemmatized_words)  # Join the list into a single string
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as temp_file:
            input_file = temp_file.name
            temp_file.write(f"id,text\n1,{text}")  # Mallet expects a CSV-like format

        temp_output_file = input_file + ".mallet"

        # Arguments for Csv2Vectors
        arguments = [
            "--input",
            input_file,
            "--output",
            temp_output_file,
            "--keep-sequence",
            "--use-pipe-from",
            self.pipe_file,
        ]

        logging.info(f"Calling Mallet Csv2Vector with arguments: {arguments}")
        self.vectorizer.main(arguments)
        logging.debug("Csv2Vector processing completed.")

        # Read the generated Mallet file and save its content to the specified output file
        with open(temp_output_file, "rb") as mallet_file:  # Change mode to 'rb'
            mallet_output = mallet_file.read().decode('utf-8', errors='ignore')  # Decode with 'ignore' errors

        with open(self.output_file, "w", encoding="utf-8") as output_file:
            output_file.write(mallet_output)

        if not self.keep_tmp_file:
            os.remove(input_file)
            os.remove(temp_output_file)
            logging.info(f"Temporary files deleted: {input_file}, {temp_output_file}")
