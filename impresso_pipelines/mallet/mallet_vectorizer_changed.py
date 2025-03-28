import jpype
import jpype.imports
import os
import logging
import tempfile
from huggingface_hub import hf_hub_download  # Import for downloading JAR files
from typing import List

# Ensure Mallet JAR files are available
def setup_mallet_jars():
    mallet_dir = "/tmp/mallet"  # Use a temporary directory for Colab
    os.makedirs(mallet_dir, exist_ok=True)

    # Define the required JAR files
    jar_files = {
        "mallet-deps.jar": "mallet/lib/mallet-deps.jar",
        "mallet.jar": "mallet/lib/mallet.jar",
    }

    # Download JAR files from Hugging Face if they are missing
    for jar_name, jar_path in jar_files.items():
        local_path = os.path.join(mallet_dir, jar_name)
        if not os.path.exists(local_path):
            print(f"Downloading {jar_name}...")
            hf_hub_download(
                repo_id="impresso-project/mallet-topic-inferencer",
                filename=jar_path,
                local_dir=mallet_dir,
                local_dir_use_symlinks=False,
            )
    return mallet_dir

# Start JVM if not already running
if not jpype.isJVMStarted():
    mallet_dir = setup_mallet_jars()
    classpath = f"{mallet_dir}/mallet-deps.jar:{mallet_dir}/mallet.jar"
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
        temp_input_file = tempfile.NamedTemporaryFile(
            prefix="temp_input_", suffix=".csv", dir=os.path.dirname(self.output_file), delete=False
        )
        with open(temp_input_file.name, "w", encoding="utf-8") as temp_file:
            temp_file.write("id\tclass\ttext\n")
            temp_file.write(f"1\tdummy\t{' '.join(lemmatized_words)}\n")

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