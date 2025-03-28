import jpype
import jpype.imports
import os
import logging
import tempfile
import subprocess
from typing import List
from urllib.request import urlretrieve

# Ensure Mallet JAR files are available
def setup_mallet():
    mallet_dir = "/opt/mallet"
    mallet_deps_jar = os.path.join(mallet_dir, "lib/mallet-deps.jar")
    mallet_jar = os.path.join(mallet_dir, "lib/mallet.jar")

    if not os.path.exists(mallet_deps_jar) or not os.path.exists(mallet_jar):
        os.makedirs(os.path.join(mallet_dir, "lib"), exist_ok=True)
        logging.info("Downloading Mallet JAR files...")
        try:
            urlretrieve(
                "https://github.com/mimno/Mallet/archive/refs/heads/master.zip",
                "/tmp/mallet.zip",
            )
            subprocess.run(
                ["unzip", "-o", "/tmp/mallet.zip", "-d", "/tmp"], check=True
            )
            subprocess.run(
                ["cp", "-r", "/tmp/Mallet-master/class", mallet_dir], check=True
            )
            subprocess.run(
                ["cp", "/tmp/Mallet-master/lib/mallet-deps.jar", mallet_deps_jar], check=True
            )
            subprocess.run(
                ["cp", "/tmp/Mallet-master/lib/mallet.jar", mallet_jar], check=True
            )
            logging.info("Mallet JAR files downloaded and configured.")
        except Exception as e:
            raise RuntimeError(f"Failed to download or configure Mallet JAR files: {e}")

    return mallet_dir

# Start JVM if not already running
if not jpype.isJVMStarted():
    try:
        mallet_path = setup_mallet()
        classpath = f"{mallet_path}/class:{mallet_path}/lib/mallet-deps.jar"
        jpype.startJVM(jpype.getDefaultJVMPath(), f"-Djava.class.path={classpath}")
    except Exception as e:
        raise RuntimeError(f"Failed to start JVM with Mallet classpath: {e}")

# Import Mallet Java class
try:
    from cc.mallet.classify.tui import Csv2Vectors  # Import after JVM starts
except ImportError as e:
    raise ImportError(f"Failed to import Mallet classes. Ensure Mallet JAR files are correctly configured: {e}")

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