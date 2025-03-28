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
import urllib.request
from typing import Dict
from impresso_pipelines.mallet.mallet_vectorizer import MalletVectorizer
import subprocess
import sys

# Ensure jpype1 is installed
try:
    import jpype
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jpype1"])
    import jpype

# Ensure Mallet JAR files are available
def setup_mallet_jars():
    mallet_dir = "/content/mallet"  # Directory to store Mallet files on Colab
    os.makedirs(mallet_dir, exist_ok=True)

    jar_files = {
        "mallet-deps.jar": "https://huggingface.co/impresso-project/mallet-topic-inferencer/resolve/main/mallet/lib/mallet-deps.jar",
        "mallet.jar": "https://huggingface.co/impresso-project/mallet-topic-inferencer/resolve/main/mallet/lib/mallet.jar",
    }

    for jar_name, jar_url in jar_files.items():
        jar_path = os.path.join(mallet_dir, jar_name)
        if not os.path.exists(jar_path):
            logging.info(f"Downloading {jar_name} from {jar_url}")
            urllib.request.urlretrieve(jar_url, jar_path)

    return mallet_dir

# Start JVM if not already running
if not jpype.isJVMStarted():
    mallet_dir = setup_mallet_jars()
    classpath = f"{mallet_dir}/mallet.jar:{mallet_dir}/mallet-deps.jar"
    
    # Start JVM with Mallet's classpath
    jpype.startJVM(jpype.getDefaultJVMPath(), f"-Djava.class.path={classpath}")


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

        # Create a temporary copy of the pipe file because Mallet modifies it in this
        # version of mallet. If run in parallel, the pipe file will be corrupted.
        with tempfile.NamedTemporaryFile(delete=True) as temp_pipe_file:
            shutil.copyfile(self.pipe_file, temp_pipe_file.name)
            temp_pipe_file_path = temp_pipe_file.name

            # Vectorize the input file and write to a temporary file
            vectorizer = MalletVectorizer(
                language=self.language,
                pipe_file=temp_pipe_file_path,
                keep_tmp_file=self.keep_tmp_files,
            )
            mallet_file = vectorizer.run_csv2vectors(csv_file)

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
                os.remove(mallet_file)
                logging.debug("Deleting temporary mallet input file: %s", mallet_file)

        return topics_file

    def run_csv2topics(
        self, mallet_file: str, delete_mallet_file_after: bool = True
    ) -> str:
        """
        Perform topic inference on a pre-vectorized Mallet file.
        Returns the path to the output topics file.
        """
        if not os.path.exists(mallet_file):
            raise FileNotFoundError(
                f"Mallet file not found: {mallet_file}. Ensure the file exists before running inference."
            )

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

        # Check if the .doctopics file was created
        if not os.path.exists(topics_file):
            raise FileNotFoundError(
                f"Expected output file not found: {topics_file}. "
                f"Check if the Mallet InferTopics command executed correctly."
            )

        if (
            logging.getLogger().getEffectiveLevel() != logging.DEBUG
            and delete_mallet_file_after
            and not self.keep_tmp_files
        ):
            os.remove(mallet_file)
            logging.debug("Deleting temporary mallet input file: %s", mallet_file)

        return topics_file