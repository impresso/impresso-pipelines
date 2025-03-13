"""
This module provides a language identification pipeline using the floret model.
"""

from typing import Optional
from huggingface_hub import hf_hub_download, list_repo_files
import floret
import re

class LangIdentPipeline:
    """
    A pipeline for language identification using a pre-trained floret model.
    """
    
    def __init__(self, model_id: Optional[str] = None, repo_id: str = "impresso-project/impresso-floret-langident", revision: str = "main"):
        """
        Initialize the LangIdentPipeline with the specified or newest model from the repository.

        Args:
            model_id (str, optional): The specific model file to use. If not provided, the newest model will be used.
            repo_id (str): The repository ID on Hugging Face Hub.
            revision (str): The revision of the repository.
        """
        if model_id is None:
            repo_files = list_repo_files(repo_id, revision=revision)
            model_files = [file for file in repo_files if re.match(r"langident-v\d+\.\d+\.\d+\.bin", file)]
            if not model_files:
                raise ValueError("No model files found in the repository.")
            
            # Sort model files by version and select the newest one
            model_files.sort(key=lambda x: list(map(int, re.search(r"v(\d+\.\d+\.\d+)", x).group(1).split('.'))), reverse=True)
            model_id = model_files[0]

        model_path = hf_hub_download(repo_id=repo_id, filename=model_id, revision=revision)
        self.model = floret.load_model(model_path)
        self.model_name = model_id

    def __call__(self, text: str, diagnostics: bool = False, model_id: bool = False) -> dict:
        """
        Identify the language of the given text.

        Args:
            text (str): The input text to identify the language for.
            diagnostics (bool): Whether to include diagnostic information in the output.
            model_id (bool): Whether to include the model name in the output.

        Returns:
            dict: The identified language code, score, and optionally diagnostics and model name.
        """
        text = text.replace("\n", " ") # Remove newlines

        output = self.model.predict(text, k=300 if diagnostics else 1)
        language, value = output
  
        value = [round(num, 2) for num in value]
        
        score = value[0]
    

        result = {"language": language[0].replace("__label__", ""), "score": score}

        if diagnostics:
            languages = [{"language": lang.replace("__label__", ""), "score": val} for lang, val in zip(language, value)]
            result["diagnostics"] = {"languages": languages}

        if model_id:
            result["model_id"] = self.model_name

        return result
