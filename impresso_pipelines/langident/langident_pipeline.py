"""
This module provides a language identification pipeline using the floret model.
"""

from huggingface_hub import hf_hub_download
import floret

class LangIdentPipeline:
    """
    A pipeline for language identification using a pre-trained floret model.
    """
    
    def __init__(self, model_name: str = "LID-40-3-2000000-1-4.bin", repo_id: str = "impresso-project/impresso-floret-langident", revision: str = "main"):
        """
        Initialize the LangIdentPipeline with the specified model.

        Args:
            model_name (str): The name of the model file.
            repo_id (str): The repository ID on Hugging Face Hub.
            revision (str): The revision of the repository.
        """
        model_path = hf_hub_download(repo_id=repo_id, filename=model_name, revision=revision)
        self.model = floret.load_model(model_path)
        self.model_name = model_name

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
        output = self.model.predict(text, k=300 if diagnostics else 1)
        language, value = output
  
        value = [round(num, 3) for num in value]
        
        score = value[0]
    

        result = {"language": language[0].replace("__label__", ""), "score": score}

        if diagnostics:
            language_dist = [{"language": lang.replace("__label__", ""), "score": val} for lang, val in zip(language, value)]
            result["diagnostics"] = {"language_dist": language_dist}

        if model_id:
            result["model_name"] = self.model_name

        return result
