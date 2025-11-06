"""
Language identification pipeline using the floret model.

This module provides a pipeline for detecting the language of text using
pre-trained floret models from the Hugging Face Hub.

"""

import logging
import re
from typing import Dict, List, Optional, Any

import floret
from huggingface_hub import hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)


class LangIdentPipeline:
    """
    Pipeline for language identification using a pre-trained floret model.
    
    The pipeline automatically downloads and caches models from Hugging Face Hub.
    If no specific model is provided, it will use the latest version available.
    
    Example:
        >>> pipeline = LangIdentPipeline()
        >>> result = pipeline("Ceci est un texte en français")
        >>> print(result['language'])
        'fr'
    """
    
    def __init__(
        self, 
        model_id: Optional[str] = None, 
        repo_id: str = "impresso-project/impresso-floret-langident", 
        revision: str = "main"
    ) -> None:
        """
        Initialize the language identification pipeline.

        Args:
            model_id: Specific model file to use (e.g., "langident-v1.2.3.bin").
                If None, automatically selects the newest available model.
            repo_id: Hugging Face repository ID containing the models.
            revision: Git revision of the repository (branch, tag, or commit).
            
        Raises:
            ValueError: If no model files are found in the repository.
        """
        if model_id is None:
            logger.info(f"No model specified, fetching latest from {repo_id}")
            repo_files = list_repo_files(repo_id, revision=revision)
            model_files = [
                file for file in repo_files 
                if re.match(r"langident-v\d+\.\d+\.\d+\.bin", file)
            ]
            
            if not model_files:
                raise ValueError(f"No model files found in repository {repo_id}")
            
            # Sort model files by semantic version and select the newest
            model_files.sort(
                key=lambda x: list(
                    map(int, re.search(r"v(\d+\.\d+\.\d+)", x).group(1).split('.'))
                ), 
                reverse=True
            )
            model_id = model_files[0]
            logger.info(f"Selected latest model: {model_id}")
        else:
            logger.info(f"Using specified model: {model_id}")

        logger.debug(f"Downloading model from {repo_id}/{model_id}")
        model_path = hf_hub_download(
            repo_id=repo_id, 
            filename=model_id, 
            revision=revision
        )
        logger.debug(f"Model downloaded to: {model_path}")
        
        self.model = floret.load_model(model_path)
        self.model_name = model_id
        logger.info(f"Language identification model loaded: {self.model_name}")

    def __call__(
        self, 
        text: str, 
        diagnostics: bool = False, 
        model_id: bool = False
    ) -> Dict[str, Any]:
        """
        Identify the language of the given text.

        Args:
            text: Input text to identify the language for.
            diagnostics: If True, includes top 300 language predictions with scores.
            model_id: If True, includes the model filename in the output.

        Returns:
            Dictionary containing:
                - language (str): Detected language code (e.g., 'en', 'fr', 'de')
                - score (float): Confidence score between 0 and 1
                - diagnostics (dict, optional): Top language predictions if requested
                - model_id (str, optional): Model filename if requested
                
        Example:
            >>> pipeline = LangIdentPipeline()
            >>> result = pipeline("Hello world", diagnostics=True)
            >>> result['language']
            'en'
            >>> result['score']
            0.98
        """
        # Normalize text: replace newlines with spaces
        normalized_text = text.replace("\n", " ")
        
        logger.debug(f"Identifying language for text of length {len(text)}")
        
        # Get predictions (k=300 for diagnostics, k=1 for standard)
        k = 300 if diagnostics else 1
        output = self.model.predict(normalized_text, k=k)
        languages, scores = output
  
        # Round scores to 2 decimal places
        scores = [round(score, 2) for score in scores]
        
        top_language = languages[0].replace("__label__", "")
        top_score = scores[0]
        
        logger.debug(f"Detected language: {top_language} (score: {top_score})")

        result: Dict[str, Any] = {
            "language": top_language, 
            "score": top_score
        }

        if diagnostics:
            language_predictions: List[Dict[str, Any]] = [
                {
                    "language": lang.replace("__label__", ""), 
                    "score": score
                } 
                for lang, score in zip(languages, scores)
            ]
            result["diagnostics"] = {"languages": language_predictions}
            logger.debug(f"Returning {len(language_predictions)} language predictions")

        if model_id:
            result["model_id"] = self.model_name

        return result
