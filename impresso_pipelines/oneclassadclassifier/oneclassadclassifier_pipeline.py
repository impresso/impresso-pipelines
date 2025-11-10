"""
Ad Classification Pipeline for historical newspaper content.

This pipeline identifies advertisements in historical newspaper text using a fine-tuned
XLM-RoBERTa model combined with rule-based features and adaptive thresholding.
"""

import re
import math
from typing import Union, List, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Text normalization
NORM_SPACES = re.compile(r"\s{2,}")

def normalize_text(t: str) -> str:
    """Light OCR cleaning and normalization."""
    if not isinstance(t, str):
        return ""
    t = re.sub(r"[_`~^]+", " ", t)
    t = re.sub(r"\s+([.,:;!?()])", r"\1", t)
    t = NORM_SPACES.sub(" ", t)
    return t.strip()


def chunk_words(text: str, max_words: int):
    """Split text into word-based chunks."""
    if max_words <= 0:
        yield text
        return
    ws = text.split()
    for i in range(0, len(ws), max_words):
        yield " ".join(ws[i:i+max_words])


# Rule-based patterns
PHONE = re.compile(r"(?:\+?\d{2,3}[\s./-]?)?(?:\(?0\d{1,3}\)?[\s./-]?)\d(?:[\d\s./-]{5,})")
PRICE = re.compile(
    r"(?:CHF|SFr\.?|Fr\.?|fr\.?|€|\$|USD|EUR)\s?\d{1,4}(?:[''`\s\.,]?\d{3})*(?:\.-|[.-]|'-)?"
    r"|(?:\d{1,4}(?:[''`\s\.,]?\d{3})*)(?:\s?(?:CHF|SFr\.?|Fr\.?|€|\$|USD|EUR))(?:\.-|[.-]|'-)?",
    re.I
)
AREA = re.compile(r"\b\d{2,4}\s?m(?:²|2)\b")
ROOMS_FR = re.compile(r"\b(\d{1,2})\s?pi[eè]ce?s?\b", re.I)
CUES_FR = r"(?:à\s?vendre|a\s?vendre|à\s?louer|a\s?louer|à\s?remettre|prix\s+à\s+discuter|écrire\s+à|sous\s+chiffres|tél\.?|téléphone|loyer|charges|villa|attique|expertisée|contact|offre|demande|urgent|occasion|affaire)"
CUES_DE = r"(?:zu\s?verkaufen|zu\s?vermieten|Preis|Schreib(?:en)?\s+an|unter\s+Chiffre|Tel\.?|Telefon|Miete|Zimmer|Attika|expertisiert|Kontakt|Angebot|dringend|Gelegenheit)"
CUES_LB = r"(?:ze\s?verkafen|ze\s?verlounen|Präis|Annonce|Tel\.?|Telefon|Kontakt)"
CUES = re.compile(fr"\b(?:{CUES_FR}|{CUES_DE}|{CUES_LB})\b", re.I)

ADDRESS = re.compile(r"\b(Rue|Av\.?|Avenue|Platz|Str\.?|Strasse|Grand'Rue|Place)\b", re.I)
ZIP_CH = re.compile(r"\b\d{4}\b")


def rule_flags(t: str) -> Dict[str, Any]:
    """Extract rule-based features from text."""
    return {
        "has_phone": bool(PHONE.search(t)),
        "has_price": bool(PRICE.search(t)),
        "has_area": bool(AREA.search(t)),
        "has_rooms": bool(ROOMS_FR.search(t)),
        "has_cue": bool(CUES.search(t)),
        "has_address": bool(ADDRESS.search(t)),
        "has_zip": bool(ZIP_CH.search(t)),
        "len_words": len(t.split()),
    }


def calculate_rule_score_and_confidence(flags: Dict[str, Any]):
    """Calculate rule score with balanced weights and confidence measure."""
    rule_score = (
        2.0 * float(flags["has_price"])
        + 2.0 * float(flags["has_phone"])
        + 1.5 * float(flags["has_cue"])
        + 1.0 * float(flags["has_area"])
        + 1.0 * float(flags["has_rooms"])
        + 0.8 * float(flags["has_address"])
        + 0.5 * float(flags["has_zip"])
    )
    
    strong_indicators = flags["has_price"] + flags["has_phone"]
    medium_indicators = flags["has_cue"] + flags["has_area"] + flags["has_rooms"]
    weak_indicators = flags["has_address"] + flags["has_zip"]
    
    rule_confidence = min(1.0, (strong_indicators * 0.4 + medium_indicators * 0.2 + weak_indicators * 0.1))
    
    return rule_score, rule_confidence


def calculate_ensemble_ad_signal(promo_prob: float, top_label: str, all_probs: np.ndarray, id2label: Dict[int, str]) -> float:
    """Calculate ensemble signal from all predictions."""
    ad_like_labels = ["Promotion", "Obituary", "Call for participation"]
    non_ad_labels = ["News", "Opinion", "Article", "Report"]
    
    ad_signal = sum(all_probs[i] for i, lbl in id2label.items() if lbl in ad_like_labels)
    non_ad_signal = sum(all_probs[i] for i, lbl in id2label.items() if lbl in non_ad_labels)
    
    ensemble = ad_signal * 0.7 + (1.0 - non_ad_signal) * 0.3
    return float(ensemble)


def parse_lang_thresholds(s: str) -> Dict[str, float]:
    """Parse language-specific thresholds from string like 'fr:0.58,de:0.62'."""
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if ":" in pair:
            lang, val = pair.split(":", 1)
            out[lang.strip().lower()] = float(val)
    return out


def lang_len_threshold(lang: str, n_words: int, lang_thr_map: Dict[str, float], 
                       default_thr: float, short_bonus: float, short_len: int) -> float:
    """Get adaptive threshold based on language and text length."""
    base = lang_thr_map.get(lang, default_thr)
    if n_words < short_len:
        base = max(0.0, base - short_bonus)
    return base


class AdClassificationPipeline:
    """
    Pipeline for classifying advertisements in historical newspaper content.
    
    Usage:
        pipeline = AdClassificationPipeline()
        
        # Single text
        result = pipeline("Dies ist ein Beispieltext...")
        
        # List of texts
        results = pipeline(["Text 1", "Text 2"])
        
        # Dictionary with 'ft' field
        result = pipeline({"id": "doc1", "ft": "Text content", "lg": "de"})
        
        # List of dictionaries
        results = pipeline([{"ft": "Text 1", "lg": "fr"}, {"ft": "Text 2"}])
    """
    
    def __init__(
        self,
        model_name: str = "impresso-project/impresso-ad-classification-xlm-one-class",
        batch_size: int = 16,
        max_length: int = 512,
        chunk_words: int = 0,
        pool: str = "logits_max",
        ad_threshold: float = 0.9991338849067688,
        lang_thresholds: str = "other:0.9991,fr:0.0755",
        short_len: int = 30,
        short_bonus: float = 0.2,
        temperature: float = 0.8,
        device: Optional[str] = None,
    ):
        """
        Initialize the ad classification pipeline.
        
        Args:
            model_name: HuggingFace model ID or local path
            batch_size: Batch size for processing
            max_length: Maximum token length per chunk
            chunk_words: Words per chunk (0 = no chunking)
            pool: Pooling strategy for chunks
            ad_threshold: Default threshold for ad classification
            lang_thresholds: Language-specific thresholds (e.g., "fr:0.58,de:0.62")
            short_len: Word count considered 'short'
            short_bonus: Threshold reduction for short texts
            temperature: Calibration temperature
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.chunk_words = chunk_words
        self.pool = pool
        self.ad_threshold = ad_threshold
        self.short_len = short_len
        self.short_bonus = short_bonus
        self.temperature = temperature
        self.lang_thr_map = parse_lang_thresholds(lang_thresholds) if lang_thresholds else {}
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.to(self.device).eval()
        
        self.id2label = self.model.config.id2label
        self.promo_id = None
        for i, lab in self.id2label.items():
            if lab.lower() == "promotion":
                self.promo_id = i
                break
        if self.promo_id is None:
            raise RuntimeError("Could not find 'Promotion' label in model config")
    
    def __call__(
        self, 
        inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Classify text(s) as ad or non-ad.
        
        Args:
            inputs: Text string, list of strings, dict with 'ft' field, or list of dicts
            
        Returns:
            Dictionary or list of dictionaries with classification results
        """
        # Normalize inputs to list of dicts
        normalized_inputs = self._normalize_inputs(inputs)
        is_single = isinstance(inputs, (str, dict))
        
        # Process
        results = self._process_batch(normalized_inputs)
        
        # Return single result or list
        return results[0] if is_single else results
    
    def _normalize_inputs(
        self, 
        inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Normalize various input formats to list of dicts."""
        if isinstance(inputs, str):
            return [{"ft": inputs}]
        elif isinstance(inputs, dict):
            return [inputs]
        elif isinstance(inputs, list):
            if not inputs:
                return []
            if isinstance(inputs[0], str):
                return [{"ft": text} for text in inputs]
            else:
                return inputs
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
    
    def _process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of items."""
        results = []
        
        # Prepare chunks for all items
        all_texts = []
        all_meta = []
        all_chunk_counts = []
        all_chunk_lens = []
        
        for item in items:
            txt = item.get("ft", "")
            txt = normalize_text(txt)
            
            if self.chunk_words > 0:
                parts = list(chunk_words(txt, self.chunk_words))
            else:
                parts = [txt]
            
            lens = [len(p.split()) for p in parts]
            
            all_texts.extend(parts)
            all_meta.append(item)
            all_chunk_counts.append(len(parts))
            all_chunk_lens.extend(lens)
        
        # Process in batches
        all_logits = []
        for i in range(0, len(all_texts), self.batch_size):
            batch_texts = all_texts[i:i + self.batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            
            with torch.no_grad():
                logits = self.model(**enc).logits
            all_logits.append(logits.cpu().numpy())
        
        all_logits = np.concatenate(all_logits, axis=0)
        
        # Process each item
        pos = 0
        for meta, n_chunks in zip(all_meta, all_chunk_counts):
            L = all_logits[pos:pos+n_chunks]
            lens = all_chunk_lens[pos:pos+n_chunks]
            pos += n_chunks
            
            # Pool across chunks
            pooled_probs = self._pool_logits(L, lens)
            
            promo_prob = float(pooled_probs[self.promo_id])
            top_id = int(np.argmax(pooled_probs))
            top_label = self.id2label[top_id]
            top_prob = float(pooled_probs[top_id])
            
            # Calculate ensemble signal
            ensemble_ad_signal = calculate_ensemble_ad_signal(
                promo_prob, top_label, pooled_probs, self.id2label
            )
            
            # Get text and language
            lg = (meta.get("lg") or meta.get("lang") or "").lower()
            text_raw = meta.get("ft", "")
            text_norm = normalize_text(text_raw)
            flags = rule_flags(text_norm)
            
            # Calculate threshold
            base_thr = lang_len_threshold(
                lg, flags["len_words"], self.lang_thr_map, 
                self.ad_threshold, self.short_bonus, self.short_len
            )
            
            # Blend probabilities with ensemble signal
            final_prob = promo_prob * 0.85 + ensemble_ad_signal * 0.15
            
            # Apply rule-based adjustments
            rule_score, rule_confidence = calculate_rule_score_and_confidence(flags)
            model_confidence = abs(promo_prob - 0.5) * 2
            model_uncertainty = 1.0 - model_confidence
            
            if model_confidence < 0.75:
                rule_influence = 0.3 + (model_uncertainty * 1.2)
                
                if rule_confidence > 0.7 and rule_score >= 4.0:
                    boost = max(0.15 * rule_influence, 0.03)
                    final_prob = max(final_prob, base_thr + boost)
                elif rule_confidence > 0.5 and rule_score >= 3.0:
                    boost = max(0.12 * rule_influence, 0.02)
                    final_prob = max(final_prob, base_thr + boost)
                
                # Combination bonuses
                if flags["has_price"] and flags["has_phone"]:
                    combination_boost = 0.16 * rule_influence
                    final_prob = max(final_prob, min(final_prob + combination_boost, 0.92))
            
            # Final classification
            is_ad_pred = bool(final_prob >= base_thr)
            
            # Build result
            result = {
                "id": meta.get("id"),
                "type": "ad" if is_ad_pred else "non-ad",
                "promotion_prob": round(promo_prob, 6),
                "promotion_prob_final": round(final_prob, 6),
                "ensemble_ad_signal": round(ensemble_ad_signal, 6),
                "xgenre_top_label": top_label,
                "xgenre_top_prob": round(top_prob, 6),
                "threshold_used": round(base_thr, 6),
                "rule_score": round(rule_score, 3),
                "rule_confidence": round(rule_confidence, 3),
                "model_confidence": round(model_confidence, 3),
            }
            
            results.append(result)
        
        return results
    
    def _pool_logits(self, L: np.ndarray, lens: List[int]) -> np.ndarray:
        """Pool logits across chunks."""
        if self.pool == "logits_max":
            pooled_logits = L.max(axis=0)
        elif self.pool == "logits_mean":
            pooled_logits = L.mean(axis=0)
        elif self.pool == "logits_weighted":
            w = np.array(lens, dtype=np.float32)
            w = w / (w.sum() + 1e-9)
            pooled_logits = (L * w[:, None]).sum(axis=0)
        else:
            pooled_logits = L.mean(axis=0)
        
        # Apply temperature
        pooled_logits = pooled_logits / max(self.temperature, 1e-6)
        pooled_probs = torch.softmax(torch.tensor(pooled_logits), dim=-1).numpy()
        
        return pooled_probs