import os
import re
import pickle
import logging
import json
from collections import Counter
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def _load_pickle(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


class ApiResponseProcessor:
    """Processes raw API responses to extract, clean, and save structured metadata."""

    def __init__(self, datadir: str, config):
        self.config = config
        self.datadir = datadir
        self.llmemb_dir = os.path.join(os.getcwd(), 'data', datadir, 'LLMEmb_output')
        self.entity_mapping_path = os.path.join(os.getcwd(), 'data', datadir, 'splitting', 'train',
                                                'entity_name_mapping.pkl')
        self.api_user_file_path = os.path.join(self.llmemb_dir, 'api_user_responses.pkl')
        self.api_item_file_path = os.path.join(self.llmemb_dir, 'api_item_responses.pkl')

    def process_all_responses(self):
        """Processes both user and item API responses."""
        self.process_user_responses()
        self.process_item_responses()

    def process_user_responses(self):
        """Extracts and saves structured metadata from user API responses."""
        raw_responses = _load_pickle(self.api_user_file_path)
        extracted_data = self._extract_sections_from_responses(raw_responses)
        self._log_missing_stats(extracted_data, "user")

        user_metadata = {
            uid: {"profile": self._clean_text(data.get("profile")),
                  "reasoning": self._clean_text(data.get("reasoning"))}
            for uid, data in extracted_data.items()
        }

        self._save_metadata(user_metadata, "user_metadata.pkl")

    def process_item_responses(self):
        """Extracts and saves structured metadata from item API responses."""
        raw_responses = _load_pickle(self.api_item_file_path)
        extracted_data = self._extract_sections_from_responses(raw_responses)
        self._log_missing_stats(extracted_data, "item")

        entity_to_id = _load_pickle(self.entity_mapping_path)
        item_metadata = {}
        for name, data in extracted_data.items():
            item_id = entity_to_id.get(name)
            if item_id is not None:
                item_metadata[item_id] = {
                    "profile": self._clean_text(data.get("profile")),
                    "reasoning": self._clean_text(data.get("reasoning"))
                }

        self._save_metadata(item_metadata, "item_metadata.pkl")

    def _extract_sections_from_responses(self, responses: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Parses a dictionary of raw API responses."""
        sections = {}
        for entity_id, response in responses.items():
            try:
                text_content = ''.join(
                    part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                profile, reasoning = self._parse_json_from_text(text_content)
                sections[entity_id] = {"profile": profile, "reasoning": reasoning}
            except Exception:
                sections[entity_id] = {"profile": None, "reasoning": None}
        return sections

    def _parse_json_from_text(self, text: str) -> Tuple[str, str]:
        """Extracts profile and reasoning from a JSON block within a string."""
        cleaned = re.sub(r"^```[^\n]*\n|\n```$", "", text.strip())
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1: return None, None

        json_str = cleaned[start:end + 1]
        json_str = re.sub(r",\s*}$", "}", json_str)  # Fix trailing comma

        try:
            data = json.loads(json_str)
            profile = str(data.get("profile", ""))
            reasoning = str(data.get("reasoning", ""))
            return profile, reasoning
        except json.JSONDecodeError:
            return None, None

    def _clean_text(self, text: str) -> str:
        """Removes predefined labels and extra whitespace from text."""
        if not isinstance(text, str): return ""

        labels = getattr(self.config, "text_cleaning_labels", [])
        for label in labels:
            pattern = re.escape(label).replace(r"\ ", r"[ _]")
            text = re.sub(rf"(?i)(?<!\w){pattern}(?!\w)", "", text)

        text = re.sub(r"[,\:\.\-\_]", " ", text)
        return re.sub(r"\s{2,}", " ", text).strip()

    def _log_missing_stats(self, data: Dict, entity_type: str):
        """Logs statistics about missing data sections."""
        missing_counter = Counter(k for item in data.values() for k, v in item.items() if not v)
        total = len(data)
        logger.info(f"Missing data stats for {total} {entity_type}s: {dict(missing_counter)}")

    def _save_metadata(self, metadata: Dict, filename: str):
        """Saves processed metadata to a pickle file."""
        path = os.path.join(self.llmemb_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved {len(metadata)} {filename.split('_')[0]} metadata entries to {path}")