# index_manager.py

import json
import os
import logging
from config import INDEX_PATH

logger = logging.getLogger("autonomous_system.index_manager")


class IndexManager:
    def __init__(self, index_path=INDEX_PATH):
        self.index_path = index_path
        self.logger = logging.getLogger("autonomous_system.index_manager")

    def extract_keywords(self, text):
        """Extracts keywords from a given text."""
        return list(set(text.lower().split()))

    def load_index(self):
        """Loads the index from the file."""
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as index_file:
                return json.load(index_file)
        return {}

    def save_index(self, index):
        """Saves the index to the file."""
        with open(self.index_path, "w") as index_file:
            json.dump(index, index_file)

    def index_interaction(self, entry):
        """Indexes an interaction entry."""
        self.logger.debug("Indexing interaction")
        index = self.load_index()
        keywords = self.extract_keywords(
            entry.get("input", "") + " " + entry.get("output", "")
        )
        for keyword in keywords:
            if keyword in index:
                index[keyword].append(entry)
            else:
                index[keyword] = [entry]
        self.save_index(index)

    def search_context(self, keyword):
        """Searches the index for a given keyword."""
        index = self.load_index()
        return index.get(keyword.lower(), [])
