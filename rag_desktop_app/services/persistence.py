import json
import os
from typing import List

from rag_desktop_app.models.conversation import Conversation


class ConversationStorage:
    """
    Handles saving and loading conversations from disk.
    Pure persistence layer – no UI, no backend logic.
    """

    def __init__(self, file_path: str = "conversations.json"):
        self.file_path = file_path

    # ------------------------------------------------------------------
    # Save conversations
    # ------------------------------------------------------------------
    def save(self, conversations: List[Conversation]) -> None:
        try:
            data = [conv.to_dict() for conv in conversations]
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Intentionally not raising UI errors here
            # UI layer decides how to notify the user
            print(f"[Persistence] Error saving conversations: {e}")

    # ------------------------------------------------------------------
    # Load conversations
    # ------------------------------------------------------------------
    def load(self) -> List[Conversation]:
        if not os.path.exists(self.file_path):
            return []

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return [Conversation.from_dict(item) for item in data]

        except Exception as e:
            print(f"[Persistence] Error loading conversations: {e}")
            return []
