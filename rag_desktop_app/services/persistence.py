import json
from pathlib import Path
from typing import List

from rag_desktop_app.models.conversation import Conversation, Message


DATA_DIR = Path.home() / ".rag_desktop"
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "conversations.json"


class ConversationStore:
    def load_all(self) -> List[Conversation]:
        if not DATA_FILE.exists():
            return []

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        conversations = []
        for c in raw:
            conversations.append(
                Conversation(
                    id=c["id"],
                    title=c["title"],
                    created_at=c["created_at"],
                    messages=[Message(**m) for m in c["messages"]],
                )
            )
        return conversations

    def save_all(self, conversations: List[Conversation]):
        data = []
        for c in conversations:
            data.append({
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp
                    }
                    for m in c.messages
                ]
            })

        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
