from datetime import datetime
from typing import List, Dict, Any

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont


class Conversation:
    """
    Represents a single conversation session.
    """

    def __init__(self, title: str = "New Conversation"):
        self.id: str = str(int(datetime.now().timestamp() * 1000))
        self.title: str = title
        self.messages: List[Dict[str, Any]] = []
        self.created_at: str = datetime.now().isoformat()
        self.last_message_time: datetime = datetime.now()

    def add_message(self, content: str, is_user: bool, timestamp: str | None = None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")

        self.messages.append({
            "content": content,
            "is_user": is_user,
            "timestamp": timestamp
        })

        self.last_message_time = datetime.now()

        if is_user and len([m for m in self.messages if m["is_user"]]) == 1:
            clean_title = content.strip()
            self.title = clean_title[:35] + "..." if len(clean_title) > 35 else clean_title

    def get_relative_time(self) -> str:
        now = datetime.now()
        diff = now - self.last_message_time

        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60}m ago"
        elif diff.days == 0:
            return f"{diff.seconds // 3600}h ago"
        elif diff.days == 1:
            return "Yesterday"
        else:
            return f"{diff.days}d ago"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": self.messages,
            "created_at": self.created_at,
            "last_message_time": self.last_message_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        conv = cls(data.get("title", "Conversation"))
        conv.id = data["id"]
        conv.messages = data.get("messages", [])
        conv.created_at = data.get("created_at", datetime.now().isoformat())
        if "last_message_time" in data:
            conv.last_message_time = datetime.fromisoformat(data["last_message_time"])
        return conv


class ConversationItem(QWidget):
    """
    Sidebar list item widget for a conversation.
    """

    def __init__(self, conversation: Conversation):
        super().__init__()
        self.conversation = conversation
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        title_label = QLabel(self.conversation.title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        title_label.setStyleSheet("color: #e8e8e8;")

        time_label = QLabel(self.conversation.get_relative_time())
        time_label.setFont(QFont("Segoe UI", 9))
        time_label.setStyleSheet("color: rgba(200, 200, 200, 0.6);")

        layout.addWidget(title_label)
        layout.addWidget(time_label)
