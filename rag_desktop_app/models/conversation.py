from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import uuid


@dataclass
class Message:
    role: str
    content: str
    timestamp: str


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    messages: List[Message] = field(default_factory=list)
