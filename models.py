from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Foresight:
    id: Optional[int] = None
    description: str = ""
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    evidence: str = ""
    duration_days: Optional[int] = None
    is_active: bool = True
    created_at: Optional[datetime] = None


@dataclass
class ConflictLog:
    id: Optional[int] = None
    category: str = ""
    old_value: str = ""
    new_value: str = ""
    resolution: str = "recency_wins"
    detected_at: Optional[datetime] = None


@dataclass
class UserProfile:
    id: Optional[int] = None
    profile_text: str = ""
    updated_at: Optional[datetime] = None


@dataclass
class ChatThread:
    id: str = ""
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ChatMessage:
    id: Optional[int] = None
    thread_id: str = ""
    role: str = ""  # 'user' or 'assistant'
    content: str = ""
    created_at: Optional[datetime] = None
    ingested: bool = False


@dataclass
class QueryLog:
    id: Optional[int] = None
    thread_id: Optional[str] = None
    query_text: str = ""
    response_text: str = ""
    memory_context: str = ""
    retrieval_metadata: Optional[dict] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    query_time: Optional[datetime] = None
