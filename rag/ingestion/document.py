from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Document instance into a dictionary format."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Creates a Document instance from a dictionary."""
        return cls(
            text=data.get("text", ""),
            metadata=data.get("metadata", {})
        )