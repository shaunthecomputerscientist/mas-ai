"""
Custom Document class to replace LangChain's Document schema.
This is a drop-in replacement that maintains API compatibility.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    A document object that represents a piece of text with optional metadata.
    
    This is a drop-in replacement for langchain.schema.Document that maintains
    full API compatibility while removing the LangChain dependency.
    
    Attributes:
        page_content (str): The main text content of the document.
        metadata (Dict[str, Any]): Optional metadata associated with the document.
    
    Example:
        >>> doc = Document(page_content="Hello world", metadata={"source": "test"})
        >>> print(doc.page_content)
        Hello world
        >>> print(doc.metadata)
        {'source': 'test'}
    """
    
    page_content: str = Field(
        ...,
        description="The main text content of the document"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the document"
    )
    
    class Config:
        """Pydantic configuration."""
        # Allow arbitrary types for metadata values
        arbitrary_types_allowed = True
        # Enable JSON schema generation
        json_schema_extra = {
            "example": {
                "page_content": "This is a sample document.",
                "metadata": {"source": "example.txt", "page": 1}
            }
        }
    
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize a Document.
        
        Args:
            page_content (str): The main text content.
            metadata (Dict[str, Any], optional): Metadata dictionary. Defaults to {}.
            **kwargs: Additional keyword arguments (for compatibility).
        """
        if metadata is None:
            metadata = {}
        super().__init__(page_content=page_content, metadata=metadata, **kwargs)
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"
    
    def __repr__(self) -> str:
        """Detailed representation of the document."""
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the document.
        """
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create a Document from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary with 'page_content' and optional 'metadata'.
        
        Returns:
            Document: A new Document instance.
        """
        return cls(
            page_content=data.get("page_content", ""),
            metadata=data.get("metadata", {})
        )


# For backward compatibility with LangChain imports
__all__ = ["Document"]

