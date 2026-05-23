"""Abstract LLM interface. Swap providers without touching agent code."""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a list of messages and return the assistant reply as a string."""
        ...
