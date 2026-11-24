from abc import ABC, abstractmethod

#class that defines the interface for a custom client
class CustomClient(ABC):
    
    @abstractmethod
    def call(self, prompt: str, system_prompt: str,
             temperature: float, max_tokens: int, **_ignored) -> str:
        pass