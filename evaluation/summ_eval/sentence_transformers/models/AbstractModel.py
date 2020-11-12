from torch import Tensor
from typing import Union, Tuple, List, Iterable, Dict
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def get_sentence_embedding_dimension(self) -> int:
        """Returns the size of the sentence embedding"""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a text. Returns a list with tokens"""
        pass

    @abstractmethod
    def get_sentence_features(self, tokens: List[str], pad_seq_length: int):
        """This method is passed a tokenized text (tokens). pad_seq_length defines the needed length for padding."""
        pass
