from abc import ABC, abstractmethod
from datasets import Dataset
import numpy as np


class BasePreprocessor(ABC):
    def __init__(self, special_tokens: dict, **kwargs):
        """	
        :@ special_tokens: dict, special tokens for dialogues (history, speaker1, speaker2)
        :@ map_params: dict, parameters for mapping

        Note: base preprocessor always will explode the dataset to the texts column
        """
        # check if special_tokens has all the required keys
        assert all(key in special_tokens for key in ["history", "speaker1", "speaker2"]), "special_tokens must have keys: history, speaker1, speaker2"
        
        self.special_tokens = special_tokens
        self.params = kwargs

    @abstractmethod
    def filter_one(sample: dict) -> bool:
        """	
        :@ sample: dict, a sample from the dataset
        :@ return: bool, True if the sample is valid, False otherwise
        """
        pass

    @abstractmethod
    def preprocess_one(sample: dict) -> dict:
        """	
        :@ sample: dict, a sample from the dataset
        :@ return: dict, a preprocessed sample
        Here you can do some preprocessing on the sample
        Here you should get final dataset format
        """
        pass

    
    def preprocess(self, dataset: Dataset) -> Dataset:
        """	
        :@ dataset: Dataset, a dataset
        :@ columns: list, columns to be left after preprocessing
        :@ return: Dataset, a preprocessed dataset. With only one column: text
        """
        dataset = dataset.filter(self.filter_one, **self.params)
        dataset = dataset.map(self.preprocess_one, **self.params)
        # explode dataset to the texts column
        dataset = dataset.map(lambda x: {'text': sum(x['texts'], [])}, batched=True, **self.params, remove_columns=dataset.column_names)

        return dataset
    
    
    def _format_dialogue(self, history, prompt, answer, is_speaker1=True, is_speaker2=True):
        """
        :@ history: str, history of dialogues
        :@ prompt: str, prompt
        :@ answer: str, answer
        :@ is_speaker1: bool, True if first speaker is the first one, False otherwise
        :@ is_speaker2: bool, True if second speaker is the second one, False otherwise
        """
        speaker1 = self.special_tokens['speaker1'] if is_speaker1 else self.special_tokens['speaker2']
        speaker2 = self.special_tokens['speaker2'] if is_speaker2 else self.special_tokens['speaker1']
        return f"{self.special_tokens['history']} {history}\n{speaker1} {prompt}\n{speaker2} {answer}\n"
        
