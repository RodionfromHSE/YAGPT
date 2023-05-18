# little hack, let's add the parent directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from utils.preprocess.preprocessor import BasePreprocessor
from datasets import Dataset, load_dataset

class DiasumPreprocessor(BasePreprocessor):
    def __init__(self, special_tokens: dict, **kwargs):
        super().__init__(special_tokens, **kwargs)

    def filter_one(self, sample: dict) -> bool:
        """Save only if dialogue has two speakers"""
        dialogue = sample['dialogue']
        splits = [line.strip().split(': ') for line in dialogue.split('\n') if line.strip()]
        splits = list(filter(lambda x: len(x) == 2, splits)) # remove lines without ': '
        # speaker: text
        speakers = set([s[0] for s in splits])
        return len(speakers) == 2
    
    def preprocess_one(self, sample: dict) -> dict:
        dialogue = sample['dialogue']
        texts = self.__proceed_dialogue(dialogue)
        return {'texts': texts}
    
    def __proceed_dialogue(self, dialogue, sep=': '):
        """
        :@ dialogue: str, dialogue
        :@ sep: str, separator between speaker and text

        Convert dialogue to a list of dialogues with history and speaker tokens
        """
        # remove empty lines
        lines = list(filter(lambda x: x.strip(), dialogue.split('\n'))) 
        # split by first ': ' to get speaker and text. Lower and strip text
        splits = list(map(lambda x: x.lower().strip().split(sep, 1), lines)) 
        splits = list(filter(lambda x: len(x) == 2, splits)) # remove lines without ': '
        
        res = [] # list of dialogues
        history = '' # history of dialogues
        prev_speaker, prev_text = splits[0] # first speaker and text
        first_speaker = prev_speaker
        
        for speaker, text in splits[1:]:
            speaker, text = speaker.strip(), text.strip()
            is_speaker1 = prev_speaker == first_speaker
            is_speaker2 = speaker != first_speaker
            res.append(self._format_dialogue(history, prev_text, text, is_speaker1, is_speaker2))
            history += f'{self.special_tokens["speaker1"] if is_speaker1 else self.special_tokens["speaker2"]} {prev_text}\t'
            prev_speaker, prev_text = speaker, text
        return res


if __name__ == "__main__":
    # preprocess dataset
    preprocessor = DiasumPreprocessor(special_tokens={'history': '<history>', 'speaker1': '<speaker1>', 'speaker2': '<speaker2>'})
    dialogues = [
        {
            'dialogue': "a: hi\na: hi\na: hi\n"
        },
        {
            'dialogue': "a: hi\nb: hi"
        },
        {
            'dialogue': "a: hi\nb: hi\nc: hi"
        }
    ]
    assert [preprocessor.filter_one(d) for d in dialogues] == [False, True, False], "filter_one is not working"

    simple_dialogue = """
    A: Hi
    B: Hi
    A: How are you?
    """
    st = preprocessor.special_tokens
    history, speaker1, speaker2 = st['history'], st['speaker1'], st['speaker2']
    assert preprocessor.preprocess_one({'dialogue': simple_dialogue}) == {
        'texts': [
            f'{history} \n{speaker1} hi\n{speaker2} hi\n', 
            f'{history} {speaker1} hi\t\n{speaker2} hi\n<speaker1> how are you?\n'
        ]
    }, "preprocess_one is not working"
    