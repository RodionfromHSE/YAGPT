# little hack, let's add the parent directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from utils.preprocess.preprocessor import BasePreprocessor

class YandexPreprocessor(BasePreprocessor):
    def __init__(self, special_tokens: dict, **kwargs):
        super().__init__(special_tokens, **kwargs)

    def filter_one(self, sample: dict) -> bool:
        return True

    def preprocess_one(self, sample: dict) -> dict:
        sample['question'], sample['answer'] = sample['question'].lower().strip(), sample['answer'].lower().strip()
        return {'texts': [self._format_dialogue('', sample['question'], sample['answer'])]}


if __name__ == "__main__":
    # preprocess dataset
    preprocessor = YandexPreprocessor(special_tokens={'history': '<history>', 'speaker1': '<speaker1>', 'speaker2': '<speaker2>'})
    simple_sample = {'question': 'Hi', 'answer': 'Hello'}
    st = preprocessor.special_tokens
    history, speaker1, speaker2 = st['history'], st['speaker1'], st['speaker2']
    presample = preprocessor.preprocess_one(simple_sample)
    assert presample == {'texts': [f'{history} \n{speaker1} hi\n{speaker2} hello\n']}, "preprocess_one is not working" 
    print("All tests passed!")