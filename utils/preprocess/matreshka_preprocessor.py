# little hack, let's add the parent directory to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from utils.preprocess.preprocessor import BasePreprocessor

class MatreshkaPreprocessor(BasePreprocessor):
    def __init__(self, special_tokens: dict, **kwargs):
        super().__init__(special_tokens, **kwargs)

    def filter_one(self, sample: dict) -> bool:
        """
        In matreshka was found one (one!) none dialogue, so we need to filter it
        """
        return sample['dialog'] is not None and sample['role'] is not None
    
    def __proceed_dialog(self, roles, lines):
        """
        :@ roles: list of roles (user, bot)
        :@ lines: list of lines

        Convert dialogue to a list of dialogues with history and speaker tokens
        """
        assert len(roles) == len(lines), "roles and lines must have the same length"
        lines = [line.lower().strip() for line in lines]
        replics = list(zip(roles, lines) )
        res = []
        history = ""

        prev_role, prev_line = replics[0]
        for replic in replics[1:]:
            role, line = replic
            # print(role, line)
            is_speaker1 = prev_role == 'user'
            # print("Is speaker1", is_speaker1)
            is_speaker2 = role == 'bot'
            res.append(self._format_dialogue(history, prev_line, line, is_speaker1, is_speaker2))
            history += f"{self.special_tokens['speaker1' if is_speaker1 else 'speaker2']} {prev_line}\t"
            prev_role, prev_line = role, line
        
        return res

    def preprocess_one(self, sample: dict) -> dict:
        roles, lines = sample['role'], sample['dialog']
        return {'texts': self.__proceed_dialog(roles, lines)}


if __name__ == "__main__":
    # preprocess dataset
    preprocessor = MatreshkaPreprocessor(special_tokens={'history': '<history>', 'speaker1': '<speaker1>', 'speaker2': '<speaker2>'})
    simple_sample = {
        'role': ['user', 'bot', 'user', 'bot'],
        'dialog': ['Hi', 'Hello', 'How are you?', 'Fine, thanks.']
    }
    st = preprocessor.special_tokens
    history, speaker1, speaker2 = st['history'], st['speaker1'], st['speaker2']
    presample = preprocessor.preprocess_one(simple_sample)
    # print(presample)
    assert presample['texts'] == [
        f'{history} \n{speaker1} hi\n{speaker2} hello\n',
        f'{history} {speaker1} hi\t\n{speaker2} hello\n{speaker1} how are you?\n',
        f'{history} {speaker1} hi\t{speaker2} hello\t\n{speaker1} how are you?\n{speaker2} fine, thanks.\n'
    ] 
    
    print("All tests passed!")