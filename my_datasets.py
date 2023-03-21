from torch.utils.data import Dataset

class DiasumDataset(Dataset):
    def __init__(self, dataset, tokenizer=None, max_len=1024):
        self.dataset = list(map(self.__proceed_dialogue, dataset['dialogue']))  
        self.tokenizer = tokenizer
        self.max_len = max_len


    # 'Брэд: Эй, ты помнишь меня?\r\nКлаудия: Нет, не совсем.\r\nКлаудия: Я должна помнить?\r\n' -> dict {'Брэд': 1, 'Клаудия': 2}
    def __proceed_dialogue(self, dialogue):
        dialogue_dict = {}
        last = 1
        for i, line in enumerate(dialogue.split('\n')):
            if len(line.split(': ')) > 1:
                splits = line.strip().split(': ')
                for speaker, text in zip(splits[::2], splits[1::2]):
                    speaker = speaker.strip()
                    if speaker not in dialogue_dict:
                        dialogue_dict[speaker] = last
                        last += 1
        # apply dict to dialogue with replacing speakers with their numbers
        for speaker in dialogue_dict:
            dialogue = dialogue.replace(speaker + ': ', str(dialogue_dict[speaker]) + ': ')

        return dialogue.strip().lower() # .replace('\r', '\t').replace('\n', '\t')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        dialog = self.dataset[idx]
        if self.tokenizer is not None:
            dialog = self.tokenizer(dialog, max_length=self.max_len, padding='max_length', truncation=True)
        return dialog
    

class YandexDataset(Dataset):
    def __init__(self, dataset, tokenizer=None, max_len=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        elem = self.dataset[idx]
        dialog = '1: ' + elem['question'] + '\t' + '2: ' + elem['answer']
        if self.tokenizer is not None:
            dialog = self.tokenizer(dialog, max_length=self.max_len, padding='max_length', truncation=True)
        return dialog