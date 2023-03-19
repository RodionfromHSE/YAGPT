from torch.utils.data import Dataset

class DialogDataset(Dataset):
    def __init__(self, df, conf, tokenizer=None, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dialogs = []
        self.conf = conf
        self.__init_dialogs(df)

    @staticmethod
    def _fix_dialog(dialog):
        return dialog.replace('<span class=participant_2>', '||').replace('</span><br />', ' ').replace('<span class=participant_1>', '||').replace('<br />', ' ')
    
    def _fix_persona(self, persona, label):
        return persona.replace(f'<span class=participant_{label}>', self.conf[f'user{label}']).replace('<br />',' ').replace('</span>','')
        
    def __init_dialogs(self, df):
        dialogs = []
        for person1, person2, dialog in zip(df.persona_1_profile, df.persona_2_profile, df.dialogue):
            dilogue = []
            dilogue1 = []

            person1 = self._fix_persona(person1, 1)
            person2 = self._fix_persona(person2, 2)
            dialog = DialogDataset._fix_dialog(dialog).split('||')

            dilogue += [person1]
            dilogue += dialog

            dilogue1 += [person2]
            dilogue1 += dialog

            dialogs += [dilogue]
            dialogs += [dilogue1]
        self.dialogs = dialogs

    def __len__(self):
        return len(self.dialogs)
    
    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        if self.tokenizer is not None:
            dialog = self.tokenizer(dialog, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return dialog
    

class YandexDataset(Dataset):
    def __init__(self, dataset, conf, tokenizer=None, max_len=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.conf = conf

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        elem = self.dataset[idx]
        dialog = self.conf['user1'] + elem['question'] + '\t' + self.conf['user2'] + elem['answer']
        if self.tokenizer is not None:
            dialog = self.tokenizer(dialog, max_length=self.max_len, padding='max_length', truncation=True)
        return dialog