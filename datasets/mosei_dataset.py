"""
CMU-MOSEI Dataset + Collate Function
----------------------------------
Supports:
- Variable-length sequences
- Text (tokenized), Audio, Vision
- Emotion (6-dim multi-label)
- Sentiment regression


Assumes preprocessed .pkl files from CMU-MultimodalSDK.
"""

import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer




class MoseiDataset(Dataset):
def __init__(self, pkl_path, text_model='roberta-base', max_text_len=128):
with open(pkl_path, 'rb') as f:
self.data = pickle.load(f)


self.tokenizer = AutoTokenizer.from_pretrained(text_model)
self.max_text_len = max_text_len


def __len__(self):
return len(self.data)


def __getitem__(self, idx):
item = self.data[idx]


# item fields expected:
# 'text' : raw string OR list of tokens
# 'audio': np.ndarray [T, D]
# 'vision': np.ndarray [T, D]
# 'emotion': np.ndarray [6]
# 'sentiment': float


text = item['text']
audio = torch.tensor(item['audio'], dtype=torch.float32)
vision = torch.tensor(item['vision'], dtype=torch.float32)


emotion = torch.tensor(item['emotion'], dtype=torch.float32)
sentiment = torch.tensor(item['sentiment'], dtype=torch.float32)


tokens = self.tokenizer(
text,
truncation=True,
padding=False,
max_length=self.max_text_len,
return_tensors='pt'
)


return {
'text': {
'input_ids': tokens['input_ids'].squeeze(0),
'attention_mask': tokens['attention_mask'].squeeze(0)
},
'audio': audio,
'vision': vision,
'emotion': emotion,
'sentiment': sentiment
}

