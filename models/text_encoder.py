import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
def __init__(self, model_name='roberta-base', out_dim=256, freeze=True):
super().__init__()
self.model = AutoModel.from_pretrained(model_name)
if freeze:
for p in self.model.parameters(): p.requires_grad = False
self.proj = nn.Linear(self.model.config.hidden_size, out_dim)


def forward(self, input_ids, attention_mask):
out = self.model(input_ids=input_ids, attention_mask=attention_mask)
return self.proj(out.last_hidden_state)
