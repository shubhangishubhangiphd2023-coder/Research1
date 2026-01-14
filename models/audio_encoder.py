from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
def __init__(self, model_name='facebook/wav2vec2-base', out_dim=256, freeze=True):
super().__init__()
self.model = Wav2Vec2Model.from_pretrained(model_name)
if freeze:
for p in self.model.parameters(): p.requires_grad = False
self.proj = nn.Linear(self.model.config.hidden_size, out_dim)


def forward(self, x):
out = self.model(x)
return self.proj(out.last_hidden_state)
