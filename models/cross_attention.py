class CrossModalAttention(nn.Module):
def __init__(self, dim, heads=4):
super().__init__()
self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
self.norm = nn.LayerNorm(dim)


def forward(self, q, kv):
out, _ = self.attn(q, kv, kv)
return self.norm(q + out)
