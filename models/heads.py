class EmotionHeads(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(6)])
  
  
  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim=1)
