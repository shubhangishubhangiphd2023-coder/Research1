class InfoNCELoss(nn.Module):
  def __init__(self, temp=0.07):
    super().__init__()
    self.temp = temp
  
  
  def forward(self, z1, z2):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.T / self.temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    return nn.functional.cross_entropy(logits, labels)
