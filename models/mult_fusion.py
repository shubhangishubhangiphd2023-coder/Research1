class MulTFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.t_a = CrossModalAttention(dim)
    self.t_v = CrossModalAttention(dim)
    self.a_t = CrossModalAttention(dim)
    self.a_v = CrossModalAttention(dim)
    self.v_t = CrossModalAttention(dim)
    self.v_a = CrossModalAttention(dim)
  
  
  def forward(self, t, a, v):
    t = self.t_a(t, a) + self.t_v(t, v)
    a = self.a_t(a, t) + self.a_v(a, v)
    v = self.v_t(v, t) + self.v_a(v, a)
    return torch.cat([t.mean(1), a.mean(1), v.mean(1)], dim=1)
