from transformers import ViTModel


class VisionEncoder(nn.Module):
  def __init__(self, model_name='google/vit-base-patch16-224', out_dim=256, freeze=True):
    super().__init__()
    self.model = ViTModel.from_pretrained(model_name)
    if freeze:
      for p in self.model.parameters(): p.requires_grad = False
    self.proj = nn.Linear(self.model.config.hidden_size, out_dim)
  
  
  def forward(self, pixel_values):
    out = self.model(pixel_values=pixel_values)
    return self.proj(out.last_hidden_state)
