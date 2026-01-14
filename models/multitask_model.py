class MulTMultiTask(nn.Module):
  def __init__(self, text_enc, audio_enc, vision_enc, dim=256):
    super().__init__()
    self.text = text_enc
    self.audio = audio_enc
    self.vision = vision_enc
    self.fusion = MulTFusion(dim)
    self.emotion = EmotionHeads(dim * 3)
    self.sentiment = nn.Linear(dim * 3, 1)
  
  
  def forward(self, text, audio, vision):
    t = self.text(**text)
    a = self.audio(audio)
    v = self.vision(vision)
    fused = self.fusion(t, a, v)
    return self.emotion(fused), self.sentiment(fused)
