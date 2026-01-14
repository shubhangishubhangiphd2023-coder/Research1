def mosei_collate_fn(batch):
"""
Pads variable-length sequences for MulT
"""


  # -------- TEXT --------
  input_ids = [b['text']['input_ids'] for b in batch]
  attention_mask = [b['text']['attention_mask'] for b in batch]
  
  
  input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
  attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
  
  
  # -------- AUDIO / VISION --------
  audio = pad_sequence([b['audio'] for b in batch], batch_first=True)
  vision = pad_sequence([b['vision'] for b in batch], batch_first=True)
  
  
  # -------- LABELS --------
  emotion = torch.stack([b['emotion'] for b in batch])
  sentiment = torch.stack([b['sentiment'] for b in batch])
  
  
  return {
    'inputs': {
      'text': {
        'input_ids': input_ids,
        'attention_mask': attention_mask
      },
      'audio': audio,
      'vision': vision
    },
    'emotion': emotion,
    'sentiment': sentiment
  }
