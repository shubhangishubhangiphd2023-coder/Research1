# ============================

def load_config(path):
  with open(path) as f:
    return yaml.safe_load(f)

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # ---------------- DATA ----------------
  train_ds = MoseiDataset(cfg['data']['train_pkl'])
  val_ds = MoseiDataset(cfg['data']['val_pkl'])
  
  
  train_loader = DataLoader(
    train_ds,
    batch_size=cfg['training']['batch_size'],
    shuffle=True,
    collate_fn=mosei_collate_fn
  )
  
  
  # ---------------- MODEL ----------------
  text_enc = TextEncoder(out_dim=cfg['model']['dim'], freeze=cfg['model']['freeze_encoders'])
  audio_enc = AudioEncoder(out_dim=cfg['model']['dim'], freeze=cfg['model']['freeze_encoders'])
  vision_enc = VisionEncoder(out_dim=cfg['model']['dim'], freeze=cfg['model']['freeze_encoders'])
  
  
  model = MulTMultiTask(text_enc, audio_enc, vision_enc, dim=cfg['model']['dim'])
  model.to(device)
  
  
  # ---------------- OPTIM ----------------
  optim = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg['training']['lr']
  )
  
  
  losses = {
    'emotion': torch.nn.BCEWithLogitsLoss(),
    'sentiment': torch.nn.MSELoss()
  }
  
  
  # ---------------- TRAIN ----------------
  for epoch in range(cfg['training']['epochs']):
    total = 0
    for batch in train_loader:
      for k in batch['inputs']:
        batch['inputs'][k] = batch['inputs'][k].to(device)
      batch['emotion'] = batch['emotion'].to(device)
      batch['sentiment'] = batch['sentiment'].to(device)
      
      
      loss = train_step(model, batch, optim, losses)
      total += loss
    
    
    print(f"Epoch {epoch}: loss={total/len(train_loader):.4f}")




if __name__ == '__main__':
cfg = load_config('configs/multitask.yaml')
main(cfg)
