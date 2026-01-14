def train_step(model, batch, optim, losses):
  model.train()
  optim.zero_grad()
  emo, sent = model(**batch['inputs'])
  loss = losses['emotion'](emo, batch['emotion']) + \
  losses['sentiment'](sent.squeeze(), batch['sentiment'])
  loss.backward()
  optim.step()
  return loss.item()
