import torch

from rnnt.AudioDataset import AudioDataset, encode_string, decode_labels
from models import *
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
  def __init__(self, model, lr):
    self.model = model
    self.lr = lr
    self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
  
  def train(self, dataset, print_interval = 20):
    train_loss = 0
    num_samples = 0
    self.model.train()
    pbar =  (dataset.loader)
    for idx, batch in enumerate(pbar):
      x,y,T,U = batch
      x = x.to(self.model.device); y = y.to(self.model.device)
      batch_size = len(x)
      num_samples += batch_size
      loss = self.model.compute_loss(x,y,T,U)
      self.optimizer.zero_grad()
      pbar.set_description("%.2f" % loss.item())
      loss.backward()
      self.optimizer.step()
      train_loss += loss.item() * batch_size
      if idx % print_interval == 0:
        self.model.eval()
        guesses = self.model.greedy_search(x,T)
        self.model.train()
        print("\n")
        for b in range(2):
          print("guess:", decode_labels(guesses[b]))
          print("truth:", decode_labels(y[b,:U[b]]))
          print("")
    train_loss /= num_samples
    return train_loss

  def test(self, dataset, print_interval=1):
    test_loss = 0
    num_samples = 0
    self.model.eval()
    pbar = tqdm(dataset.loader)
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
          x,y,T,U = batch
          x = x.to(self.model.device); y = y.to(self.model.device)
          batch_size = len(x)
          num_samples += batch_size
          loss = self.model.compute_loss(x,y,T,U)
          pbar.set_description("%.2f" % loss.item())
          test_loss += loss.item() * batch_size
          if idx % print_interval == 0:
            print("\n")
            print("input:", decode_labels(x[0,:T[0]]))
            print("guess:", decode_labels(self.model.greedy_search(x,T)[0]))
            print("truth:", decode_labels(y[0,:U[0]]))
            print("")
    test_loss /= num_samples
    return test_loss
    