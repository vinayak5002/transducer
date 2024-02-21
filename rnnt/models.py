import torch
import string
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
import unidecode


from torchaudio.transforms import RNNTLoss
transducer_loss = RNNTLoss(0)


NULL_INDEX = 0

encoder_dim = 1024
predictor_dim = 1024
joiner_dim = 1024

class Encoder(torch.nn.Module):
  def __init__(self, num_inputs):
    super(Encoder, self).__init__()
    self.embed = torch.nn.Embedding(num_inputs, encoder_dim)
    self.rnn = torch.nn.GRU(input_size=encoder_dim, hidden_size=encoder_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
    self.linear = torch.nn.Linear(encoder_dim*2, joiner_dim)

  def forward(self, x):
    out = x
    out = self.embed(out)
    out = self.rnn(out)[0]
    out = self.linear(out)
    return out
  
class Predictor(torch.nn.Module):
  def __init__(self, num_outputs):
    super(Predictor, self).__init__()
    self.embed = torch.nn.Embedding(num_outputs, predictor_dim)
    self.rnn = torch.nn.GRUCell(input_size=predictor_dim, hidden_size=predictor_dim)
    self.linear = torch.nn.Linear(predictor_dim, joiner_dim)
    
    self.initial_state = torch.nn.Parameter(torch.randn(predictor_dim))
    self.start_symbol = NULL_INDEX # In the original paper, a vector of 0s is used; just using the null index instead is easier when using an Embedding layer.

  def forward_one_step(self, input, previous_state):
    embedding = self.embed(input)
    state = self.rnn.forward(embedding, previous_state)
    out = self.linear(state)
    return out, state

  def forward(self, y):
    batch_size = y.shape[0]
    U = y.shape[1]
    outs = []
    state = torch.stack([self.initial_state] * batch_size).to(y.device)
    for u in range(U+1): # need U+1 to get null output for final timestep 
      if u == 0:
        decoder_input = torch.tensor([self.start_symbol] * batch_size, device=y.device)
      else:
        decoder_input = y[:,u-1]
      out, state = self.forward_one_step(decoder_input, state)
      outs.append(out)
    out = torch.stack(outs, dim=1)
    return out

class Joiner(torch.nn.Module):
  def __init__(self, num_outputs):
    super(Joiner, self).__init__()
    self.linear = torch.nn.Linear(joiner_dim, num_outputs)

  def forward(self, encoder_out, predictor_out):
    out = encoder_out + predictor_out
    out = torch.nn.functional.relu(out)
    out = self.linear(out)
    return out

class Transducer(torch.nn.Module):
	def __init__(self, num_inputs, num_outputs):
		super(Transducer, self).__init__()
		self.encoder = Encoder(num_inputs)
		self.predictor = Predictor(num_outputs)
		self.joiner = Joiner(num_outputs)

		if torch.cuda.is_available(): self.device = "cuda:0"
		else: self.device = "cpu"
		self.to(self.device)

	def compute_single_alignment_prob(self, encoder_out, predictor_out, T, U, z, y):
		"""
		Computes the probability of one alignment, z.
		"""
		t = 0; u = 0
		t_u_indices = []
		y_expanded = []
		for step in z:
			t_u_indices.append((t,u))
			if step == 0: # right (null)
				y_expanded.append(NULL_INDEX)
				t += 1
			if step == 1: # down (label)
				y_expanded.append(y[u])
				u += 1
		t_u_indices.append((T-1,U))
		y_expanded.append(NULL_INDEX)

		t_indices = [t for (t,u) in t_u_indices]
		u_indices = [u for (t,u) in t_u_indices]
		encoder_out_expanded = encoder_out[t_indices]
		predictor_out_expanded = predictor_out[u_indices]
		joiner_out = self.joiner.forward(encoder_out_expanded, predictor_out_expanded).log_softmax(1)
		logprob = -torch.nn.functional.nll_loss(input=joiner_out, target=torch.tensor(y_expanded).long().to(self.device), reduction="sum")
		return logprob
  
	def greedy_search(self, x, T):
		y_batch = []
		B = len(x)
		encoder_out = self.encoder.forward(x)
		U_max = 200
		for b in range(B):
			t = 0; u = 0; y = [self.predictor.start_symbol]; predictor_state = self.predictor.initial_state.unsqueeze(0)
			while t < T[b] and u < U_max:
				predictor_input = torch.tensor([ y[-1] ], device = x.device)
				g_u, predictor_state = self.predictor.forward_one_step(predictor_input, predictor_state)
				f_t = encoder_out[b, t]
				h_t_u = self.joiner.forward(f_t, g_u)
				argmax = h_t_u.max(-1)[1].item()
				if argmax == NULL_INDEX:
					t += 1
				else: # argmax == a label
					u += 1
					y.append(argmax)
			y_batch.append(y[1:]) # remove start symbol
		return y_batch
  
	def compute_loss(self, x, y, T, U):
		encoder_out = self.encoder.forward(x)
		predictor_out = self.predictor.forward(y)
		joiner_out = self.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3)
		#loss = -self.compute_forward_prob(joiner_out, T, U, y).mean()
		T = T.to(joiner_out.device)
		U = U.to(joiner_out.device)
		loss = transducer_loss(joiner_out, y, T, U) #, blank_index=NULL_INDEX, reduction="mean")
		return loss

def encode_string(s):
  for c in s:
    if c not in string.printable:
      print(s)
  return [string.printable.index(c) + 1 for c in s]

def decode_labels(l):
  return "".join([string.printable[c - 1] for c in l])

class Trainer:
  def __init__(self, model, lr):
    self.model = model
    self.lr = lr
    self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
  
  def train(self, dataset, print_interval = 20):
    train_loss = 0
    num_samples = 0
    self.model.train()
    pbar = tqdm(dataset.loader)
    for idx, batch in enumerate(pbar):
      x,y,T,U = batch
      x = x[0].to(self.model.device)
      y = y[0].to(self.model.device)
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
          print("input:", decode_labels(x[b,:T[b]]))
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
    