import torch
import string
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
  def __init__(self, audio_files, transcripts, accent, sample_rate=16000, max_length=128, batch_size=32):
    self.audio_files = audio_files
    self.transcripts = transcripts
    self.accents = accent
    self.sample_rate = sample_rate
    self.max_length = max_length

    # Initialize DataLoader
    self.loader = DataLoader(self, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=self.collate)

  def __len__(self):
    return len(self.audio_files)

  def __getitem__(self, idx):
    audio_file = self.audio_files[idx]
    transcript = self.transcripts[idx]
    accent = self.accents[idx]

    waveform, sample_rate = librosa.load('Data/clips/' + audio_file, sr=self.sample_rate)

    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=128)
    mfccs = np.mean(mfccs.T, axis=0)  # Take the mean along the time axis

    mfccs = torch.tensor(mfccs, dtype=torch.float32)
    

    return mfccs, transcript

  def collate(self, batch):
    """
    Collate function for DataLoader.
    """
    batch_size = len(batch)
    mfccs_batch, transcript_batch = zip(*batch)
    
    T = [len(x) for x in mfccs_batch]
    U = [len(y) for y in transcript_batch]
    
    U_max = max(U)
    y = []
    for i in range(batch_size):
      enc_trans = encode_string(transcript_batch[i])
      enc_trans += [0] * (U_max - len(enc_trans))
      y.append(torch.tensor(enc_trans))  
    
    mfccs_batch = torch.stack(mfccs_batch)
    y = torch.stack(y)
    T = torch.tensor(T)
    U = torch.tensor(U)
    
    return mfccs_batch, transcript_batch, T, U


def encode_string(s):
  for c in s:
    if c not in string.printable:
      print(s)
  return [string.printable.index(c) + 1 for c in s]

def decode_labels(l):
  return "".join([string.printable[c - 1] for c in l])

df = pd.read_csv('Data/validated_top10.csv', index_col=False)
df.head()
end = round(0.9 * df.shape[0])
train = df[:end]
test = df[end:]

train_set = AudioDataset(train['path'], train['sentence'], train['accents'])
test_set = AudioDataset(test['path'], test['sentence'], test['accents'])