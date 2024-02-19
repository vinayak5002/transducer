import torch
import string
from torch.utils.data import Dataset
import pandas as pd
import librosa
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

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

    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=128)
    mfccs = np.mean(mfccs.T, axis=0)  # Take the mean along the time axis

    # # Pad or truncate MFCC features to a fixed length
    # if len(mfccs) < self.max_length:
    #     mfccs = np.pad(mfccs, (0, self.max_length - len(mfccs)))
    # else:
    #     mfccs = mfccs[:self.max_length]

    # Convert MFCCs to PyTorch tensor
    mfccs = torch.tensor(mfccs, dtype=torch.float32)
    
    encoded_transcript = encode_string(transcript)

    return mfccs, encoded_transcript, accent

  def collate(self, batch):
    """
    Collate function for DataLoader.
    """
    # Separate inputs, transcripts, and accents
    mfccs_batch, transcript_batch, accent_batch = zip(*batch)
    # Stack MFCCs into a batch tensor
    mfccs_batch = torch.stack(mfccs_batch)
    
    # Pad transcripts and accents
    transcript_batch = pad_sequence(transcript_batch, batch_first=True)
    
    return mfccs_batch, transcript_batch, accent_batch


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