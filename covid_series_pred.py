import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader



# Importing data
df = pd.read_csv('times_series/covid_studyes/covid03/data/time_series_covid19_confirmed_global.csv')
df = df.iloc[:, 4:]



# Getting the total
daily_cases = df.sum(axis=0)

# Converting the date to date
daily_cases.index = pd.to_datetime(daily_cases.index)

# Converting accumulative to daily cases
daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)



# Spliting data
test_data_size = 14

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]



# Normalizig the data
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))




# Getting the data inputs and the targets
def sliding_windows(data, seq_length):
  xs = []
  ys = []

  for i in range(len(data) - seq_length - 1):
    x = data[i: (i+seq_length)]
    y = data[i+seq_length]
    xs.append(x)
    ys.append(y)

  return np.array(xs), np.array(ys)


seq_length = 5

X_train, y_train = sliding_windows(train_data, seq_length)
X_test, y_test = sliding_windows(test_data, seq_length)


# Converting data to tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

print(X_train.size())


X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()


# Creating dataloaders
batch_size = 50

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size, shuffle=True)




# MODEL
class CovidPredictor(nn.Module):
  def __init__(self, in_dim, hidden_dim, seq_len, num_layers=2):
    super(CovidPredictor, self).__init__()

    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.seq_len = seq_len
    self.num_layers = num_layers


    self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, dropout=0.5)
    self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=0.5)

    self.linear = nn.Linear(hidden_dim, 1)

  
  def reset_hidden_state(self):
    self.hidden = (
      torch.zeros(self.num_layers, self.seq_len, self.hidden_dim),
      torch.zeros(self.num_layers, self.seq_len, self.hidden_dim)
    )

  def forward(self, xb):
    xb = xb.view(len(xb), self.seq_len, -1)

    lstm_out, self.hidden = self.lstm(xb, self.hidden)
    lstm_out = torch.sigmoid(lstm_out)

    lstm_out, self.hidden = self.lstm1(lstm_out, self.hidden)
    lstm_out = torch.sigmoid(lstm_out)


    lstm_out = lstm_out.view(self.seq_len, len(xb), self.hidden_dim)[-1]

    pred = self.linear(lstm_out)

    return pred




# TRAINING MODEL
def fit(num_epochs, model, loss_fn, train_dl, lr, opt_fn=None):

  if opt_fn is None: opt_fn = optim.Adam
  opt = opt_fn(model.parameters(), lr=lr)


  for epoch in range(num_epochs):
    print("NEW EPOCH", epoch + 1, "/", num_epochs)
    for xb, yb in train_dl:
      model.reset_hidden_state()

      pred = model(xb)
      loss = loss_fn(pred, yb)

      loss.backward()
      opt.step()
      opt.zero_grad()
    print(loss)



in_size = 1
hidden_size = 512
n_layers = 2

model = CovidPredictor(in_size, hidden_size, seq_length, n_layers)


loss_fn = nn.MSELoss(reduction='sum')
opt_fn = optim.Adam


fit(10, model, loss_fn, train_dl, 1e-3, opt_fn)

torch.save(model.state_dict(), 'times_series/covid_studyes/covid03/models/model1')



# MAKING PREDICTION
DAYS_TO_PREDICT = 12

def make_pred(x_data, y_data):
  with torch.no_grad():
    
    test_seq = x_data[:1]
    preds = []
    for _ in range(DAYS_TO_PREDICT):
      y_test_pred = model(test_seq)
      pred = torch.flatten(y_test_pred).item()
      preds.append(pred)

      new_seq = test_seq.numpy().flatten()
      new_seq = np.append(new_seq, [pred])
      new_seq = new_seq[1:]
      test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

  true_cases = scaler.inverse_transform(
    np.expand_dims(y_data.flatten().numpy(), axis=0)
  ).flatten()

  predict_cases = scaler.inverse_transform(
    np.expand_dims(preds, axis=0)
  ).flatten()

  return true_cases, predict_cases


true_cases, predict_cases = make_pred(X_test, y_test)

print(true_cases[0:3])
print(predict_cases[0:3])