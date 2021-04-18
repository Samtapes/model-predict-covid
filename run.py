import torch

import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


import numpy as np



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




## MODEL
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



# CREATING MODEL
seq_length = 5
in_size = 1
hidden_size = 512
n_layers = 2

model = CovidPredictor(in_size, hidden_size, seq_length, n_layers)

model.load_state_dict(torch.load('times_series/covid_studyes/covid03/models/model1'))





# GETTING DATA TO PREDICT
day1 = float(input("DAY 1: "))
day2 = float(input("DAY 2: "))
day3 = float(input("DAY 3: "))
day4 = float(input("DAY 4: "))
day5 = float(input("DAY 5: "))


true_cases = torch.tensor([day1, day2, day3, day4, day5])



# NORMALIZING
true_cases = scaler.transform(np.expand_dims(true_cases, axis=1))


true_cases = torch.from_numpy(true_cases).float()
true_cases = true_cases.unsqueeze(0)




# MAKING PREDICTION
DAYS_TO_PREDICT = 1

def make_pred(x_data):
  model.reset_hidden_state()

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


  predict_cases = scaler.inverse_transform(
    np.expand_dims(preds, axis=0)
  ).flatten()

  return predict_cases


pred = make_pred(true_cases)


print("DAY 6: ", pred)

# 82186
# 73513
# 73174
# 85774
# 67636