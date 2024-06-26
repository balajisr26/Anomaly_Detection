import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

df = pd.read_csv('T20_Stats_Career.txt',sep='|')

# Feature Engineering
df['Maiden_PCT'] = df['Maidens'] / df['Overs'] * 100
df['Wickets_Per_over'] = df['Wickets'] / df['Overs']

# Remove Players with less than 15 Innings
df =df[df['Innings'] >=15]

# Create Train and Test Dataset
test = df[df['Country'] == 'India']
train = df[df['Country'] != 'India']

features = ['Average','Economy','StrikeRate','FourWickets','FiveWickets','Maiden_PCT','Wickets_Per_over']
X_train = train[features]
X_test = test[features]

# Standarize Data
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Create Tensor Dataset and Dataloders
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(13)
x_train_tensor = torch.as_tensor(X_train).float().to(device)
y_train_tensor = torch.as_tensor(X_train).float().to(device)

x_test_tensor = torch.as_tensor(X_test).float().to(device)
y_test_tensor = torch.as_tensor(X_test).float().to(device)

train_dataset = TensorDataset(x_train_tensor,y_train_tensor)
test_dataset = TensorDataset(x_test_tensor,y_test_tensor)

train_loader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=16)



# AutoEncoder Architecture

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('Hidden1',nn.Linear(7,4))
        self.encoder.add_module('Relu1',nn.ReLU())
        self.encoder.add_module('Hidden2',nn.Linear(4,2))
        
        self.decoder = nn.Sequential()
        self.decoder.add_module('Hidden3',nn.Linear(2,4))
        self.decoder.add_module('Relu2',nn.ReLU())
        self.decoder.add_module('Hidden4',nn.Linear(4,7))

    def forward(self,x):
        encoder = self.encoder(x)
        return self.decoder(encoder)
    
# Predict Method
def predict(model,x):

        model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat = model(x_tensor.to(device))
        model.train()

        return y_hat.detach().cpu().numpy()

# Plot Losses
def plot_losses(train_losses,test_losses):
        fig = plt.figure(figsize=(10,4))
        plt.plot(train_losses,label='training_loss',c='b')
        #plt.plot(self.val_losses,label='val loss',c='r')
        if test_loader:
            plt.plot(test_losses,label='test loss',c='r')
        #plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig


    
# Model Loss and Optimizer
lr = 0.001
torch.manual_seed(21)
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(),lr = lr)
loss_fn =nn.MSELoss()

num_epochs=250
train_loss=[]
test_loss=[]
seed=42
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
for epoch in range(num_epochs):
    mini_batch_train_loss=[]
    mini_batch_test_loss=[]
    for train_batch,y_train in train_loader:
        train_batch =train_batch.to(device)
        model.train()
        yhat = model(train_batch)
        loss = loss_fn(yhat,y_train)
        mini_batch_train_loss.append(loss.cpu().detach().numpy())
  
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
  
    train_epoch_loss = np.mean(mini_batch_train_loss)
    train_loss.append(train_epoch_loss)

    with torch.no_grad():
        for test_batch,y_test in test_loader:
            test_batch = test_batch.to(device)
            model.eval()
            yhat = model(test_batch)
            loss = loss_fn(yhat,y_test)
            mini_batch_test_loss.append(loss.cpu().detach().numpy())
        test_epoch_loss = np.mean(mini_batch_test_loss)
        test_loss.append(test_epoch_loss)

fig = plot_losses(train_loss,test_loss)
fig.savefig('Train_Test_Loss.png')

# Predict Train Dataset and get error
train_pred = predict(model,X_train)
print(train_pred.shape)
error = np.mean(np.power(X_train - train_pred,2),axis=1)
print(error.shape)
train['error'] = error
mean_error = np.mean(train['error'])
std_error =np.std(train['error'])
train['zscore'] = (train['error'] - mean_error) / std_error
train = train.sort_values(by='error').reset_index()
train.to_csv('Train_Output.txt',sep="|",index=None)

fig = plt.figure(figsize=(10,4))
plt.title('Distribution of MSE in Train Dataset')
train['error'].plot(kind='line')
plt.ylabel('MSE')
plt.show()
fig.savefig('Train_MSE.png')

# Predict Test Dataset and get error
test_pred = predict(model,X_test)
test_error = np.mean(np.power(X_test - test_pred,2),axis=1)
test['error'] = test_error
test['zscore'] = (test['error'] - mean_error) / std_error
test = test.sort_values(by='error').reset_index()
test.to_csv('Test_Output.txt',sep="|",index=None)

fig = plt.figure(figsize=(10,4))
plt.title('Distribution of MSE in Test Dataset')
test['error'].plot(kind='line')
plt.ylabel('MSE')
plt.show()
fig.savefig('Test_MSE.png')

