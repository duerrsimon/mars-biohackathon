import numpy as np
from bio_embeddings.embed import  ProtTransBertBFDEmbedder, UniRepEmbedder

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr, spearmanr
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

torch.manual_seed(5)

single = pd.read_csv('data/single_muts_train.csv', index_col=0).dropna()
single_test = pd.read_csv('data/single_muts_test.csv', index_col=0).dropna()
multiple = pd.read_csv('data/multiple_muts_train.csv', index_col=0).dropna()
multiple_test = pd.read_csv('data/multiple_muts_test.csv', index_col=0).dropna()

# train on single and multi mutant data
train = pd.concat([single, multiple])

train = train.reset_index()
test = multiple_test.reset_index()

class SequenceDatasetProtBert(Dataset):
    def __init__(self, data):
        self.df = data
        self.sequences = list(self.df.sequence)
        self.label = self.df.stabilityscore.astype('float32')
        self.embedder = ProtTransBertBFDEmbedder()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        embedding = np.asarray(self.embedder.embed(self.sequences[idx]))

        return  torch.from_numpy(embedding), self.label[idx]


class Encoder(nn.Module):
    """ The encoder part of the VAE."""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(43, 20,8)
        self.conv2 = nn.Conv1d(20, 10,8)          
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.dropout = nn.Dropout(0.25)
        self.pred1 = nn.Linear(latent_dim,10) # this part is the stability predictions
        self.pred2 = nn.Linear(10,1)
        self.training = True
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        x = x.flatten(start_dim=1)
        h_       = torch.relu(self.FC_input(x))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q")
        var      = torch.exp(0.5*log_var)              # takes exponential function
        z        = self.reparameterization(mean, var)
        stability1 = torch.relu(self.pred1(z)) # torch.relu
        predicted_stability = self.pred2(self.dropout(stability1))
        return z, mean, log_var, predicted_stability
    

    
    def reparameterization(self, mean, var,):
        epsilon = torch.rand_like(var).to(device)        # sampling epsilon
        
        z = mean + var*epsilon                          # reparameterization trick
        
        return z

class Decoder(nn.Module):
    """This is the decoder part"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.conv1T = nn.ConvTranspose1d(10, 20,8)
        self.conv2T = nn.ConvTranspose1d(20, 43,8)
        
    def forward(self, x):
        h     = torch.relu(self.FC_hidden(x))
        x = torch.relu(self.FC_output(h))
        x = self.conv1T(x.view(x.shape[0], 10, 1010))
        x_hat = torch.sigmoid(self.conv2T(x))
        return x_hat

class Model(nn.Module):
    """Putting Decoder and Encoder together"""
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        z, mean, log_var, stability = self.Encoder(x)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var, stability

# options
x_dim = 10100
hidden_dim = 200
latent_dim = 80
lr = 1e-3

epochs = 10
cuda = True
device = torch.device("cuda" if cuda else "cpu")

batch_size = 50 
# seperate loaders for different training sets
dataset_train = SequenceDatasetProtBert(train)
train_loader = DataLoader(dataset_train, batch_size=batch_size,  shuffle=True, num_workers=0)
dataset_test = SequenceDatasetProtBert(test)
batch_size_test = len(dataset_test)
test_loader = DataLoader(dataset_test, batch_size=batch_size_test,  shuffle=False, num_workers=0)



encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder)
model = nn.DataParallel(model)
model.to(device)

# set up the loss function 
mse_loss = nn.MSELoss()
def loss_function(x, x_hat, mean, log_var, stability,label):
    # for the VAE
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    # for the latent space regression, could try SmoothL1
    MSE = mse_loss(stability,label)
    return reproduction_loss + KLD + MSE


optimizer = Adam(model.parameters(), lr=lr)

# training loop

for epoch in range(epochs):
    overall_loss = 0
    model.train() # training loop
    for batch_idx, (x, label) in enumerate(train_loader):
        #x = x.view(batch_size, x_dim)
        x,label = x.to(device),label.to(device)
        x=x.float()
        optimizer.zero_grad()
        x_hat, mean, log_var, stability = model(x)
        loss = loss_function(x, x_hat, mean, log_var,stability,torch.unsqueeze(label, 1))
  
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0
    for batch_idx_test, (x, label) in enumerate(test_loader):
      with torch.no_grad():
        x,label = x.float().to(device),label.to(device)
        x_hat, mean, log_var, stability = model(x)
        tloss = loss_function(x, x_hat, mean, log_var,stability,torch.unsqueeze(label, 1))
        test_loss += tloss.item()


        
    print("\tEpoch", epoch + 1, "\t Loss: ", overall_loss / (batch_idx*batch_size),"\t Test Loss: ", test_loss / batch_size_test) 

torch.save(model.state_dict(), 'modelweights/VAE_ProtBert_singlemultitrain_multipredict.pth')

topologies = test.name.str.split('_',n=1, expand=True)[0]
model.eval()

label = label.cpu().detach().numpy().flatten()
stability_predicted = stability.cpu().detach().numpy().flatten()

Rp=pearsonr(label, stability_predicted )
print(f'Pearson {Rp[0]:.2f} p={Rp[1]:.4f}')
Sp=spearmanr(label, stability_predicted )
print(f'Spearman {Sp[0]:.2f} p={Sp[1]:.4f}')


fig = make_subplots(rows=1, cols=1)

results = pd. concat([topologies,pd.Series(label),pd.Series(stability_predicted) ], axis=1)
results.columns=['topo','exp','pred']
for top,g in results.groupby('topo'):
  fig.add_trace(go.Scatter(x=g.exp, y=g.pred,mode='markers',
                           name=top,
                    hovertemplate=
         "<b>Topology: %{text}</b><br><br>" +
         "Experimental: %{x:.2f}<br>" +
         "Predicted: %{y:.2f}<br>" +
         "<extra></extra>",   text=g.topo))

  

fig.update_xaxes(title_text='Experimental stability score')
fig.update_yaxes(title_text='Predicted stability score')

reg = LinearRegression().fit(np.vstack(label), stability_predicted)
fit = reg.predict(np.vstack(label))
fig.add_trace(go.Scatter(name=f'Pearson {Rp[0]:.2f} p={Rp[1]:e} <br> Spearman {Sp[0]:.2f} p={Sp[1]:e}',
                         marker=dict(color='black'),  
                         x=label, y=fit, 
                         mode='lines', showlegend=True, hoverinfo='skip'), row=1, col=1)

fig.update_layout(
    height=600,
    width=700,
    showlegend=True,
    template='simple_white',
    title_text='Multi mutant Test set predictions with ProtBertEmbedding'
)
fig.write_html('plots/embeddings_protbert_multi.html')
