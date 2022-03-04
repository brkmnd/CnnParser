import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

models_path = "models/"

model_dict = {
          "m1": {
              "lstm_dim":1024
            , "n_layers":1
            , "emb_dim":128
            , "model_name":"cnn_model1"
            , "batch_size":1
            , "dropout":0
            , "bi-directional":False
            }
        , "m2": {
              "lstm_dim":256
            , "n_layers":2
            , "emb_dim":128
            , "model_name":"cnn_model2"
            , "batch_size":1
            , "dropout":0.2
            , "bi-directional":False
            }
        , "m3": {
              "lstm_dim":100
            , "n_layers":2
            , "emb_dim":128
            , "model_name":"cnn_model3"
            , "batch_size":1
            , "dropout":0.1
            , "bi-directional":True
            }
        # above models are absolete
        , "m4": { # did not include '=' token
              "lstm_dim":256
            , "n_layers":2
            , "emb_dim":128
            , "model_name":"cnn_model4"
            , "batch_size":1
            , "dropout":0.2
            , "bi-directional":False
            }
        , "m5": {
              "lstm_dim":256
            , "n_layers":2
            , "emb_dim":128
            , "model_name":"cnn_model5"
            , "batch_size":1
            , "dropout":0.2
            , "bi-directional":False
            }
        }

class CnnLstm(nn.Module):
    def __init__(self,vocab_size,model_data):
        super(CnnLstm,self).__init__()

        emb_dim = model_data["emb_dim"]
        lstm_dim = model_data["lstm_dim"]
        n_layers = model_data["n_layers"]
        dropout_p = model_data["dropout"]
        bi_dir = model_data["bi-directional"]
        model_name = model_data["model_name"]

        self.n_layers = n_layers
        self.lstm_dim = lstm_dim
        self.bi_dir = bi_dir
        self.model_name = model_name

        #layers
        self.emb = nn.Embedding(vocab_size,emb_dim)
        self.lstm = nn.LSTM(emb_dim,lstm_dim,n_layers,dropout=dropout_p,bidirectional=bi_dir)
        self.lin_trans = nn.Linear(lstm_dim,vocab_size)

    def forward(self,inputs,hiddens):
        e = self.emb(inputs).view(len(inputs),1,-1)
        lstm_out,hiddens = self.lstm(e,hiddens)
        u = self.lin_trans(lstm_out.view(len(inputs),-1))
        return F.log_softmax(u,dim=1),hiddens

    def init_hiddens(self,b_size,device):
        weight = next(self.parameters())
        n_layers = self.n_layers
        if self.bi_dir:
            n_layers *= 2
        return ( weight.new_zeros(n_layers,b_size,self.lstm_dim).to(device)
               , weight.new_zeros(n_layers,b_size,self.lstm_dim).to(device)
               )

def get_model(load_model,model_data,vocab,device):
    model_name = model_data["model_name"]
    model = CnnLstm(vocab["size"],model_data)
    if load_model:
        model_name += ".ptm"
        model.load_state_dict( ts.load( models_path + model_name
                             , map_location=ts.device(device))
                             )
        print("model loaded from " + model_name)
    model.to(device)
    return model

def save_model(model_name,model):
    model_name += ".ptm"
    ts.save(model.state_dict(),models_path + model_name)
    print("model saved as '" + model_name + "'")
