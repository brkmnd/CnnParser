import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import time

from process_data import load_data
from seqs import create_seqs_splits,create_seqs,create_vocab,reverse_seqs
from utils import comp_time,save_acc,load_initcode

def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    ts.manual_seed(seed)
    ts.cuda.manual_seed_all(seed)
    # For atomic operations there is currently 
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    ts.backends.cudnn.deterministic = True
    ts.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)
enforce_reproducibility()

class CnnModel(nn.Module):
    def __init__(self,vocab_size,emb_dim,lstm_dim,n_layers,dropout_p,bi_dir):
        super(CnnModel,self).__init__()

        self.n_layers = n_layers
        self.lstm_dim = lstm_dim
        self.bi_dir = bi_dir

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

def concat_tokens(words,word2id,device):
    return ts.tensor([word2id[w] for w in words],dtype=ts.long).to(device)

def get_model(load_model,model_data,vocab_size):
    lstm_dim = model_data["lstm_dim"]
    emb_dim = model_data["emb_dim"]
    model_name = model_data["model_name"]
    dropout_p = model_data["dropout"]
    bi_dir = model_data["bi-directional"]
    n_layers = model_data["n_layers"]
    model = CnnModel(vocab_size,emb_dim,lstm_dim,n_layers,dropout_p,bi_dir)
    if load_model:
        model_name += ".ptm"
        model.load_state_dict(ts.load(model_name))
        print("model loaded " + model_name)
    model.to(device)
    return model

def train_model(t_data,model,l_rate,b_size,word2id,device):
    n_epochs,seqs,seqs_val = t_data

    loss_f = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr=l_rate)

    xs,ys = seqs[0],seqs[1]
    n = xs.shape[0]
    est_time_n = 0
    avg_acc = 0
    avg_acc3 = 0

    for epoch in range(n_epochs):
        avg_loss = []
        start_time = time.time()

        model.train()
        hiddens = model.init_hiddens(b_size,device)

        for inputs,targets in zip(xs,ys):
            model.zero_grad()

            inputs = concat_tokens(inputs,word2id,device)
            targets = concat_tokens(targets,word2id,device)
            hiddens = [h.detach() for h in hiddens]
            logits,hiddens = model(inputs,hiddens)

            loss = loss_f(logits,targets)

            avg_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            est_time_n += 1
            if est_time_n == 1000:
                print("est-time per epoch: " + comp_time(start_time,lambda t0 : t0 * n / 1000))
                print("")

        acc,_  = eval_model(model,b_size,seqs_val,1,word2id,device)
        acc3,_ = eval_model(model,b_size,seqs_val,3,word2id,device)
        avg_loss = np.array(avg_loss)
        avg_acc  += acc
        avg_acc3 += acc3

        print("accuracy [" + str(epoch) + "] : " + str(acc))
        print("accuracy3[" + str(epoch) + "] : " + str(acc3))
        print("loss [" + str(epoch) + "]     : " + str(avg_loss.mean()))
        print("----took " + comp_time(start_time,lambda x: x))

    return model,avg_acc / n_epochs,avg_acc3 / n_epochs

def eval_model(model,b_size,seqs,n_comp,word2id,device):
    xs = seqs[0]
    ys = seqs[1]

    n = xs.shape[0] * xs.shape[1]

    corrects = 0
    incorrects = 0

    model.eval()

    hiddens = model.init_hiddens(b_size,device)

    with ts.no_grad():
        for (inputs,targets) in zip(xs,ys):
            inputs = concat_tokens(inputs,word2id,device)

            preds,hiddens = model(inputs,hiddens)

            for pred,t in zip(preds,targets):
                ys_hat = pred.argsort()[-n_comp:]
                t = word2id[t]

                if t in ys_hat:
                    corrects += 1
                else:
                    incorrects += 1
    
    acc = corrects / n
    err_rate = incorrects / n

    return acc,err_rate

device = ts.device("cpu")
if ts.cuda.is_available():
    print("has cuda")
    device = ts.device("cuda")

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
        }

m0 = model_dict["m2"]

def complete_me(input_txt,n_sugs):
    from lexer import lex

    model_name = m0["model_name"]
    b_size = m0["batch_size"]
    
    txts = load_data()
    vocab,vocab_size,word2id,id2word = create_vocab(txts)

    model = get_model(True,m0,vocab_size)
    model.eval()
    hiddens = model.init_hiddens(b_size,device)

    res = []

    with ts.no_grad():
        xs = lex(input_txt)
        x0 = concat_tokens([xs[-1]],word2id,device)
        print(xs)
        for x in xs[:-1]:
            # initialize hidden states on inputs src
            x = concat_tokens([x],word2id,device)
            _,hiddens = model(x,hiddens)
        
        preds,_ = model(x0,hiddens)
        n_preds = preds.view(-1).argsort()[-n_sugs:]
        res = [id2word[i] for i in n_preds]

    return res

def save_model(model_name,model):
    model_name += ".ptm"
    ts.save(model.state_dict(),model_name)
    print("model saved as '" + model_name + "'")


def main():
    txts = load_data()
    txts_train = txts[:150]
    txts_test = txts[150:200]

    n_epochs = 2
    load_model = True
    model_name = m0["model_name"]

    l_rate = 1 / 10 ** 5
    seq_len = 128
    n_splits = 8
    b_size = m0["batch_size"]

    seqs = create_seqs_splits(txts_train,seq_len,n_splits)
    seqs = np.array(seqs)
    np.random.shuffle(seqs)
    seqs_test = create_seqs(txts_test,seq_len)
    vocab,vocab_size,word2id,id2word = create_vocab(txts)

    model = get_model(load_model,m0,vocab_size)
    
    print("vocab size:" + str(vocab_size))
    if n_epochs > 0:
        split_i = 1
        avg_acc = 0
        avg_acc3 = 0
        start_time = time.time()
        for seqs_train,seqs_val in seqs:
            print("\n")
            print("********train split[" + str(split_i) + "/" + str(n_splits) + "]")
            print("train_split shape : " + str(seqs_train[0].shape))
            print("val_split shape   : " + str(seqs_val[0].shape))
            t_data = n_epochs,seqs_train,seqs_val
            model,acc,acc3 = train_model(t_data,model,l_rate,b_size,word2id,device)
            avg_acc += acc
            avg_acc3 += acc3
            split_i += 1
        avg_acc = avg_acc / n_splits
        avg_acc3 = avg_acc3 / n_splits
        print("")
        print("**avg accuracy     : " + str(round(100 * avg_acc,2)) + "%")
        print("**avg accuracy3    : " + str(round(100 * avg_acc3,2)) + "%")
        print("**total time taken : " + comp_time(start_time,lambda x: x))
        if model_name != None:
            save_model(model_name,model)
            save_acc(model_name,n_epochs,avg_acc,avg_acc3)
    elif n_epochs == -1:
        split_i = 1
        avg_acc = 0
        avg_acc3 = 0
        for _,seqs_val in seqs:
            acc,_  = eval_model(model,b_size,seqs_val,1,word2id,device)
            acc3,_ = eval_model(model,b_size,seqs_val,3,word2id,device)
            avg_acc  += acc
            avg_acc3 += acc3
            print("")
            print("********eval split[" + str(split_i) + "]")
            print("split shape : " + str(seqs_val[0].shape))
            print("accuracy    : " + str(acc))
            print("accuracy3   : " + str(acc3))
            split_i += 1
        avg_acc /= n_splits
        avg_acc3 /= n_splits
        print("")
        print("**avg accuracy : " + str(round(100 * avg_acc,2)) + "%")
        print("**avg accuracy3: " + str(round(100 * avg_acc3,2)) + "%")
    elif n_epochs == -2:
        print("")
        print("********eval on test set")
        print("test set shape : " + str(seqs_test[0].shape))
        start_time = time.time()
        acc,_  = eval_model(model,b_size,seqs_test,1,word2id,device)
        acc3,_ = eval_model(model,b_size,seqs_test,3,word2id,device)
        print("accuracy       : " + str(acc))
        print("accuracy3      : " + str(acc3))
        print("took           : " + comp_time(start_time,lambda x: x))

def do_complete():
    src = load_initcode("1.c")
    sugs = complete_me(src,3)
    print("")
    print(sugs)

if __name__ == "__main__":
    #main()
    do_complete()
    
