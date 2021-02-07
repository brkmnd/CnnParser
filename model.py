import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import time

from process_data import load_data
from seqs import create_seqs_splits,create_seqs,create_vocab
from utils import comp_time,save_acc

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
    def __init__(self,vocab_size,emb_dim,lstm_dim):
        super(CnnModel,self).__init__()

        self.n_layers = 1
        self.lstm_dim = lstm_dim

        #layers
        self.emb = nn.Embedding(vocab_size,emb_dim)
        self.lstm = nn.LSTM(emb_dim,lstm_dim)
        self.lin_trans = nn.Linear(lstm_dim,vocab_size)

    def forward(self,inputs,hiddens):
        e = self.emb(inputs).view(len(inputs),1,-1)
        lstm_out,hiddens = self.lstm(e,hiddens)
        u = self.lin_trans(lstm_out.view(len(inputs),-1))
        return F.log_softmax(u,dim=1),hiddens

    def init_hiddens(self,b_size,device):
        weight = next(self.parameters())
        return ( weight.new_zeros(self.n_layers,b_size,self.lstm_dim).to(device)
               , weight.new_zeros(self.n_layers,b_size,self.lstm_dim).to(device)
               )

def concat_tokens(words,word2id,device):
    return ts.tensor([word2id[w] for w in words],dtype=ts.long).to(device)

def get_model(load_model,model_data,vocab_size):
    lstm_dim = model_data["lstm_dim"]
    emb_dim = model_data["emb_dim"]
    model_name = model_data["model_name"]
    model = CnnModel(vocab_size,emb_dim,lstm_dim)
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

    n = seqs[0].shape[0]
    est_time_n = 0
    avg_acc = 0

    for epoch in range(n_epochs):
        avg_loss = []
        start_time = time.time()

        model.train()
        hiddens = model.init_hiddens(b_size,device)

        for inputs,targets in zip(seqs[0],seqs[1]):
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
                print("est-time per epoch:" + comp_time(start_time,lambda t0 : t0 * n / 1000))
                print("")

        acc,err_rate = eval_model(model,b_size,seqs_val,word2id,device)
        avg_loss = np.array(avg_loss)
        avg_acc += acc

        print("accuracy[" + str(epoch) + "] : " + str(acc))
        print("loss[" + str(epoch) + "]     : " + str(avg_loss.mean()))
        print("----took " + comp_time(start_time,lambda x: x))

    return model,avg_acc / n_epochs

def eval_model(model,b_size,seqs,word2id,device):
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
            targets = concat_tokens(targets,word2id,device)

            preds,hiddens = model(inputs,hiddens)

            for pred,t in zip(preds,targets):
                y_hat = pred.argmax().item()
                t = t.item()
                if y_hat == t:
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
            , "emb_dim":128
            , "model_name":"cnn_model1"
            , "batch_size":1
            }
        }

def complete_me(input_txt,n_sugs):
    from lexer import lex

    m0 = model_dict["m1"]
    model_name = m0["model_name"]
    b_size = m0["batch_size"]
    tok_start = "<s>"
    
    txts = load_data()
    vocab,vocab_size,word2id,id2word = create_vocab(txts)

    model = get_model(True,m0,vocab_size)
    model.eval()
    hiddens = model.init_hiddens(b_size,device)
    with ts.no_grad():
        xs = [tok_start] + lex(input_txt)
        for x in xs:
            x = concat_tokens([x],word2id,device)
            
            preds,hiddens = model(x,hiddens)
            preds = np.array(preds.tolist()).reshape(vocab_size)
            n_preds = preds.argsort()[:n_sugs]
            print(n_preds)

            break
        #inputs = concat_tokens(tok_start + input_txt)

def save_model(model_name,model):
    model_name += ".ptm"
    ts.save(model.state_dict(),model_name)
    print("model saved as '" + model_name + "'")


def main():
    m0 = model_dict["m1"]

    txts = load_data()
    txts_train = txts[:150]
    txts_test = txts[150:200]

    n_epochs = 3
    load_model = True
    model_name = m0["model_name"]

    l_rate = 1 / 10 ** 5
    seq_len = 128
    n_splits = 5
    b_size = m0["batch_size"]

    seqs = create_seqs_splits(txts_train,seq_len,n_splits)
    seqs_test = create_seqs(txts_test,seq_len)
    vocab,vocab_size,word2id,id2word = create_vocab(txts)

    model = get_model(load_model,m0,vocab_size)
    
    print("vocab size:" + str(vocab_size))
    if n_epochs > 0:
        split_i = 1
        avg_acc = 0
        start_time = time.time()
        for seqs_train,seqs_val in seqs:
            print("\n")
            print("********train split[" + str(split_i) + "/" + str(n_splits) + "]")
            print("train_split shape : " + str(seqs_train[0].shape))
            print("val_split shape   : " + str(seqs_val[0].shape))
            t_data = n_epochs,seqs_train,seqs_val
            model,acc = train_model(t_data,model,l_rate,b_size,word2id,device)
            avg_acc += acc
            split_i += 1
        avg_acc = avg_acc / n_splits
        print("")
        print("**avg accuracy     : " + str(round(100 * avg_acc,2)) + "%")
        print("**total time taken : " + comp_time(start_time,lambda x: x))
        if model_name != None:
            save_model(model_name,model)
            save_acc(model_name,n_epochs,avg_acc)
    elif n_epochs == -1:
        split_i = 1
        avg_acc = 0
        for _,seqs_val in seqs:
            acc,err_rate = eval_model(model,b_size,seqs_val,word2id,device)
            avg_acc += acc
            print("")
            print("********eval split[" + str(split_i) + "]")
            print("split shape : " + str(seqs_val[0].shape))
            print("accuracy    : " + str(acc))
            split_i += 1
        avg_acc /= n_splits
        print("")
        print("**avg accuracy: " + str(round(100 * avg_acc,2)) + "%")
    elif n_epochs == -2:
        start_time = time.time()
        acc,err_rate = eval_model(model,b_size,seqs_test,word2id,device)
        print("")
        print("********eval on test set")
        print("test set shape : " + str(seqs_test[0].shape))
        print("accuracy       : " + str(acc))
        print("took           : " + comp_time(start_time,lambda x: x))


if __name__ == "__main__":
    main()
    #sugs = complete_me("int main()",5)
    #print(sugs)
    
