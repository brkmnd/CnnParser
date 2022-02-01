import torch as ts
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

from process_data import load_data
from seqs import create_seqs_splits,create_seqs,create_vocab,reverse_seqs
from utils import comp_time,get_time,save_acc,load_initcode,enforce_reproducibility
from models import get_model,save_model

enforce_reproducibility()

device = ts.device("cpu")
if ts.cuda.is_available():
    print("has cuda")
    device = ts.device("cuda")

def concat_tokens(words,word2id,device):
    return ts.tensor([word2id[w] for w in words],dtype=ts.long).to(device)

def train_model(t_data,model,l_rate,b_size,word2id,device):
    n_epochs,seqs_train,seqs_val = t_data

    loss_f = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr=l_rate)

    xs,ys = seqs_train[0],seqs_train[1]
    n = len(xs)
    est_time_n = 0
    avg_acc = 0
    avg_acc3 = 0

    for epoch in range(n_epochs):
        avg_loss = []
        start_time = get_time()

        model.train()

        for inputs,targets in zip(xs,ys):
            model.zero_grad()
            # reset hiddens after each input since input = whole c-program
            hiddens = model.init_hiddens(b_size,device)

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

    n = 0

    corrects = 0
    incorrects = 0

    model.eval()

    with ts.no_grad():
        for (inputs,targets) in zip(xs,ys):
            hiddens = model.init_hiddens(b_size,device)
            inputs = concat_tokens(inputs,word2id,device)

            preds,hiddens = model(inputs,hiddens)

            for pred,t in zip(preds,targets):
                ys_hat = pred.argsort()[-n_comp:]
                t = word2id[t]
                n += 1
                
                if t in ys_hat:
                    corrects += 1
                else:
                    incorrects += 1
    
    acc = corrects / n
    err_rate = incorrects / n

    return acc,err_rate

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
        , "m4": {
              "lstm_dim":256
            , "n_layers":2
            , "emb_dim":128
            , "model_name":"cnn_model4"
            , "batch_size":1
            , "dropout":0.2
            , "bi-directional":False
            }
        }

def main_train(model,m0,vocab,seqs,n_epochs=1,l_rate=1 / 10 ** 6):
    split_i = 1
    avg_acc = 0
    avg_acc3 = 0
    start_time = get_time()

    n_splits = len(seqs)
    model_name = m0["model_name"]
    b_size = m0["batch_size"]
    word2id = vocab["word2id"]

    for seqs_train,seqs_val in seqs:
        print("\n")
        print("********train split[" + str(split_i) + "/" + str(n_splits) + "]")
        print("train_split length : " + str(len(seqs_train[0])))
        print("val_split length   : " + str(len(seqs_val[0])))
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

def main_evalsplit(model,m0,vocab,seqs):
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

def main_evaltest(model,m0,vocab,seqs_test):
    b_size = m0["batch_size"]
    word2id = vocab["word2id"]

    print("")
    print("********eval on test set")
    print("test set shape : " + str(seqs_test[0].shape))
    start_time = get_time()
    acc,_  = eval_model(model,b_size,seqs_test,1,word2id,device)
    acc3,_ = eval_model(model,b_size,seqs_test,3,word2id,device)
    print("accuracy       : " + str(acc))
    print("accuracy3      : " + str(acc3))
    print("took           : " + comp_time(start_time,lambda x: x))


def main():
    txts = load_data()
    txts_train = txts[:150]
    txts_test = txts[150:]

    m0 = model_dict["m4"]
    load_model = True

    n_splits = 10
    n_datas = len(txts)
    n_trains = len(txts_train)
    n_tests = len(txts_test)

    seqs = create_seqs_splits(txts_train,n_splits)
    random.shuffle(seqs)
    seqs_test = create_seqs(txts_test)
    vocab = create_vocab(txts)
    model = get_model(load_model,m0,vocab,device)

    print("")
    print("vocab size      : " + str(vocab["size"]))
    print("dataset size    : " + str(n_datas))
    print("train-set size  : " + str(n_trains))
    print("test-set size   : " + str(n_tests))
    print("train-set ratio : " + str(n_trains / n_datas))
    print("")
    
    main_train(model,m0,vocab,seqs,n_epochs=5,l_rate=1/10**5)
    #main_evalsplit(model,m0,vocab,seqs)
    #main_evaltest(model,m0,vocab,seqs_test)

    return True

def do_complete():
    src = load_initcode("1.c")
    sugs = complete_me(src,3)
    print("")
    print(sugs)

if __name__ == "__main__":
    main()
    #do_complete()
    
