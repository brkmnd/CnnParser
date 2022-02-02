import numpy as np

toks = { "pad"   : "<pad>"
       , "start" : "<s>"
       , "end"   : "</s>"
       , "unk"   : "<unk>"
       , "eof"   : "<eof>"
       }

def find_split_sizes(n_txts):
    res = []
    for i in range(1,40):
        if n_txts % i == 0:
            res.append(i)
    return res

def compile_seqs(txts):
    # returns seqs xs,ys each of length len(txts)
    # ys are xs but rotated 1 pos right:
    # [1,2,3,4] -> [2,3,4,eof]
    xs = []
    ys = []
    
    for txt in txts:
        xs.append(txt)
        ys.append(txt[1:] + [toks["eof"]])

    return xs,ys

def create_splits(txts,n_splits):
    # returns a list of seq_train,seq_val
    # this list composes of cross-split combination
    # for example ([1,2],[3]) , ([1,3],[2]) , ([2,3],[1])

    # the structure of res is
    #   split_id -> seqs_train,seqs_val
    #   seqs_train : [xs,ys]
    n_txts = len(txts)
    split_len = n_txts // n_splits
    if n_txts % n_splits == 0:
        print("splits are even")
    else:
        print("warning: splits are uneven, good split sizes are: " + str(find_split_sizes(n_txts)))

    res = []

    for si in range(0,n_txts,split_len):
        txts_val = txts[si:si + split_len]
        txts_train = txts[:si] + txts[si + split_len:]
        seqs_train = compile_seqs(txts_train)
        seqs_val = compile_seqs(txts_val)

        res.append((seqs_train,seqs_val))

    return res

def reverse_seqs(seqs):
    return np.flip(seqs,(0,1))

def create_seqs_splits(txts,n_splits):
    return create_splits(txts,n_splits)

def create_seqs(txts):
    return compile_seqs(txts)

def create_vocab(txts):
    ts = [x for xs in txts for x in xs]

    vocab = list(set(ts)) + [toks[k] for k in toks]
    vocab.sort()
    vocab_size = len(vocab)

    word2id = {vocab[i] : i for i in range(vocab_size)}
    id2word = vocab

    def word2id_fun(w):
        if w not in word2id:
            return word2id[toks["unk"]]
        return word2id[w]
    
    def id2word_fun(i):
        if i >= vocab_size:
            return toks["unk"]
        return id2word[i]

    retval = { "vocab"    : vocab
             , "size"     : vocab_size
             , "word2id"  : word2id
             , "id2word"  : id2word
             }

    return retval

if __name__ == "__main__":
    a1 = [range(i,i+3) for i in range(0,10,3)]
    a1 = np.array(a1)
    print(a1)
    b1 = reverse_seqs(a1)
    print(b1)
