import numpy as np

tok_pad = "<pad>"
tok_start = "<s>"
tok_end = "</s>"

def find_split_sizes(n_txts):
    res = []
    for i in range(1,40):
        if n_txts % i == 0:
            res.append(i)
    return res

def compile_seqs(txts,seq_len):
    # returns seqs xs,ys each of shape (*,seq_len)
    # ys are equal to xs but rotated 1 pos to the right
    # for example xs = [1,2,3,...] => ys = [2,3,4,...]
    # pad tokens are added to the end to align
    ts = [x for xs in txts for x in xs]
    n_ts = len(ts)

    res_len = n_ts % seq_len
    ts = ts + [tok_pad] * (seq_len - res_len)

    n_ts = len(ts)
    xs = np.array(ts)
    xs = xs.reshape(n_ts // seq_len,seq_len)
    ys = np.array(ts[1:] + [tok_pad])
    ys = ys.reshape(n_ts // seq_len,seq_len)

    return xs,ys

def create_splits(txts,seq_len,n_splits):
    # returns a list of seq_train,seq_val
    # this list composes of cross-split combination
    # for example ([1,2],[3]) , ([1,3],[2]) , ([2,3],[1])
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
        seqs_train = compile_seqs(txts_train,seq_len)
        seqs_val = compile_seqs(txts_val,seq_len)

        res.append((seqs_train,seqs_val))

    return res

def create_seqs_splits(txts,seqs_len,n_splits):
    return create_splits(txts,seqs_len,n_splits)

def create_seqs(txts,seqs_len):
    return compile_seqs(txts,seqs_len)

def create_vocab(txts):
    ts = [x for xs in txts for x in xs]

    vocab = list(set(ts)) + [tok_pad]
    vocab.sort()
    vocab_size = len(vocab)

    word2id = {vocab[i] : i for i in range(vocab_size)}
    id2word = vocab

    return vocab,vocab_size,word2id,id2word
