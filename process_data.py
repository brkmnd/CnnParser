from lexer import lex
import numpy as np
import os
import re
import json

file_endings = [
          ".txt"
        , ".swp"
        , ".json"
        ]

#split_token = "§§§§"

def is_file(d):
    return (  d[-4:] in file_endings
           or d[-5:] in file_endings
           )

def fsave(txt):
    with open("datasets/all_data.txt","w") as f:
        f.write(txt)

def fsave_json(res):
    with open("datasets/all_data.json","w") as f:
        json.dump(res,f)
    print("data saved to all_data.json")

def fopen_json():
    res = {}
    with open("datasets/all_data.json","r") as f:
        res = json.load(f)
    return res

def save_search(txt):
    with open("searc_results.txt","w") as f:
        f.write(txt)

def fopen(fname):
    res = ""
    with open(fname,"r") as f:
        res = f.read()
    return res

def search_in_files(needle):
    rx = re.compile(needle)
    needle = needle.replace("\\\\","\\")
    path = "datasets"
    dirs = os.listdir(path)
    res = ""
    for d in dirs:
        if d[-4:] in file_endings:
            continue
        p0 = path + "/" + d
        fs = os.listdir(p0)
        for f in fs:
            p1 = p0 + "/" + f
            txt = fopen(p1)
            if len(re.findall(rx,txt)) > 0:
                inds = [m.start() for m in re.finditer(rx,txt)]
                res += "'" + needle + "' found in " + p1 + "\n"
                for i in inds:
                    res += "----around: " + txt[i - 10:i + 10] + "\n"
    return res

def dir_stats(d):
    path = "datasets/" + d
    fs = os.listdir(path)

    lens = []
    vocab = []

    for f in fs:
        if f[-4:] in file_endings:
            continue
        f_str = fopen(path + "/" + f)
        ts = lex(f_str)
        lens.append(len(ts))
        vocab = vocab + ts

    lens = np.array(lens)
    vocab = list(set(vocab))
    print("lens:")
    print(lens)
    print("stats for dir '" + d + "'")
    print("  vocab len               : " + str(len(vocab)))
    print("  prg len-tokens          : " + str(lens.shape[0]))
    print("  prg max-len-tokens      : " + str(lens.max()))
    print("  prg min-len-tokens      : " + str(lens.min()))
    print("  prg avg-len-tokens      : " + str(round(lens.mean())))
    print("  prg variance-len-tokens : " + str(round(np.std(lens))))
    print("  prg len-tokens<=5000    : " + str(np.count_nonzero(lens <= 5000)))

def data_stats():
    None

def process_data():
    # we store using json
    # so path => tokens

    path = "datasets"
    dirs = os.listdir(path)
    res = {}
    lens = []
    lens_disc = []
    lens_limit = 5000

    for d in dirs:
        if is_file(d):
            continue
        p0 = path + "/" + d
        fs = os.listdir(p0)
        for f in fs:
            p1 = p0 + "/" + f
            txt = fopen(p1)
            ts = lex(txt)
            len_ts = len(ts)
            if len_ts <= lens_limit and len_ts > 0:
                res[p1] = ts
                lens.append(len_ts)
            else:
                lens_disc.append(len_ts)

    lens = np.array(lens)
    print("data processed, stats:")
    print("  nr prgs       : " + str(lens.shape[0]))
    print("  avg len       : " + str(round(lens.mean())))
    print("  var len       : " + str(round(lens.std())))
    print("  max len       : " + str(lens.max()))
    print("  min len       : " + str(lens.min()))
    print("  discarded len : " + str(len(lens_disc)))
    fsave_json(res)

def load_data():
    res = []
    res_json = fopen_json()

    for k in res_json:
        res.append(res_json[k])

    return res

if __name__ == "__main__":
    process_data()
    #dir_stats("01")
    #sres = search_in_files("\\\\\"")
    #save_search(sres)
    
