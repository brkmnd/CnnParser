from lexer import lex
import os
import re

file_endings = [
          ".txt"
        , ".swp"
        ]

split_token = "§§§§"

def fsave(txt):
    with open("datasets/all_data.txt","w") as f:
        f.write(txt)

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

def process_data():
    path = "datasets"
    dirs = os.listdir(path)
    res = []
    for d in dirs:
        if d[-4:] in file_endings:
            continue
        p0 = path + "/" + d
        fs = os.listdir(p0)
        for f in fs:
            p1 = p0 + "/" + f
            txt = fopen(p1)
            ts = lex(txt)
            if len(ts) <= 4000:
                res.append(split_token.join(ts))
    fsave("\n".join(res))

def load_data():
    res = []
    with open("datasets/all_data.txt","r") as f:
        lines = f.read().split("\n")
        for l in lines:
            if l == "":
                continue
            ts = l.split(split_token)
            res.append(ts)
    return res

if __name__ == "__main__":
    process_data()
    #sres = search_in_files("\\\\\"")
    #save_search(sres)
    
