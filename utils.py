import time
import numpy as np
import torch as ts
import random

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

def gen_rand_int(time_f):
    def run_tf():
        return time_f(np.random.random())

    t0 = str(time.time())[-2:]
    t0 = int(t0)
    rand_i = run_tf()

    for _ in range(t0):
        rand_i = run_tf()

    return rand_i

def get_time():
    return time.time()

def comp_time(t0,time_fun):
    if time_fun is None:
        time_fun = lambda x: x

    used_time = time_fun(round(time.time() - t0,2))
    measure = "seconds"

    if used_time >= 60.0:
        used_time /= 60.0
        #used_time = round(used_time,2)
        measure = "minutes"
    if used_time >= 60.0:
        used_time /= 60.0
        #used_time = round(used_time,2)
        measure = "hours"
    used_time = round(used_time,2)
    return str(used_time) + " " + measure

def save_acc(model_name,n_epochs,acc,acc3):
    with open("models/" + model_name + ".acc.txt","a") as f:
        res  = "*" * 8 + "\n"
        res += "[" + str(n_epochs) + "] : " + str(acc) + "\n"
        res += "[" + str(n_epochs) + "]3: " + str(acc3) + "\n"
        f.write(res)

def load_initcode(fname):
    res = ""
    with open("init_code/" + fname,"r") as f:
        res = f.read()
    return res

