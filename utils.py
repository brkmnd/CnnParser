import time
import numpy as np

def gen_rand_int(time_f):
    def run_tf():
        return time_f(np.random.random())

    t0 = str(time.time())[-2:]
    t0 = int(t0)
    rand_i = run_tf()

    for _ in range(t0):
        rand_i = run_tf()

    return rand_i

def comp_time(t0,time_fun):
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
    with open(model_name + ".acc.txt","a") as f:
        res  = "*" * 8 + "\n"
        res += "[" + str(n_epochs) + "] : " + str(acc) + "\n"
        res += "[" + str(n_epochs) + "]3: " + str(acc3) + "\n"
        f.write(res)

def load_initcode(fname):
    res = ""
    with open("init_code/" + fname,"r") as f:
        res = f.read()
    return res

