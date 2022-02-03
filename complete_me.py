from models import model_dict,get_model
from run_model import device,pred_model
from process_data import load_data
from seqs import create_vocab

init_code1 = ["#include","@string"] * 10

def main_loop():
    answ_quit = [":q",":quit"]
    answ = ""
    init_prg = init_code1
    prg = []

    txts = load_data()
    m0 = model_dict["m5"]
    vocab = create_vocab(txts)
    model = get_model(True,m0,vocab,device)
    sugs = []
    
    while answ not in answ_quit:
        print("")
        print("-- current program:")
        print(" ".join(prg))
        print("")

        print("-- suggestions:")
        if len(init_prg) + len(prg) > 0:
            sugs = pred_model(model,m0,vocab,init_prg + prg,5,device)
            sugs = {str(i) : sugs[i] for i in range(len(sugs))}
            for s0 in sugs:
                print(s0 + " : " + sugs[s0])
        print("")

        print("(:q for quit, :n for suggestion n, :r for reset, :b for backspace)")
        answ = input("write next token : ")
        if answ[0] == ":":
            c = answ[1:]
            if c in sugs:
                answ = sugs[c]
            elif c == "r":
                prg = []
                continue
            elif c == "b":
                prg = prg[:-1]
                continue
            else:
                print("command :" + c + " not recognized")
                continue
        elif not vocab["in-vocab"](answ) and answ not in answ_quit:
            print("token not recognized '" + answ + "'")
            continue
    
        prg.append(answ)

if __name__ == "__main__":
    main_loop()
