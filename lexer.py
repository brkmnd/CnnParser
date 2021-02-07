import re

c_specchars = [
          "!"
        , "&"
        , "\\|"
        , "~"
        , "\\^"
        , "+"
        , "\\-"
        , "\\*"
        , "/"
        , "%"
        , "<"
        , ">"
        , "."
        , ","
        , ":"
        , ";"
        , "?"
        ]
c_keyword_list = [
        # https://gcc.gnu.org/onlinedocs/gcc/Keyword-Index.html
          "asm"
        , "auto"
        , "bool"
        , "break"
        , "case"
        , "char"
        , "const"
        , "continue"
        , "default"
        , "do"
        , "double"
        , "else"
        , "enum"
        , "extern"
        , "float"
        , "for"
        , "goto"
        , "if"
        , "inline"
        , "int"
        , "long"
        , "register"
        , "return"
        , "short"
        , "signed"
        , "sizeof"
        , "static"
        , "struct"
        , "switch"
        , "typedef"
        , "union"
        , "unsigned"
        , "void"
        , "volatile"
        , "while"
        ]
c_prepros_decs = [
        # https://gcc.gnu.org/onlinedocs/cpp/Index-of-Directives.html
          "assert"
	, "define"
	, "elif"
	, "else"
	, "endif"
	, "error"
	, "ident"
	, "if"
	, "ifdef"
	, "ifndef"
	, "import"
	, "include"
	, "include_next"
	, "line"
	, "pragma GCC dependency"
	, "pragma GCC error"
	, "pragma GCC poison"
	, "pragma GCC system_header"
	, "pragma GCC system_header"
	, "pragma GCC warning"
	, "pragma once"
	, "sccs"
	, "unassert"
	, "undef"
	, "warning"
        ]
c_pars = [
          "(",")"
        , "[","]"
        , "{","}"
        ]

def read_file(fpath):
    res = ""
    with open(fpath,"r") as f:
        res = f.read()
    return res

def remove_multline_comm(txt):
    comm_on = False
    res = ""
    i = 0
    txt += " "
    while i < len(txt) - 1:
        c0 = txt[i]
        c1 = txt[i + 1]
        if c0 + c1 == "/*":
            comm_on = True
            i += 2
            continue
        elif c0 + c1 == "*/" and comm_on:
            comm_on = False
            i += 2
            continue
        elif not comm_on:
            res += c0
        i += 1
    return res

def create_token(t):
    if t[:2] == "//":
        return None
    elif t[0] == "@" or t[0] == "#":
        return t
    elif t[0] == "\"":
        return "@string"
    elif t[0] == "'":
        return "@char"
    elif t[0] in [str(x) for x in range(10)]:
        return "@num"
    elif t[0] in c_specchars or t[0] in c_pars or t in c_keyword_list:
        return t
    elif t not in c_keyword_list:
        return "@ident"
    else:
        print("lexer warning: no match for token : " + t)
        return None

def lex(txt):
    rx_exprs = [
              "\"[^\"]*\"" # strings
            , "'[^']'" # chars
            , "//[^\n]*\n"
            , "(?:" + "|".join(["#" + x for x in c_prepros_decs]) + ")" # decs for preprocessor
            , "@[a-zA-Z0-9_\-]+" # internal tokens
            , "[a-zA-Z]+[a-zA-Z_0-9]*" # identifiers and keywords
            , "[0-9]+(?:.[0-9]+){0,1}" # numbers - int and floats
            , "|".join(["\\" + x for x in c_pars]) # parentheses
            , "[" + "".join(c_specchars) + "]+" # operators
            ]
    rx = re.compile("|".join(rx_exprs))
    txt_nocomm = remove_multline_comm(txt)
    txt_nocomm = re.sub("<[a-zA-Z]+.[^\s>]*>"," @croc-dir ",txt_nocomm) # remove <file.txt> includes
    # remove escaped esc-chars inside strings
    # odd times esc-char means escape the quote "
    # even means keep quote
    txt_no_esc = txt_nocomm.replace("\\" * 4 + "\"","dup-esc\"")
    txt_no_esc = txt_no_esc.replace("\\" * 3 + "\"","dup-esc`")
    txt_no_esc = txt_no_esc.replace("\\" * 2 + "\"","dup-esc\"")
    txt_no_esc = txt_no_esc.replace("\\" + "\"","`")
    ts = re.findall(rx,txt_no_esc)
    ts0 = []
    for t in ts:
        t = create_token(t)
        if t != None:
            ts0.append(t)
    return ts0



if __name__ == "__main__":
    f0 = read_file("datasets/06/aarch64.c")
    f1 = read_file("datasets/04/c-format.c")
    f2 = read_file("datasets/07/pt.c")
    ts = lex(f2)
    #print(ts)
    print(len(ts))
    vocab = set(ts)
    print(vocab)

