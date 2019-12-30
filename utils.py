import os


def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def StrToInt(s):
    return int(s)

def StrToBool(s):
    if s.lower() == 'false':
        return False
    else:
        return True

def clc_pad(h,w,st=32):## default st--> 32
    def _f(s):
        n = s//st
        r = s %st
        if r == 0:
            return 0
        else:
            return st-r
    return _f(h),_f(w)
    