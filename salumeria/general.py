
def readlist(filename):
    # just read a list from a file
    # return a python list
    with open(filename) as fn:
        return [i.rstrip() for i in fn.readlines()]
