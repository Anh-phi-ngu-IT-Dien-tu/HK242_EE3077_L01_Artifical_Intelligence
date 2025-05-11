import numpy as np
import json

h=np.array([[1],[0],[0],[0]])
e=np.array([[0],[1],[0],[0]])
l=np.array([[0],[0],[1],[0]])
o=np.array([[0],[0],[0],[1]])

with open("h.txt", "w") as w:
    w.write(str(h))

with open("e.txt", "w") as w:
    w.write(str(e))

with open("l.txt", "w") as w:
    w.write(str(l))

with open("o.txt", "w") as w:
    w.write(str(o))

