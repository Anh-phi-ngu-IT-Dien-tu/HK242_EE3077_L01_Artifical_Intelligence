import numpy as np
import math


def cross_entrophi_loss(p=np.zeros((3,1)),q=np.zeros((3,1))):
    loss=0
    if p.shape[0]==q.shape[0] and p.shape[1]==q.shape[1]:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                loss+=p[i,j]*np.log2(q[i,j])

    return loss

with open("h.txt", "r") as r:
    temp=r.read()
    temp=temp.replace("["," ")
    temp=temp.replace("]"," ")


print(temp)
h=np.fromstring(temp,sep='\n')

print(h)

    