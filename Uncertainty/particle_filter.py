import numpy as np


deltaV=np.array([1.01,0.61,-0.95,-0.3,-0.95,0.64,-0.35,-0.9,0,0.21,-0.07,-0.11,1.25,-0.15,0.71,-0.68,-0.01,-0.72,0.53,-0.8])
deviation_d=0.1
xl=10
xt=2
deltaT=0.2
V=5
d=7

xt_p=np.zeros((len(deltaV)))
wt_p=np.zeros((len(deltaV)))
wdisk_p=np.zeros((len(deltaV)))
dhat_p=np.zeros((len(deltaV)))

for i in range(len(deltaV)):
    xt_p[i]=xt+(V+deltaV[i])*deltaT
    dhat_p[i]=np.abs(xt_p[i]-xl)
    wt_p[i]=np.exp(-0.5*((dhat_p[i]-d)**2)/(deviation_d**2))/(np.sqrt(2*np.pi)*deviation_d)
    print(f"xt_p[{i}]: {xt_p[i]}, dhat_p[{i}]: {dhat_p[i]}, wt_p[{i}]: {wt_p[i]}")

with open('weights.csv', 'w') as f:
    f.write('index,xt_p,wt_p\n')
    for i in range(len(deltaV)):
        f.write(f"{i},{xt_p[i]},{wt_p[i]}\n")

sum=np.sum(wt_p)
for i in range(len(deltaV)):
    wdisk_p[i]=wt_p[i]/sum
    if i>0:
        wdisk_p[i]=wdisk_p[i-1]+wdisk_p[i]
    print(f"wdisk_p[{i}]: {wdisk_p[i]}")
   
with open('weights_to_disk.csv', 'w') as f:
    f.write('index,xt_p,wt_p\n')
    for i in range(len(deltaV)):
        f.write(f"{i},{xt_p[i]},{wdisk_p[i]}\n")

pick_disk=np.array([0.78422, 0.41706, 0.55642, 0.77481, 0.57024, 0.2405, 0.35061, 0.93738, 0.33608, 0.87091, 0.10942, 0.61648, 0.86454, 0.29777, 0.49001, 0.07101, 0.39464, 0.61276, 0.57121, 0.13289])
xt_after_resampling=np.zeros((len(deltaV)))
wt_after_resampling=np.zeros((len(deltaV)))
for i in range(len(pick_disk)):
    for j in range(len(wdisk_p)):
        if j>0:
            if pick_disk[i]<wdisk_p[j] and pick_disk[i]>wdisk_p[j-1]:
                xt_after_resampling[i]=xt_p[j]
                wt_after_resampling[i]=wdisk_p[j]
                break
        else:
            if pick_disk[i]<wdisk_p[j]:
                xt_after_resampling[i]=xt_p[j]
                wt_after_resampling[i]=wdisk_p[j]
                break
    print(f"pick_disk[{i}]: {pick_disk[i]}, xt_after_resampling[{i}]: {xt_after_resampling[i]}, wt_after_resampling[{i}]: {wt_after_resampling[i]}")

with open('resampling.csv', 'w') as f:
    f.write('index,xt_after_resampling,wt_after_resampling\n')
    for i in range(len(deltaV)):
        f.write(f"{i},{xt_after_resampling[i]},{wt_after_resampling[i]}\n")