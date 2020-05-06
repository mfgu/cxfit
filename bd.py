import os

e0=2.0
e1=8.0
n = 20
es = [10**(e0+i*(e1-e0)/(n-1)) for i in range(n)]
for k in [1,2]:
    for i in range(n):
        c = 'python d.py --z=26 --k=%d --np=8 --nm=18 --mm=1 --km=5 --bea=0.0 --bf=4e4 --ef=%.6e --pf=F%02d'%(k,es[i],i)
        print(c)
        os.system(c)
    
