import sys, os

z = int(sys.argv[1])
k = int(sys.argv[2])
n0 = int(sys.argv[3])
n1 = int(sys.argv[4])
sm = int(sys.argv[5])
sdir = sys.argv[6]

for n in range(n0, n1+1):
    if sm == 0:
        os.system('python s.py --z=%d --k=%d --md=%d%02d --sdir=%s'%(z,k,n,n,sdir))
        os.system('python pnk.py %d %d 0 %d %s'%(z, k, n, sdir))
    elif sm == 1 or sm == 2:
        if sm == 2:
            swm = 4
        else:
            swm = 0
        for m in [1, 3]:
            os.system('python s.py --z=%d --k=%d --swm=%d --md=%d%02d%02d --sdir=%s'%(z, k, swm, m, n, n, sdir))
            os.system('python pnk.py %d %d %d %d %s'%(z, k, m, n, sdir))
    elif sm == 3:
        swm = 5
        for m in [101, 103, 701, 703]:
            os.system('python s.py --z=%d --k=%d --swm=%d --md=%d%02d%02d --sdir=%s'%(z, k, swm, m, n, n, sdir))
            os.system('python pnk.py %d %d %d %d %s'%(z, k, m, n, sdir))
    elif sm == 4:
        swm = 6
        for m in [2101,2103,4301,4303,6501,6503,8701,8703,99901,99903]:
            os.system('python s.py --z=%d --k=%d --swm=%d --md=%d%02d%02d --sdir=%s'%(z, k, swm, m, n, n, sdir))
            os.system('python pnk.py %d %d %d %d %s'%(z, k, m, n, sdir))
    elif sm == 5:
        for m in [1, 2]:
            for j in [-3, -1, 1, 3]:
                jm = abs(j)*100 + m
                if j < 0:
                    jm = -jm
                os.system('python s.py --z=%d --k=%d --swm=3 --md=%d%02d%02d --sdir=%s'%(z, k, jm, n, n, sdir))
                os.system('python pnk.py %d %d %d %d %s'%(z, k, jm, n, sdir))
    
