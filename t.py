from pfac.crm import *
from pfac import fac
import sys
import os

z = int(sys.argv[1])
k = int(sys.argv[2])
de = 4.5
if len(sys.argv) > 3:
    de = float(sys.argv[3])
if len(sys.argv) > 4:
    sdir = sys.argv[4]
else:
    sdir = 'spec'
a = fac.ATOMICSYMBOL[z]
p = '%s%02d'%(a, k)
ps = '%s/%s'%(sdir,p)
z1 = z-k+1
if k <= 2:
    e1 = (z1*z1)*13.6*1.5
elif k <= 10:
    e1 = (z1*z1)*13.6*0.25*1.5
Print('emax: %11.4E'%e1)
ofn = ps + 'a.ln'
os.system('rm -f %s'%ofn)
SelectLines(ps+'b.sp', ofn, k, 0, 0, e1, 1e-4)
ofn = ps + 'a.pt'
os.system('rm -f %s'%ofn)
PlotSpec(ps+'b.sp', ofn, k, 0, 0, e1, de)

