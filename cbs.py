from cxfit import *
import sys
from pfac import fac

z = int(sys.argv[1])
k = int(sys.argv[2])
tgt = sys.argv[3]
s = int(sys.argv[4])
n0 = 6
n1 = 16
e0 = 5e2
e1 = 1e4
a = fac.ATOMICSYMBOL[z]
fsav = '%s%02d%sKL%d.pkl'%(a, k, tgt, s)
ssp_basis(z, k, e0, e1, n0, n1, 5, 20, 0.01, 100.0,
          fsav, sw=s,
          ddir='data', sdir='spec',
          tgt=tgt, ecx=0, rxf=1.0)
