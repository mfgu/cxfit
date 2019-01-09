from pylab import *
from pfac import fac
from pfac.crm import *
import os

def pnk(z, k, sw, nn, sdir, dc):
    z1 = z-k+1
    if k <= 2 or dc > 0:
        e1 = (z1*z1)*13.6*1.5
    elif k <= 10:
        e1 = (z1*z1)*13.6*0.25*1.5
    de=25.0
    a = fac.ATOMICSYMBOL[z]
    ps0 = '%s/%s%02d'%(sdir, a, k)
    clf()
    ylim(-0.02,0.2+0.1*(nn-0.75))
    if dc > 0:
        cm = 'd'
    else:
        cm = 'k'
    for kk in range(nn):
        if sw >= 0:
            ps = '%sn%02d%s%02d'%(ps0, sw*100+nn, cm, kk)
        else:
            ps = '%sm%02d%s%02d'%(ps0, (-sw)*100+nn, cm, kk)
        ofn = ps + 'a.ln'
        os.system('rm -f %s'%ofn)
        SelectLines(ps+'b.sp', ofn, k, 0, 0, e1, 0)
        ofn = ps + 'a.pt'
        os.system('rm -f %s'%ofn)
        PlotSpec(ps+'b.sp', ofn, k, 0, 0, e1, de)    
        s = transpose(loadtxt(ps+'a.pt'))
        plot(1e-3*(s[0]+0.5*(s[0][1]-s[0][0])+kk*100), s[1]*1e5+(kk)*0.1)
        xlabel('Energy (keV)')
        ylabel('Intensity (Arb. Unit)')
    savefig('%sn%02d.eps'%(ps0, sw*100+nn))
    
if __name__ == '__main__':
    dc = 0
    if len(sys.argv) > 6:
        dc = int(sys.argv[6])
    pnk(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], dc)
    
