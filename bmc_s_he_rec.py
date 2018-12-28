import cxfit
import pickle
from pylab import *

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_He.txt'
ns1 = [6,7,8,9,10]
ns2 = [5,6,7,8,9]
js = [0, 1, 3]
f = open('zs_he.pkl', 'r')
z0 = pickle.load(f)
f.close()
yc = poisson(z0.ym)
yd = yc/z0.sp.eff
ye = sqrt(yc)/z0.sp.eff
z=cxfit.fit_spec((z0.sp.elo, z0.sp.ehi, yd, ye), 16, [1, 2, 2], [ns1,ns2,ns2], js,
                 [1.8], 0.001, 0, wsig=[(1.5,2.5)],
                 ecf='data/SK.ecf',
                 fixnd = [-1, -1, 1],
                 sav=['zs_he_rec.pkl',500])

