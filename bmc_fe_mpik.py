import cxfit
import pickle

fn = 'fe25_108_beamOFF.mp'
ns1 = [11]
ns2 = [10]
js = [0, 1, 3]
f = open('zfe_mpik_calib.pkl', 'r')
z0 = pickle.load(f)
f.close()
es = z0.rs[0].es
sig = z0.rs[0].sig
wsig = [(sig[i]*0.999,sig[i]*1.001) for i in range(len(sig))]
wes=[(es[i]*0.9999,es[i]*1.0001) for i in range(1,len(es))]
z=cxfit.fit_spec(fn, 26, [1, 2, 2], [ns1,ns2,ns2], js,
                 #[ 59.77338742,0.31329662,0.07111712,0.35941966,0.0],
                 sig,
                 1.0, 2, er=[440.0, 700.0], sav=['zfe_mpik.pkl',500],
                 es = es,
                 #ecf='data/ArK.ecf',
                 #bkgd=(((1e3,1.0,1e10,500),
                 #        (250.0,50.0,500.0,25.0),
                 #        (3e3,2.5e3,3.9e3,1e2)),cxfit.ar_bkgd,0.05),
                 #wsig=[(30.0,90.0),(0.25,1.0),(0.0,0.5),(0.25,1.0)],
                 wsig=wsig,
                 wes=wes,
                 fixnd=[-1,-1,1],
                 kmax=7)

