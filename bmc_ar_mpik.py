import cxfit
import pickle

fn = 'Ar_CX_6keV_Raw.dat'
ns1 = [9]
ns2 = [8]
js = [0, 1, 3]
z=cxfit.fit_spec(fn, 18, [1, 2, 2], [ns1,ns2,ns2], js,
                 [ 59.77338742,0.31329662,0.07111712,0.35941966,0.0],
                 1.0, 2,sav=['zar_mpik.pkl',500],
                 ecf='data/ArK.ecf',
                 bkgd=(((1e3,1.0,1e10,500),
                         (250.0,50.0,500.0,25.0),
                         (3e3,2.5e3,3.9e3,1e2)),cxfit.ar_bkgd,0.05),
                 wsig=[(30.0,70.0),(0.25,0.5),(0.0,0.2),(0.25,0.5)],
                 fixnd=[-1,0,0],
                 kmax=7)

