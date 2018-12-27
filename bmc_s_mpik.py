import cxfit

fn = 'MPIK-SDD-S-CX.txt'
ns1 = [10]
ns2 = [9]
js = [0, 1, 3]
z=cxfit.fit_spec(fn, 16, [1, 2, 2], [ns1,ns2,ns2], js,
                 [ 59.77338742,0.31329662,0.07111712,0.35941966,0.0],
                 1.0, 1,
                 sav=['zs_mpik.pkl',500],
                 ecf='data/SK.ecf',
                 bkgd=(((1e-3,0.0,1e10,1e-4),
                         (100.0,50.0,500.0,10.0),
                         (2.4e3,2e3,4.0e3,1e2)),cxfit.ar_bkgd,0.05),
                 wsig=[(30.0,70.0),(0.25,0.5),(0.0,0.2),(0.25,0.5)],
                 fixnd=[-1,-1,1],
                 kmax=7)
