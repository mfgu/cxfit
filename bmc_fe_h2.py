import cxfit

ft = 'FeCXEff.txt'
fn = 'FeH2CX.txt'
ns1 = [9,10,11,12,13,14,15]
ns2 = [8,9,10,11,12,13,14]
js = [0, 1, 3]
z=cxfit.fit_spec((fn,ft), 26, [1, 2, 2], [ns1,ns2,ns2], js,
                 [1.8], 0.001, 5, wsig=[(1.5,2.5)],
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1, 1],
                 fixld = [-1, -1, 1],
                 sav=['zfe_h2.pkl',500],
                 ierr=0.25,
                 nmc=5000)

