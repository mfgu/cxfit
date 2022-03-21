import cxfit

ft = 'FeCXEff.txt'
fn = 'FeHeNewCX.txt'
ns1 = [9,10,11,12,13,14]
ns2 = [8,9,10,11,12,13]
js = [0, 1, 3]
z=cxfit.fit_spec((fn,ft), 26, [1, 2, 2], [ns1,ns2,ns2], js,
                 [1.8], 0.001, 5, wsig=[(1.5,2)],
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1, 1],
                 fixld = [-1, -1, 1],
                 sav=['zfe_he.pkl',500],
                 nmc=5000)

