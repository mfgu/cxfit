import cxfit

ft = 'FeBeEff.txt'
fn = 'FeHeNewCX.txt'
ns1 = range(7,17)
ns2 = range(7,17)
ns3 = range(3,15)
ns4 = range(3,15)
js = [0, 0, 0, 0]
z=cxfit.fit_spec((fn,ft), 26, [1, 2, 3, 4], [ns1,ns2,ns3,ns4], js,
                 [2.0,0.1,2.0,0.1],
                 0.25, 5, er=[1e3, 2.3e3, 6.5e3, 9.4e3],
                 wsig=[(1.5,3.0),(0.0,0.2),
                       (1.5,3.0),(0.0,0.2)],
                 eip=[2.5e3],
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1, -1, -1],
                 fixld = [-1, -1, -1, -1],
                 sav=['wfe_he.pkl',500],
                 kmax=6, wreg=0, yb=0.5,
                 nmc=5000)
