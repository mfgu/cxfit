import cxfit

ft = 'FeBeEff.txt'
fn = 'FeH2NewCX.txt'
ns1 = range(7,17)
ns2 = range(7,17)
#ns3 = range(3,15)
#ns4 = range(3,15)
js = [0, 0]
z=cxfit.fit_spec((fn,ft), 26, [1, 2], [ns1,ns2], js,
                 [2.8,0.05,0.5,0.5,0.5],
                 0.25, 5, er=[2e3,2.3e3,6.5e3, 9.4e3],
                 wsig=[(2.0,3.0),(0.0,0.25),(-2.0,2.0),(-2.0,2.0),(-2.0,2.0)],
                 eip=[-13],
                 tgt='H2',
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1],
                 fixld = [-1, 0],
                 sav=['efe_h2a.pkl',500],
                 kmax=5, wreg=0, yb=0.5,
                 nmc=5000)
