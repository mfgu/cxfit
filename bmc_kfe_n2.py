import cxfit

ft = 'FeBeEff.txt'
fn = 'FeN2NewCX.txt'
ns1 = range(7,17)
ns2 = range(7,17)
#ns3 = range(3,15)
#ns4 = range(3,15)
js = [0, 0]
z=cxfit.fit_spec((fn,ft), 26, [1, 2], [ns1,ns2], js,
                 [2.74,0.16,1.5,0.11,0.5],
                 0.25, 5, er=[2e3,2.3e3,6.5e3, 9.4e3],
                 wsig=[(2.74,2.74),(0.16,0.16),(1.5,1.5),(0.11,0.11),(-2.,2.0)],
                 eip=[-1],
                 tgt='N2',
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1],
                 fixld = [-1, 0],
                 sav=['kfe_n2.pkl',500],
                 kmax=5, wreg=0, yb=0.5,
                 nmc=5000)
