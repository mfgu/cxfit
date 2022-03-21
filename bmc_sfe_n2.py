import cxfit

ft = 'FeBeEff.txt'
fn = 'FeN2NewCX.txt'
ns1 = range(7,17)
ns2 = range(7,17)
#ns3 = range(3,15)
#ns4 = range(3,15)
js = [0, 1, 3]
wp = cxfit.loadtxt('xrs_n2.txt', unpack=1)
wp = list(wp[0][[6,5,7,8,9]])
ws = [(x,x) for x in wp]
wp += [0.5]
ws += [(-2.0,2.0)]
print(wp)
z=cxfit.fit_spec((fn,ft), 26, [1, 2, 2], [ns1,ns2,ns2], js, wp,
                 0.25, 5, er=[2e3,2.3e3,6.5e3, 9.4e3], wsig=ws,
                 eip=[-1],
                 tgt='N2',
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1, 1],
                 fixld = [-1, 0, 0],
                 sav=['sfe_n2.pkl',500],
                 kmax=5, wreg=0, yb=0.5,
                 nmc=5000)
