import cxfit

ft = 'FeCXEff.txt'
fn = 'FeH2CX.txt'
ns1 = [9,10,11,12,13,14,15,16]
ns2 = [8,9,10,11,12,13,14,15]
ns3 = [2,3,4]
js = [0, 1, 3]
z=cxfit.fit_spec((fn,ft), 26, [1, 2, 2], [ns1,ns2,ns2], js,
                 [4.0,3.0,3.0,3.0,3.0,3.0,3.0],
                 0.1, 5, er=[2.034e3, 2.3e3, 6.5e3, 9.4e3],
                 wsig=[(2.5,15.0),(2.5,5.0),(2.5,5.0),
                       (2.5,5.0),(2.5,5.0),(2.5,5.0),(2.5,5.0)],
                 eip=[2.5e3,6.65e3, 6.67e3, 6.85e3, 7.5, 8.5e3],
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1, 1],
                 fixld = [-1, -1, 1],
                 sav=['ufe_h2.pkl',500],
                 fixwk=[(0,2,0.3),(1,2,0.3),(2,2,0.3)],
                 nmc=5000)

