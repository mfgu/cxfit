import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_S.txt'
ns1 = [7,8,9,10,11,12]
ns2 = [6,7,8,9,10,11]
z=cxfit.fit_spec(fn, 16, [1, 2], [ns1,ns2], [0,0],
                 [2.0,0.1,0.5], 0.1, 0, wsig=[(1.5,3.0),(0.0,0.25),(-2.0,2.0)],
                 ecf='data/SK.ecf',
                 eip=[-1],
                 tgt='S',
                 fixnd = [-1, -1],
                 fixld = [-1, 0],
                 sav=['ks_s.pkl', 500],
                 kmax=5,wreg=0, yb=0.1,
                 nmc=5000)

