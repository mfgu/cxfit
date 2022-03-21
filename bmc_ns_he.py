import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_He.txt'
ns1 = [5,6,7,8,9,10]
ns2 = [4,5,6,7,8,9]
z=cxfit.fit_spec(fn, 16, [1, 2], [ns1,ns2], [0,0],
                 [2.0,0.1], 0.1, 0, wsig=[(1.5,3.0),(0.0,0.25)],
                 ecf='data/SK.ecf',
                 eip=[],
                 tgt='He',
                 fixnd = [-1, -1],
                 fixld = [-1, -1],
                 sav=['ns_he.pkl', 500],
                 kmax=5,wreg=0, yb=0.1,
                 nmc=5000)

