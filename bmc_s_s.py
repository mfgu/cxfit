import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_S.txt'
ns1 = [7,8,9,10,11]
ns2 = [6,7,8,9,10]
js = [0, 1, 3]
z=cxfit.fit_spec(fn, 16, [1, 2, 2], [ns1,ns2,ns2], js, [1.75], 0.001,0,
                 wsig=[(1.7,1.8)],
                 #ecf='data/SK.ecf',
                 fixnd = [-1, -1, 1],
                 sav=['zs_s.pkl',500])
