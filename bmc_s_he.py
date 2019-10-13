import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_He.txt'
ns1 = [5,6,7,8,9,10]
ns2 = [4,5,6,7,8,9]
js = [0, 1, 3]
z=cxfit.fit_spec(fn, 16, [1, 2, 2], [ns1,ns2,ns2], js,
                 [1.8,0.0], 0.001, 0, wsig=[(1.5,2.5),(-1,1)],
                 ecf='data/SK.ecf',
                 fixnd = [-1, -1, 1],
                 sav=['zs_he.pkl',500])

