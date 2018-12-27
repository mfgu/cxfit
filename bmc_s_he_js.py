import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_He.txt'
ns1 = [6,7,8,9,10]
ns2 = [5,6,7,8,9]
js = [0, 2101,2103,4301,4303,6501,6503,8701,8703,99901,99903]
fixnd = [-1, -1, 1, -1, 3, -1, 5, -1, 7, -1, 9]
z=cxfit.fit_spec(fn, 16, [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [ns1,ns2,ns2,ns2,ns2,ns2,ns2,ns2,ns2,ns2,ns2], js, [1.75,0.0], 0.001,0,wsig=[(1.5,2.5)], sav=['zs_he_js.pkl',500], fixnd=fixnd)
