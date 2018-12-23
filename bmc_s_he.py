import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_He.txt'
ns1 = [6,7,8,9,10]
ns2 = [6,7,8,9,10]
js = [0, 1, 3]
z=cxfit.fit_spec(fn, 16, [1, 2, 2], [ns1,ns2,ns2], js, [1.8], 0.001,0,wsig=[(1.5,2.5)], sav=['zs_he.pkl',500], ierr=[0.05,0.1])
