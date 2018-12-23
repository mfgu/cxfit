import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_filt_ice_corrected_S_S.txt'
ns1 = [7,8,9,10,11]
ns2 = [7,8,9,10,11]
js = [0, 2101,2103,4301,4303,6501,6503,8701,8703,99901,99903]
z=cxfit.fit_spec(fn, 16, [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [ns1,ns2,ns2,ns2,ns2,ns2,ns2,ns2,ns2,ns2,ns2], js, [1.75,0.0], 0.001,0,wsig=[(1.5,2.5)], sav=['zs_s.pkl',500], kmax=10, ierr=[0.05, 0.1])
