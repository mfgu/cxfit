import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_CX_H2_tot_back_subtract.txt'
js = [1, 3]
ns = [list(range(6,13))]*len(js)
ks = [10]*len(js)
z=cxfit.fit_spec((fn,ft), 28, ks, ns, js, [1.8], 5e-4, 3,
                 ecf='data/Ni10a.ecf',
<<<<<<< HEAD
                 sdir='spec2',
                 kmax=7,
                 wsig=[(1.7,1.9)],
                 sav=['zni_h2.pkl',500])
=======
                 sdir='spec',
                 kmax=6,
                 wsig=[(1.5,2.5)],
                 sav=['zni_h2.pkl',500],
                 fixnd=[-1,0],
                 fixld=[-1,0],
                 wreg=0,
                 nmc=5000)
>>>>>>> 5ea1b7306447a09e021ca3225187d1dc2c442443
