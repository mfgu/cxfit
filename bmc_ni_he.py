import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_CX_He_tot_back_subtract.txt'
js = [1, 3]
ns = [list(range(6,13))]*len(js)
ks = [10]*len(js)
z=cxfit.fit_spec((fn,ft), 28, ks, ns, js, [1.8], 5e-4, 3,
                 ecf='data/Ni10a.ecf',
                 sdir='spec',
                 kmax=7,
                 wsig=[(1.7,1.9)],
                 sav=['zni_he.pkl',500])
