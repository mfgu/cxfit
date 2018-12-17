import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_CX_H2_tot_back_subtract.txt'
js = [-301, 101, 301, [-302,-101,-102,102,302]]
ns = [list(range(6,13))]*len(js)
ks = [10]*len(js)
z=cxfit.fit_spec((fn,ft), 28, ks, ns, js, [1.8], 5e-4, 3,
                 ecf='spec_sp/Ni10a.ecf',
                 sdir='spec',
                 kmax=6,
                 wsig=[(1.7,1.9)],
                 sav=['znic_h2.pkl',500])
