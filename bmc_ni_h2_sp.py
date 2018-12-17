import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_CX_H2_tot_back_subtract.txt'
js = [1, 3]
ns = [list(range(6,11))]*len(js) + [[11],[11]]
ks = [10]*len(js) + [10,10]
km = [8]*len(js) + [8,8]
js = js + [1, 3]
z=cxfit.fit_spec((fn,ft), 28, ks, ns, js, [1.8], 5e-4, 3,
                 ecf='spec_sp/Ni10a.ecf',
                 sdir='spec_sp',
                 kmax=km,
                 wsig=[(1.7,1.9)],
                 sav=['znis_h2.pkl',500])
