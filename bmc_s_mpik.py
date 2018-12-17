import cxfit

fn = 'MPIK-SDD-S-CX.txt'
ns1 = [11]
ns2 = [10]
js = [0, 1, 3]
z=cxfit.fit_spec(fn, 16, [1, 2, 2], [ns1,ns2,ns2], js, [60.0,0.1,0.1,1.0], 0.001,1,sav=['zs_mpik.pkl',500])
