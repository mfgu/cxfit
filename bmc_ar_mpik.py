import cxfit

fn = 'Ar_CX_6keV_Raw.dat'
ns1 = [9]
ns2 = [8]
js = [0, 1, 3]
z=cxfit.fit_spec(fn, 18, [1, 2, 2], [ns1,ns2,ns2], js, [60.0,0.1,0.1,1.0, 0.0], 0.01, 2,sav=['zar_mpik.pkl',500])
