import cxfit

ft = 'ice_filter_transmission.txt'
fn = 'flux_CX_H2_tot_back_subtract.txt'
ns0 = [4,5,6]
ns1 = [7,8,9]
ns2 = [10,11,12]
js = [-301, -302, -101, -102, 101, 102, 301, 302]
nj = len(js)
ni = nj*3
z=cxfit.fit_spec((fn,ft),28,[10]*ni,([ns0]*nj)+([ns1]*nj)+([ns2]*nj),js*3, [1.8], 0.001,3,sav=['zni_h2.pkl',500])
