import mcpk
import cxfit

odir='/Users/yul20/Ming/papers/cxfit/'
odir='./figures/'

d0 = mcpk.load_pkl('pkf_bof.pkl')
#d1 = mcpk.load_pkl('pkf_ofe.pkl')
zs = cxfit.load_fecx(2)
zn = cxfit.load_fecx(1)

#mcpk.plot_rspec(d0, 0, xr=[6.935e3,6.99e3], sav=odir+'beamon_lya.eps')
#mcpk.plot_rspec(d0, 1, xr=[6.935e3,6.99e3], sav=odir+'beamoff_lya.eps')
#mcpk.plot_widths(sav=odir+'red_ti.eps')

zs1 = zs[:1]
zn1 = zn[:1]
cxfit.plot_hrpoly(zn,
                  (r'Ly$_{\alpha 1}$', 1, [1]),
                  (r'Ly$_{\alpha 2}$', 1, [0]),
                  sav=odir+'poly_lya12.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=3., sav=odir+'poly_highn.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=3., sav=odir+'poly_zhighn.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=1e10, sav=odir+'s3poly_highn.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=1e10, sav=odir+'s3poly_zhighn.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=0., sav=odir+'s1poly_highn.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=0., sav=odir+'s1poly_zhighn.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=-1., sav=odir+'sfpoly_highn.eps')
cxfit.plot_hrpoly(zn,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=-1., sav=odir+'sfpoly_zhighn.eps')
cxfit.plot_hrpoly(zn1,
                  (r'Ly$_{\alpha 1}$', 1, [1]),
                  (r'Ly$_{\alpha 2}$', 1, [0]),
                  sav=odir+'poly_lya12_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=3., sav=odir+'poly_highn_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=3., sav=odir+'poly_zhighn_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=1e10, sav=odir+'s3poly_highn_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=1e10, sav=odir+'s3poly_zhighn_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=0., sav=odir+'s1poly_highn_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=0., sav=odir+'s1poly_zhighn_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=-1., sav=odir+'sfpoly_highn_h2.eps')
cxfit.plot_hrpoly(zn1,
                  (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2,3)),
                  tsr=-1., sav=odir+'sfpoly_zhighn_h2.eps')
cxfit.plot_hr(26, 1, 'H2', 14, md=1, sav=odir+'lya12.eps')
cxfit.plot_hr(26, 1, 'H2', 14, md=0, sav=odir+'hlikeh.eps')
cxfit.plot_hr(26, 2, 'H2', 14, md=0, sav=odir+'helikeh.eps')
cxfit.plot_hrpoly(zs,
                  (r'Ly$_{\alpha 1}$', 1, [1]),
                  (r'Ly$_{\alpha 2}$', 1, [0]),
                  sav=odir+'newpoly_lya12.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=3., sav=odir+'newpoly_highn.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=3., sav=odir+'newpoly_zhighn.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=1e10, sav=odir+'s3newpoly_highn.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=1e10, sav=odir+'s3newpoly_zhighn.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=0., sav=odir+'s1newpoly_highn.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=0., sav=odir+'s1newpoly_zhighn.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=-1., sav=odir+'sfnewpoly_highn.eps')
cxfit.plot_hrpoly(zs, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=-1., sav=odir+'sfnewpoly_zhighn.eps')
cxfit.plot_hrpoly(zs1,
                  (r'Ly$_{\alpha 1}$', 1, [1]),
                  (r'Ly$_{\alpha 2}$', 1, [0]),
                  sav=odir+'newpoly_lya12_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=3., sav=odir+'newpoly_highn_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=3., sav=odir+'newpoly_zhighn_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=1e10, sav=odir+'s3newpoly_highn_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=1e10, sav=odir+'s3newpoly_zhighn_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=0., sav=odir+'s1newpoly_highn_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=0., sav=odir+'s1newpoly_zhighn_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like n>9', 2, range(11, 18)),
                  tsr=-1., sav=odir+'sfnewpoly_highn_h2.eps')
cxfit.plot_hrpoly(zs1, (r'H-like n>9', 1, range(9,16)),
                  (r'He-like z', 2, range(2, 3)),
                  tsr=-1., sav=odir+'sfnewpoly_zhighn_h2.eps')
cxfit.plot_ovsp(zs, sav=odir+'spec_ov.eps')
cxfit.plot_feld(zn, k=1, sav=odir+'lo_hlike.eps')
cxfit.plot_feld(zn, k=2, sav=odir+'lo_helike.eps')
cxfit.plot_feld(zs, k=1, sav=odir+'new_ldist.eps')
cxfit.plot_fend(zs, k=1, sav=odir+'newn_hlike.eps')
cxfit.plot_fend(zs, k=2, sav=odir+'newn_helike.eps')
cxfit.plot_feld(zn1, k=1, sav=odir+'lo_hlike_h2.eps')
cxfit.plot_feld(zn1, k=2, sav=odir+'lo_helike_h2.eps')
cxfit.plot_feld(zs1, k=1, sav=odir+'new_ldist_h2.eps')
cxfit.plot_fend(zs1, k=1, sav=odir+'newn_hlike_h2.eps')
cxfit.plot_fend(zs1, k=2, sav=odir+'newn_helike_h2.eps')
cxfit.plot_ce(zs, sav=odir+'pdf_ce.eps')
cxfit.plot_ce(zs1, sav=odir+'pdf_ce_h2.eps')
cxfit.plot_nc(zs, sav=odir+'nc_ce.eps')
cxfit.plot_kbs(zs[0], 0, sav=odir+'h2fit_hlike.eps')
cxfit.plot_kbs(zs[0], 1, sav=odir+'h2fit_helike.eps')
cxfit.plot_kbs(zs[3], 0, sav=odir+'hefit_hlike.eps')
cxfit.plot_kbs(zs[3], 1, sav=odir+'hefit_helike.eps')
cxfit.plot_tsr(zs, sav=odir+'tsr.eps')
cxfit.plot_cecmp(zs, sav=odir+'cecmp.eps')

cxfit.plot_hp(zs[0], -1)
cxfit.plot_hp(zn[0], -1, op=1)
cxfit.xlabel('Log[Likelyhood]')
cxfit.ylabel('Relative Probability')
cxfit.legend(['New Cascade Model Fe+H2', 'Old Cascade Model Fe+H2'])
cxfit.savefig(odir+'loglikely_h2.eps')
