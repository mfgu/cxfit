from mcpk import *


d = run_mcpk(5000, ['FeH2NewCX.txt','FeHNewCX.txt',
                    'FeN2NewCX.txt','FeHeNewCX.txt'],
             feff='FeBeEff.txt', emin=6.6e3, emax=9.5e3,
             lns=['spec1/Fe01a.ln','spec1/Fe02a.ln'],
             sav=['pkf_fe.pkl', 500], fixlam=0,
             npr=1, xb=4, nwp=5,
             rsp=['xrs_h2.txt','xrs_h.txt','xrs_n2.txt','xrs_he.txt'])
                  
