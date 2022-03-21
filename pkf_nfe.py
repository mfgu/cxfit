from mcpk import *

d = run_mcpk(5000, ['FeN2NewCX.txt'],
             feff='FeBeEff.txt', emin=6.6e3, emax=9.5e3,
             lns=['spec1/Fe01a.ln','spec1/Fe02a.ln'],
             sav=['pkf_nfe.pkl', 500], fixlam=0,
             npr=1, xb=4, fixwd2=1.515, fixwd3=0.1099, nwp=4)
                  
