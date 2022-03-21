from mcpk import *

d = run_mcpk(5000, ['SHeCX.txt'],
             feff='SHeEff.txt', emin=2.35e3, emax=3.5e3,
             lns=['spec1/S01a.ln','spec1/S02a.ln'],
             sav=['pkf_s.pkl', 500], fixlam=0,
             npr=1, xb=4, nwp=2)
                  
