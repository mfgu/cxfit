from mcpk import *

d = run_mcpk(5000, ['beamon.txt','beamoff.txt'],
             sav=['pkf_bof.pkl', 500], fixlam=1,
             #lns=['spec1/Fe01a.ln','spec1/Fe02a.ln'],
             npr=100, xb=1, emin=6.8e3, emax=7.125e3, nwp=5)
                  
