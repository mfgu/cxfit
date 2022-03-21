from mcpk import *

wp = loadtxt('xrsprof.txt', unpack=1)

d = run_mcpk(5000, ['beamonHe.txt', 'FeHeNewCX.txt'],
             feff='FeBeEff.txt', emin=6.8e3, emax=7.125e3,
             #lns=['spec1/Fe01a.ln','spec1/Fe02a.ln'],
             sav=['pkf_fe_he.pkl', 500], fixlam=1,
             npr=100, xb=1,
             fixwd3=wp[0][3], nwp=5)
                  
