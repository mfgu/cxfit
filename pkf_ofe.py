from mcpk import *

wp = loadtxt('xrsprof.txt', unpack=1)

d = run_mcpk(5000, ['beamonH2.txt','beamonH.txt',
                    'beamonN2.txt','beamonHe.txt',
                    'FeH2NewCX.txt','FeHNewCX.txt',
                    'FeN2NewCX.txt','FeHeNewCX.txt'],
             feff='FeBeEff.txt', emin=6.8e3, emax=7.125e3,
             #lns=['spec1/Fe01a.ln','spec1/Fe02a.ln'],
             sav=['pkf_ofe.pkl', 500], fixlam=1,
             npr=100, xb=1,
             fixwd0=wp[0][0],
             fixwd3=wp[0][3], nwp=5)
                  
