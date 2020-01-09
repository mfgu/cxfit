# cxfit
scripts for generating cascade model and fitting charge exchange X-ray spectra

steps for running a fit:  
1. prepare FAC atomic data  
python d.py --z=16 --k=1 #for H-like S  
python d.py --z=16 --k=2 #for He-like S  

2. prepare cascade model:  
python bs.py 16 1 4 12 0 spec #for H-like S  
python bs.py 16 2 4 12 2 spec #for He-like S, split singlet and triplet states with CI

3. run the fit  
for the LLNL XRS data of S + He CX. it needs the transmission corrected flux data file  
python bmc_s_he.py  

4. results are saved in python pickle format, in python interactive session:  
import cxfit  
import pickle  
from pylab import *  
z = cxfit.load_pkl('zs_he.pkl')  
f.close()  
ion()  
cxfit.plot_spec(z) #shows the fitted spec  
cxfit.plot_snk(z) #shows the N/L distributions  
