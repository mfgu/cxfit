from pfac.fac import *
import sys
import os

kd = '/Users/yul20/src/Kronos_v3.1'
for t in ["H", "He", "H2", "H2O", "CO", "CO2", "O2", "N2", "Ne", "Ar", "Kr", "Xe", "CH4", "C2H6O", "O", "F", "S" , "CS2"]:
    SetCXTarget(t)
    PrintCXTarget()
    e = [10**(x*0.1-3.0) for x in range(81)]
    for z in range(2,31):
        print(z)
        a = ATOMICSYMBOL[z]
        fd = "%s/CXDatabase/Projectile_Ions/%s/Charge/%d/Targets/%s"%(kd,a,z,t)
        print(fd)
        if not os.path.exists(fd):
            os.system('mkdir -p %s'%fd)
        f1 = '%s/%s%d+%s_sec_faclz_nres.cs'%(fd,a.lower(),z,t.lower())
        print(f1)
        f = '%s/%s%d+%s_sec_faclz.cs'%(fd,a.lower(),z,t.lower())
        print(f)
        LandauZenerCX(f1, z, e)
        LandauZenerCX(f, z, e, 5)

