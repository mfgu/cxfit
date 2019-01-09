from pfac.fac import *
import sys
import os
from optparse import OptionParser

ap = OptionParser()
ap.add_option('--z', dest='z', type='int', help='atomic number')
ap.add_option('--k', dest='k', type='int', help='number of electrons')
ap.add_option('--nm', dest='nm', type='int', default=15, help='max n')
ap.add_option('--mm', dest='mm', type='int', default=-1, help='max n for full mixing')
ap.add_option('--nc', dest='nc', type='int', default=0, help='max n for collisional excitation')
ap.add_option('--ni', dest='ni', type='int', default=0, help='max n for collisional ionization')
ap.add_option('--nr', dest='nr', type='int', default=0, help='max n for radiative recombination')
ap.add_option('--nd1', dest='nd1', type='int', default=0, help='max n1 for double capture')
ap.add_option('--nd2', dest='nd2', type='int', default=0, help='max n2 for double capture')
ap.add_option('--sfac', dest='sfac', type='string', default='', help='convert to sfac input')
ap.add_option('--tgt', dest='tgt', type='string', default='H,H2,He,CO,CO2,N2,H2O,CH4,O2', help='cx target')
ap.add_option('--ldist', dest='ldist', type='int', default=5, help='ldist type')
ap.add_option('--ldistmj', dest='ldistmj', type='float', default=-1.0, help='ldist mj')

(opts, args) = ap.parse_args()
print(opts)
if opts.sfac != '':
    ConvertToSFAC('d.sf')
z = opts.z
k = opts.k
nm = opts.nm
nc = opts.nc
nm1 = nm+1
a = ATOMICSYMBOL[z]
SetAtom(a)
os.system('mkdir data')
p = 'data/%s%02d'%(a, k)
pj = '%sLSJ'%p
os.system('mkdir %s'%pj)
PrintNucleus(1, pj+'/isodata')
if k <= 2:
    nmin=1
    Config('g1', '1s%d'%k)
    for n in range(2,nm1):
        if (k == 2):
            Config('g%d'%n, '1s1 %d*1'%n)
        else:
            Config('g%d'%n, '%d*1'%n)
    if k == 2:
        Config('i1', '1s1')
    else:
        Config('i1', '')
elif k <= 10:
    nmin=2
    Config('g2', '1s2 2*%d'%(k-2))
    for n in range(3, nm1):
        Config('g%d'%n, '1s2 2*%d %d*1'%(k-3,n))
    Config('i1', '1s2 2*%d'%(k-3))
    
WallTime('OPT')
#SetPotentialMode(20)
gs=[]
ga=[]
gd=[]
if opts.mm > 0:
    mm1 = opts.mm + 1
elif opts.mm < 0:
    mm1 = nm1
else:
    mm1 = nmin+1
for n in range(nmin, mm1):
    gs.append('g%d'%n)
    ga.append('g%d'%n)
for n in range(mm1, nm1):
    ga.append('g%d'%n)

for n1 in range(2, opts.nd1+1):
    for n2 in range(n1, opts.nd2+1):
        if k < 2 or k > 3:
            continue
        gd.append('d.%d.%d'%(n1,n2))
        if k == 2:
            if n1 == n2:
                Config(gd[-1], '%d*2'%n1)
            else:
                Config(gd[-1], '%d*1 %d*1'%(n1,n2))
        else:
            if n1 == n2:
                Config(gd[-1], '1s1 %d*2'%n1)
            else:
                Config(gd[-1], '1s1 %d*1 %d*1'%(n1,n2))

ConfigEnergy(0)
if k <= 2:
    OptimizeRadial(['g1','g2'])
else:
    OptimizeRadial(ga)
ConfigEnergy(1)
GetPotential(p+'a.pot')

WallTime('EN')
Structure(p+'b.en', gs)
for n in range(mm1, nm1):
    Structure(p+'b.en', ['g%d'%n])
for d in gd:
    Structure(p+'b.en', [d])
Structure(p+'b.en', ['i1'])
MemENTable(p+'b.en')
PrintTable(p+'b.en', p+'a.en')
BasisTable(pj+'/basis', 10)
BasisTable(pj+'/fb.txt')
os.system('./jj2lsj.sh %s %d'%(a, k))
if k > 1:
    os.system('./jj2lsj.sh %s %d basis_001'%(a, k))

WallTime('TR')
TRTable(p+'b.tr', ga, ga)
if len(gd) > 0:
    TRTable(p+'b.tr', ga, gd)
    TRTable(p+'b.tr', gd, gd)    
PrintTable(p+'b.tr', p+'a.tr')

if len(gd) > 0:
    WallTime('AI')
    AITable(p+'b.ai', gd, ['i1'])
    PrintTable(p+'b.ai', p+'a.ai')
    
if opts.nc >= nmin:
    WallTime('CE')
    for n in range(nmin, opts.nc+1):
        CETable(p+'b.ce', ['g%d'%nmin], ['g%d'%n])
    PrintTable(p+'b.ce', p+'a.ce')
if opts.ni >= nmin:
    WallTime('CI')
    for n in range(nmin, opts.ni+1):
        CITable(p+'b.ci', ['g%d'%n], ['i1'])
    PrintTable(p+'b.ci', p+'a.ci')
if opts.nr >= nmin:
    WallTime('RR')
    for n in range(nmin, opts.nr+1):
        RRTable(p+'b.rr', ['g%d'%n], ['i1'])
    PrintTable(p+'b.rr', p+'a.rr')

WallTime('RO')
for n in range(nmin, nm1):
    RecOccupation(p+'b.ro', ['g%d'%n], ['i1'])
PrintTable(p+'b.ro', p+'a.ro')

if opts.tgt != '':
    if opts.ldist >= 0:
        SetOption('recombination:cxldist', opts.ldist)
    if opts.ldistmj > 0:
        SetOption('recombination:cxldistmj', opts.ldistmj)
    for t in opts.tgt.split(','):
        WallTime('CX: %s'%t)
        SetCXTarget(t)
        for n in range(nmin, nm1):
            CXTable(p+'b.cx', ['g%d'%n], ['i1'])
    PrintTable(p+'b.cx', p+'a.cx')

ReinitDBase(0)
MemENTable(p+'b.en')
WallTime('SRO')
RecoupleRO(p+'b.ro', p+'b.sro', p+'i.LS')
PrintTable(p+'b.sro', p+'a.sro')

if (opts.sfac != ''):
    CloseSFAC()
