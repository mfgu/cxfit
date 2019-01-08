from pfac.crm import *
from pfac import fac
import sys
import os
from optparse import OptionParser

ap = OptionParser()
ap.add_option('--z', dest='z', type='int', help='atomic number')
ap.add_option('--k', dest='k', type='int', help='number of electrons')
ap.add_option('--dn', dest='dn', default='/Users/yul20/atomic/chintan/Kronos_v3.1', type='string', help='kronos path')
ap.add_option('--tgt', dest='tgt', type='string', default='H2', help='neutral target')
ap.add_option('--ldist', dest='ldist', type='int', default=5, help='LDist option')
ap.add_option('--md', dest='md', type='int', default=2, help='mode')
ap.add_option('--swm', dest='swm', type='int', default=0, help='mode')
ap.add_option('--qd', dest='qd', type='int', default=0, help='qd interp cx')
ap.add_option('--cxm', dest='cxm', type='string', default='', help='CX method')
ap.add_option('--e', dest='e', type='float', default=500, help='ion energy')
ap.add_option('--de', dest='de', type='float', default=1, help='ion energy width')
ap.add_option('--em', dest='em', type='int', default=7, help='ion energy dist')
ap.add_option('--eu', dest='eu', type='int', default=0, help='ion energy unit')
ap.add_option('--scrm', dest='scrm', type='string', default='', help='scrm conversion')
ap.add_option('--dk', dest='dk', type='int', default=0, help='zeff adjust')
ap.add_option('--sdir', dest='sdir', type='string', default='spec', help='spec out dir')
ap.add_option('--exc', dest='exc', type='int', default=0, help='do collisional excitation model')

(opts, args) = ap.parse_args()
print(opts)

if (opts.scrm != ''):
    ConvertToSCRM(opts.scrm)
z = opts.z
k = opts.k
dn = opts.dn
tgt = opts.tgt

a = fac.ATOMICSYMBOL[z]
os.system('mkdir spec')
p = '%s%02d'%(a, k)
pd = 'data/'+p
ps = '%s/%s'%(opts.sdir,p)
z1 = z-k+1+opts.dk
a1 = fac.ATOMICSYMBOL[z1]
md = 0
if opts.exc > 0 and md < 100:
    md = 100
if (abs(opts.md) >= 100):
    md = opts.md
elif (opts.md == 2):
    md = 2
elif (opts.md > 2):
    if k <= 2:
        ReadKronos(dn, z, k-1, a, tgt, opts.cxm, opts.ldist)
    else:
        ReadKronos(dn, z1, 0, a1, tgt, opts.cxm, opts.ldist)
        md = 1
elif (opts.md == 0):
    ReadKronos(dn, z, k-1, a, tgt, opts.cxm, opts.ldist)
else:
    ReadKronos(dn, z1, 0, a1, tgt, opts.cxm, opts.ldist)
    md = 1

SetOption('crm:sw_mode', opts.swm)

WallTime('addion')
AddIon(k, 0.0, pd+'b')
if opts.exc > 0:
    SetBlocks(-1)
    if opts.em > 0:
        if opts.em == 1:
            SetEleDist(1, opts.e, min(0.01*opts.e,opts.de), -1, -1)
        else:
            SetEleDist(7, opts.e, min(0.01*opts.e,opts.de), -1, -1)
    else:
        SetEleDist(0, opts.e, -1, -1)
else:
    SetBlocks(1.0)
    if opts.em > 0:
        if opts.em == 1:
            SetCxtDist(1, opts.e, min(0.01*opts.e,opts.de), -1, -1)
        else:
            SetCxtDist(7, opts.e, min(0.01*opts.e,opts.de), -1, -1)
    else:
        SetCxtDist(0, opts.e, -1, -1)
WallTime('tr')
SetTRRates(0)
if k == 10:
    ModifyRates(pd+'BTR.txt')
WallTime('cx')
if md == 1 and opts.qd == 1:
    md = 2
if md < 10:
    md = opts.eu*10+md
kk0 = -1
kk1 = -1
nn = -1
kk = -2
sw = 0
amd = abs(md)
if amd >= 10000:
    sw = amd/10000
    md = amd%10000
if md >= 100:
    kk = md%100
    nn = md/100
if kk == nn:
    kk0 = 0
    kk1 = nn-1
else:
    kk0 = kk
    kk1 = kk
ps0 = ps
for kk in range(kk0, kk1+1):
    if nn > 0:
        if opts.md >= 0:
            md = sw*10000 + nn*100+kk        
            ps = '%sn%02dk%02d'%(ps0, sw*100+nn, kk)
        else:
            md = -(sw*10000 + nn*100+kk)
            ps = '%sm%02dk%02d'%(ps0, sw*100+nn, kk)
    if opts.exc == 0:
        SetCXRates(md, opts.tgt)
        SetCxtDensity(1e-5)
    else:
        SetCERates(1)
        SetAbund(k, 1.0)
        SetEleDensity(1.0)
    WallTime('pop')
    InitBlocks()
    DumpRates(ps+'a.rm', 0, 0, -1, 1)
    DumpRates(ps0+'a.r1', k, 1, -1, 1)
    if opts.exc == 0:
        DumpRates(ps+'a.r7', k, 7, -1, 1)
    else:
        DumpRates(ps+'a.r3', k, 3, -1, 1)
    SetIteration(1e-5, 0.5)
    LevelPopulation()
    
    SpecTable(ps+'b.sp')
    PrintTable(ps+'b.sp', ps+'a.sp')
    DumpRates(ps+'a.r0', k, 0, -1, 1)
    ReinitCRM(2)

if (opts.scrm != ''):
    CloseSCRM()
