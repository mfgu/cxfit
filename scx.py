from pfac.crm import *
import os

kronos_path=os.getenv('HOME')+'/src/Kronos_v3.1'
ReadKronos(kronos_path, 8, 1, 'O', 'H2', '', 5)
WallTime('addion')
AddIon(2, 0.0, 'data/O02b')
SetBlocks(1.0)
SetCxtDist(7, 50.0, 0.5, -1, -1)
WallTime('tr')
SetTRRates(0)
WallTime('cx')
SetCXRates(10, 'H2')
SetCxtDensity(1e-05)
WallTime('pop')
InitBlocks()
DumpRates('spec/O02a.rm', 0, 0, -1, 1)
DumpRates('spec/O02a.r1', 2, 1, -1, 1)
DumpRates('spec/O02a.r7', 2, 7, -1, 1)
SetIteration(1e-05, 0.5)
LevelPopulation()
SpecTable('spec/O02b.sp')
PrintTable('spec/O02b.sp', 'spec/O02a.sp')
DumpRates('spec/O02a.r0', 2, 0, -1, 1)

