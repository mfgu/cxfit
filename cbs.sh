#!/bin/bash

#generate stark mixed basis model H-like
python cbs.py 26 1 H 0 > z26k2s0h.log &
python cbs.py 26 1 H2 0 > z26k2s0h2.log &
python cbs.py 26 1 He 0 > z26k2s0he.log &
python cbs.py 26 1 N2 0 > z26k2s0n2.log &

#He-like singlet
python cbs.py 26 2 H 1 > z26k2s1h.log &
python cbs.py 26 2 H2 1 > z26k2s1h2.log &
python cbs.py 26 2 He 1 > z26k2s1he.log &
python cbs.py 26 2 N2 1 > z26k2s1n2.log &

#He-like triplet
python cbs.py 26 2 H 3 > z26k2s3h.log &
python cbs.py 26 2 H2 3 > z26k2s3h2.log &
python cbs.py 26 2 He 3 > z26k2s3he.log &
python cbs.py 26 2 N2 3 > z26k2s3n2.log &

#He-like statistical mix
python cbs.py 26 2 H 0 > z26k2s0h.log &
python cbs.py 26 2 H2 0 > z26k2s0h2.log &
python cbs.py 26 2 He 0 > z26k2s0he.log &
python cbs.py 26 2 N2 0 > z26k2s0n2.log &
