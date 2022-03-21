#!/bin/bash

python bmc_sfe_h2.py > sfe_h2.log &
python bmc_sfe_h.py > sfe_h.log &
python bmc_sfe_n2.py > sfe_n2.log &
python bmc_sfe_he.py > sfe_he.log &

python bmc_nfe_h2.py > nfe_h2.log &
python bmc_nfe_h.py > nfe_h.log &
python bmc_nfe_n2.py > nfe_n2.log &
python bmc_nfe_he.py > nfe_he.log &
