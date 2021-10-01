#!/usr/bin/sh
#CCS -N fsfr
#CCS --res=rset=1:ncpus=14:mem=50g:amd=true
#CCS -t 1h
#CCS -M jalaj@mail.uni-paderborn.de
module load lang/Python/3.8.2-GCCcore-9.3.0
cd $PC2PFS/HPC-LTH/jalaj/fsfr/3_feature_selection
python3 ./run_constrained_evaluation_with_percent.py $1 $2
