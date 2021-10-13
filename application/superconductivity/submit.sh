#!/bin/bash

#BSUB -n 1
#BSUB -W 23:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -q long
#BSUB -J "SCFLOW[1-64]"
#BSUB -o logs/out.%J.%I
#BSUB -e logs/err.%J.%I

#mpiexecjl -n $LSB_DJOB_NUMPROC 
julia ../../../application/superconductivity/eigensolver_spFreq.jl $LSB_JOBINDEX 
#julia ../../../application/superconductivity/self_energy.jl $LSB_JOBINDEX 
#julia ./KL_explicit.jl $LSB_JOBINDEX 
