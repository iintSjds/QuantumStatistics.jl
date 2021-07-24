#!/bin/bash

#BSUB -n 1
#BSUB -W 4:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -q short
#BSUB -J "Job_kl[1-96]"
#BSUB -o logs/out.%J.%I
#BSUB -e logs/err.%J.%I

#mpiexecjl -n $LSB_DJOB_NUMPROC 
#julia ./eigensolver.jl $LSB_JOBINDEX 
julia ./KL.jl $LSB_JOBINDEX 

