#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --account=education-as-msc-ce
#SBATCH --time=24:00:00
#SBATCH -J CREST

xtb L1.xyz --gfn2//gfnff  --opt --chrg 1 --uhf 1 > xtb.out
crest xtbopt.xyz --gfn2//gfnff --chrg 1 --uhf 1 > crest.out

