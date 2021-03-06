#!/bin/bash -l
#SBATCH --job-name="Urea-Small"
#SBATCH --time=24:00:00
#SBATCH --account=mr3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=mc

############################################################################
# Variables definition
############################################################################
EXE=/users/piaggip/bin/DAINT/Gromacs/gromacs-5.1.4/build_mpi/bin/mdrun
MAX_TIME=23
Project=Urea
cycles=8
############################################################################


############################################################################
# Run
############################################################################
if [ -e runno ] ; then
   #########################################################################
   # Restart runs
   #########################################################################
   nn=`tail -n 1 runno | awk '{print $1}'`
   srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ${EXE} -maxh ${MAX_TIME} -plumed plumed.dat -s topol.tpr -cpi ${Project}.cpt -deffnm ${Project} &> ${Project}.out
   #########################################################################
else
   #########################################################################
   # First run
   #########################################################################
   nn=1
   srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK ${EXE} -maxh ${MAX_TIME} -plumed plumed.dat -s topol.tpr -deffnm ${Project} &> ${Project}.out
   #########################################################################
fi
############################################################################


############################################################################
# Check number of cycles
############################################################################
mm=$((nn+1))
echo ${mm} > runno
#cheking number of cycles
if [ ${nn} -ge ${cycles} ]; then
  exit
fi
############################################################################

############################################################################
# Resubmitting again
############################################################################
sbatch < job.sh
############################################################################
