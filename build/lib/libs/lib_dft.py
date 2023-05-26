from ase.io.trajectory import Trajectory
from ase.io import write as atoms_write

import os
import subprocess
from decimal import Decimal

from libs.lib_util   import check_mkdir


def run_DFT(temperature, pressure, index, numstep):
    traj_DFT = Trajectory(
    f'traj-{temperature}K-{pressure}bar_{index+1}.traj',
    properties='energy, forces'
    )
    
    calcpath = f'calc/{temperature}K-{pressure}bar_{index+1}'
    check_mkdir(f'calc')
    check_mkdir(calcpath)
    mainpath_cwd = os.getcwd()

    os.chdir(calcpath)
    calcpath_cwd = os.getcwd()
    
    for jndex, jtem in enumerate(traj_DFT):
        if jndex < numstep:
            check_mkdir(f'{jndex}')
            os.chdir(f'{jndex}')
        
            if os.path.exists(f'aims/calculations/aims.out'):
                if 'Have a nice day.' in open('aims/calculations/aims.out').read():
                    os.chdir(calcpath_cwd)
                else:
                    subprocess.run(['sbatch', 'job-vibes.slurm'])
                    os.chdir(calcpath_cwd)
            else:
                aims_write('geometry.in', jtem)
                subprocess.run(['cp', '../../../template/aims.in', '.'])
                subprocess.run(['cp', '../../../template/job-vibes.slurm', '.'])
                subprocess.run(['sbatch', 'job-vibes.slurm'])
                os.chdir(calcpath_cwd)
        
    os.chdir(mainpath_cwd)
    
    
def aims_write(filename, atoms):
    velo_unit_conv = 98.22694788
    trajfile = open(filename, 'w')
    for jndex in range(3):
        trajfile.write(f'lattice_vector ' + '{:.14f}'.format(Decimal(str(atoms.get_cell()[jndex,0]))) + ' ' + '{:.14f}'.format(Decimal(str(atoms.get_cell()[jndex,1]))) + ' ' + '{:.14f}'.format(Decimal(str(atoms.get_cell()[jndex,2]))) + '\n')
    for kndex in range(len(atoms)):
        trajfile.write(f'atom ' + '{:.14f}'.format(Decimal(str(atoms.get_positions()[kndex,0]))) + ' ' + '{:.14f}'.format(Decimal(str(atoms.get_positions()[kndex,1]))) + ' ' + '{:.14f}'.format(Decimal(str(atoms.get_positions()[kndex,2]))) + ' ' + atoms.get_chemical_symbols()[kndex] + '\n')
        trajfile.write(f'    velocity ' + '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,0]*velo_unit_conv))) + ' ' + '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,1]*velo_unit_conv))) + ' ' + '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,2]*velo_unit_conv))) + '\n')
    trajfile.close()