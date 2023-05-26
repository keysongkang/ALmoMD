import os
import re
import sys
import son
import random
import argparse
import collections
import numpy as np
import ase.units as units
from ase import Atoms
from ase.data import atomic_numbers
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from libs.lib_util import check_mkdir

def aims2son(temperature):
    index_struc = 0
    index_force = 0
    index_stress_whole = 0
    index_stress_individual = 0
    signal = 0

    cell = []
    forces = []
    numbers = []
    numbers_symbol = []
    positions = []
    mass = []
    stress_whole = []
    stress_individual = []
    pbc = [True, True, True]
    NumAtoms = 0

    with open('aims.out', "r") as file_one:
        for line in file_one:
            if re.search('Number of atoms', line):
                NumAtoms = int(re.findall(r'\d+', line)[0])

            if re.search('Found atomic mass :', line):
                mass.append(float(float(re.findall(r'[+-]?\d+(?:\.\d+)?', line)[0])))

            if re.search('Total energy corrected', line):
                total_E = float(re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)[0])

            if re.search('Atomic structure that was used in the preceding time step of the wrapper', line):
                index_struc = 1
            if index_struc > 0:
                if index_struc > 2 and index_struc < 6:
                    cell.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    index_struc += 1
                elif index_struc > 6 and index_struc < (7+NumAtoms):
                    positions.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    numbers.append(atomic_numbers[(line[-3:].replace(' ', '')).replace('\n','')])
                    numbers_symbol.append((line[-3:].replace(' ', '')).replace('\n',''))
                    index_struc += 1
                elif index_struc == (7+NumAtoms):
                    index_struc = 0
                    signal = 1
                else:
                    index_struc += 1

            if re.search('Total atomic forces', line):
                index_force = 1
            if index_force > 0:
                if index_force > 1 and index_force < (2+NumAtoms):
                    forces.append([float(i) for i in re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)])
                    index_force += 1
                elif index_force == (2+NumAtoms):
                    index_force = 0
                else:
                    index_force += 1

            if re.search('Analytical stress tensor - Symmetrized', line):
                index_stress_whole = 1
            if index_stress_whole > 0:
                if index_stress_whole > 5 and index_stress_whole < 9:
                    stress_whole.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    index_stress_whole += 1
                elif index_stress_whole == 9:
                    index_stress_whole = 0
                else:
                    index_stress_whole += 1

            if re.search('used for heat flux calculation', line):
                index_stress_individual = 1
            if index_stress_individual > 0:
                if index_stress_individual > 3 and index_stress_individual < (4+NumAtoms):
                    stress_temp = [float(i) for i in re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)]
                    stress_individual.append([[stress_temp[0],stress_temp[4],stress_temp[5]],[stress_temp[4],stress_temp[2],stress_temp[5]],[stress_temp[5],stress_temp[5],stress_temp[3]]])
                    index_stress_individual += 1
                elif index_stress_individual == 4+NumAtoms:
                    index_stress_individual = 0
                else:
                    index_stress_individual += 1

            if signal:
                atom = Atoms(
                    numbers,
                    positions=positions,
                    cell=cell,
                    pbc=pbc
                )
                MaxwellBoltzmannDistribution(atom, temperature_K=temperature, force_temp=True)

                symbols = []
                masses = []
                idx = 0
                for key, value in collections.Counter(numbers_symbol).items():
                    symbols.append([value, key])
                    masses.append([value, mass[idx]])
                    idx += 1

                atoms_info = {
                    "pbc": pbc,
                    "cell": cell,
                    "positions": positions,
                    "velocities": atom.get_velocities().tolist(),
                    "symbols": symbols,
                    "masses": masses
                }

                calculator_info = {
                    "energy": total_E,
                    "forces": forces,
                    "stress": stress_whole,
                    "stresses": stress_individual
                }

                atom_dict = {
                    "atoms": atoms_info,
                    "calculator": calculator_info
                }

                index_struc = 0
                index_force = 0
                index_stress_whole = 0
                index_stress_individual = 0
                signal = 0
                cell = []
                forces = []
                numbers = []
                numbers_symbol = []
                positions = []
                stress_whole = []
                stress_individual = []

                son.dump(atom_dict, 'trajectory.son', is_metadata=False)


def split_son(num_split, E_gs):
    metadata, data = son.load('trajectory.son')
    test_data = random.sample(data, num_split)
    train_data = [d for d in data if d not in test_data]
    
    if os.path.exists('trajectory_test.son') or os.path.exists('trajectory_train.son'):
        print('There is already a trajectory_test.son or trajectory_train.son file.')
    else:
        for test_item in test_data:
            son.dump(test_item, 'trajectory_test.son')
        for train_item in train_data:
            son.dump(train_item, 'trajectory_train.son')

    check_mkdir('data')
    if os.path.exists('data/data-test.npz'):
        print('There is already data-test.npz file.')
    else:
        E_test      = []
        F_test      = []
        R_test      = []
        z_test      = []
        CELL_test   = []
        PBC_test    = []

        for test_item in test_data:
            E_test.append(test_item['calculator']['energy'] - E_gs);
            F_test.append(test_item['calculator']['forces']);
            R_test.append(test_item['atoms']['positions']);
            z_test.append([atomic_numbers[item[1]] for item in test_item['atoms']['symbols'] for index in range(item[0])]);
            CELL_test.append(test_item['atoms']['cell']);
            PBC_test.append(test_item['atoms']['pbc'])
            
        npz_name = 'data/data-test.npz'
        np.savez(
            npz_name[:-4],
            E=np.array(E_test),
            F=np.array(F_test),
            R=np.array(R_test),
            z=np.array(z_test),
            CELL=np.array(CELL_test),
            PBC=np.array(PBC_test)
        )
        
        print('Finish the sampling testing data: data-train.npz')


