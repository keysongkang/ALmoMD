import os
import son
import sys

import numpy as np
from mpi4py import MPI
from decimal import Decimal
from nequip.ase import nequip_calculator

from ase import Atoms
from ase.build import make_supercell
from ase.io import read as atoms_read

from libs.lib_util import check_mkdir, job_dependency, read_input_file, mpi_print
from libs.lib_npz import generate_npz_DFT_init, generate_npz_DFT, generate_npz_DFT_rand_init, generate_npz_DFT_rand
from libs.lib_train import get_train_job, execute_train_job
from libs.lib_dft import run_DFT
from libs.lib_progress    import check_progress, check_index
from libs.lib_mainloop    import MLMD_initial, MLMD_main, MLMD_random
from libs.lib_criteria    import get_criteria, get_average
from libs.lib_termination import termination

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_absolute_error

import torch
torch.set_default_dtype(torch.float64)


class almd:
    def __init__(self, input_file='input.in'):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        self.al_type = 'force'
        self.uncert_type = 'absolute'
        self.ensemble = 'NVTLangevin'
        self.name = 'C-diamond'
        self.supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        self.crtria_cnvg = 0.0000001
        self.output_format = 'trajectory.son'
        self.timestep = 1
        self.loginterval = 10
        self.steps_ther = 5000
        self.steps_init = 125
        self.nstep = 1
        self.nmodel = 1
        self.cutoff_ther = 100
        self.temperature = 2000
        self.friction = 0.02
        self.taut = 50
        self.pressure = 0
        self.compressibility = 4.57e-5
        self.taup = 100
        self.mask = (1, 1, 1)
        self.ntrain_init = 5
        self.ntrain = 5
        self.rmax = 3.5
        self.lmax = 2
        self.nfeatures = 16
        self.kB = 8.617333262e-5
        self.E_gs = 0.0

        # Read input variables from the input file
        input_variables = read_input_file(input_file)
        
        # Update instance attributes with the input variables
        self.__dict__.update(input_variables)
        
        # Perform postprocessing steps
        self.nval_init = int(self.ntrain_init / 5)
        self.nval = int(self.ntrain / 5)
        self.ntotal_init = (self.ntrain_init + self.nval_init) * self.nstep
        self.ntotal = (self.ntrain + self.nval) * self.nstep
        self.crtria = self.crtria_cnvg
        self.index = 0
        self.index_old = 0
        self.total_ntrain = 0
        self.total_nval = 0
    
    def run_dft_init(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        metadata, traj = son.load('trajectory_train.son')
        traj_ther = traj[:]

        index = 0
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{index}'
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)

        total_ntrain = self.ntrain_init
        total_nval = nval_init

        if rank == 0:
            generate_npz_DFT_init(
                traj_ther, self.ntrain_init, nval_init, self.nstep, E_gs, index, self.temperature,
                self.pressure, workpath
            )
            del traj_ther

        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel, comm, size, rank
        )
        if not self.calc_type == 'converge':
            if rank == 0:
                job_dependency('cont')
    
    def run_dft_cont(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        struc_init = atoms_read('geometry.in.next_step', format='aims')
        struc_relaxed = make_supercell(struc_init, self.supercell)
        self.NumAtoms = len(struc_relaxed)

        if self.calc_type == 'active':
            kndex, MD_index, signal = None, None, None
            if rank == 0:
                kndex, MD_index, self.index, signal = check_progress(
                    self.temperature, self.pressure, self.ensemble, self.timestep,
                    self.friction, self.compressibility, self.taut, self.taup, self.mask,
                    self.loginterval, self.steps_ther, self.name, self.supercell,
                    self.ntotal, self.ntrain, self.ntrain_init, self.nval, self.nval_init, self.rmax,
                    self.lmax, self.nfeatures, self.nstep, self.nmodel, self.steps_init, self.index,
                    self.crtria, self.crtria_cnvg, self.NumAtoms, comm, size, rank
                )
            kndex = comm.bcast(kndex, root=0)
            MD_index = comm.bcast(MD_index, root=0)
            self.index = comm.bcast(self.index, root=0)
            self.temperature = comm.bcast(self.temperature, root=0)
            signal = comm.bcast(signal, root=0)
        elif self.calc_type == 'random':
            kndex = 0
            MD_index = 0
            signal = 0
            total_index = None
            if rank == 0:
                self.index = check_index(self.index)
            self.index = comm.bcast(self.index, root=0)

        if signal == 1:
            sys.exit()

        total_index = None
        if rank == 0:
            total_index = check_index(self.index)
        total_index = comm.bcast(total_index, root=0)

        self.total_ntrain = self.ntrain * total_index + self.ntrain_init
        self.total_nval = self.nval * total_index + self.nval_init

        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{self.index}'
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)

        E_ref, calc_MLIP = get_train_job(
            struc_relaxed, self.total_ntrain, self.total_nval, self.rmax, self.lmax,
            self.nfeatures, workpath, self.nstep, self.nmodel, comm, size, rank
        )

        if self.calc_type == 'active':
            struc_step = MLMD_initial(
                kndex, self.index, self.ensemble, self.temperature, self.pressure, self.timestep, self.friction,
                self.compressibility, self.taut, self.taup, self.mask, self.loginterval, self.steps_init, self.nstep, self.nmodel, calc_MLIP, E_ref, self.al_type, comm, size, rank
            )
            MLMD_main(
                MD_index, self.index, self.ensemble, self.temperature, self.pressure, self.timestep, self.friction,
                self.compressibility, self.taut, self.taup, self.mask, self.loginterval, self.ntotal, self.nstep,
                self.nmodel, calc_MLIP, E_ref, self.steps_init, self.NumAtoms, self.kB, struc_step,
                self.al_type, self.uncert_type, comm, size, rank
            )
        elif self.calc_type == 'random':
            struc_step = MLMD_random(
                kndex, self.index, self.ensemble, self.temperature, self.pressure, self.timestep, self.friction,
                self.compressibility, self.taut, self.taup, self.mask, self.loginterval, self.steps_init * self.loginterval,
                self.nstep, self.nmodel, calc_MLIP, E_ref, comm, size, rank
            )
        else:
            raise ValueError("Invalid calc_type. Supported values are 'active' and 'random'.")

        if rank == 0:
            run_DFT(self.temperature, self.pressure, self.index, (self.ntrain + self.nval) * self.nstep)


    def run_dft_gen(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        struc_init = atoms_read('geometry.in.next_step', format='aims')
        struc_relaxed = make_supercell(struc_init, self.supercell)
        self.NumAtoms = len(struc_relaxed)

        if self.calc_type == 'active':
            kndex, MD_index, signal = None, None, None
            if rank == 0:
                # Open the uncertainty output file
                kndex, MD_index, self.index, signal = check_progress(
                    self.temperature, self.pressure, self.ensemble, self.timestep,
                    self.friction, self.compressibility, self.taut, self.taup, self.mask,
                    self.loginterval, self.steps_ther, self.name, self.supercell,
                    self.ntotal, self.ntrain, self.ntrain_init, self.nval, self.nval_init, self.rmax,
                    self.lmax, self.nfeatures, self.nstep, self.nmodel, self.steps_init, self.index,
                    self.crtria, self.crtria_cnvg, self.NumAtoms, comm, size, rank
                )
            kndex = comm.bcast(kndex, root=0)
            MD_index = comm.bcast(MD_index, root=0)
            self.index = comm.bcast(self.index, root=0)
            self.temperature = comm.bcast(self.temperature, root=0)
            signal = comm.bcast(signal, root=0)
        elif self.calc_type == 'random':
            kndex = 0
            MD_index = 0
            signal = 0
            total_index = None
            if rank == 0:
                self.index = check_index(self.index)
            self.index = comm.bcast(self.index, root=0)
        else:
            single_print('You need to assign calc_type.')

        if signal == 1:
            sys.exit()

        total_index = None
        if rank == 0:
            total_index = check_index(self.index)
        total_index = comm.bcast(total_index, root=0)

        self.total_ntrain = self.ntrain * total_index + self.ntrain_init
        self.total_nval = self.nval * total_index + self.nval_init
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{self.index}'
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)

        if rank == 0:
            generate_npz_DFT(
                self.ntrain, self.nval, self.nstep, self.E_gs, self.index, self.temperature,
                self.output_format, self.pressure, workpath
            )

        # Training process
        execute_train_job(
            self.total_ntrain, self.total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel, comm, size, rank
        )

        if signal == 1:
            mpi_print(f'{self.temperature}K is converged.')
            sys.exit()

        # if rank == 0:
        #     job_dependency('cont')


    def run_dft_rand(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        metadata, traj = son.load('trajectory.son')
        traj_ther = traj[self.cutoff_ther:]

        total_ntrain = self.ntrain_init
        total_nval = self.nval_init

        index = 0
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{index}'
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)
            traj_idx = generate_npz_DFT_rand_init(
                traj_ther, self.ntrain_init, self.nval_init, self.nstep, self.E_gs, index, self.temperature,
                self.pressure, workpath
            )

        # Training process
        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel, comm, size, rank
        )

        while index < self.random_index:
            index += 1
            total_ntrain += self.ntrain
            total_nval += self.nval

            workpath = f'./data/{self.temperature}K-{self.pressure}bar_{index}'

            if rank == 0:
                check_mkdir(workpath)
                traj_idx = generate_npz_DFT_rand(
                    traj_ther, self.ntrain, self.nval, self.nstep, self.E_gs, index, self.temperature,
                    self.pressure, workpath, traj_idx
                )

            # Training process
            execute_train_job(
                total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
                workpath, self.nstep, self.nmodel, comm, size, rank
            )


    def run_dft_rand(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        metadata, traj = son.load('trajectory.son')
        traj_ther = traj[self.cutoff_ther:]

        total_ntrain = self.ntrain_init
        total_nval = self.nval_init

        index = 0
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{index}'
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)
            traj_idx = generate_npz_DFT_rand_init(
                traj_ther, self.ntrain_init, self.nval_init, self.nstep, self.E_gs, index, self.temperature,
                self.pressure, workpath
            )

        # Training process
        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel, comm, size, rank
        )

        while index < self.random_index:
            index += 1
            total_ntrain += self.ntrain
            total_nval += self.nval

            workpath = f'./data/{self.temperature}K-{self.pressure}bar_{index}'

            if rank == 0:
                check_mkdir(workpath)
                traj_idx = generate_npz_DFT_rand(
                    traj_ther, self.ntrain, self.nval, self.nstep, self.E_gs, index, self.temperature,
                    self.pressure, workpath, traj_idx
                )

            # Training process
            execute_train_job(
                total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
                workpath, self.nstep, self.nmodel, comm, size, rank
            )


    def run_dft_test(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        npz_test = f'./../data/data-test.npz'
        data_test = np.load(npz_test)
        NumAtom = len(data_test['z'][0])

        workpath = f'./../data/{self.temperature}K-{self.pressure}bar_{self.wndex}'
        signal = 0

        calc_MLIP = []
        for index_nmodel in range(self.nmodel):
            for index_nstep in range(self.nstep):
                if (index_nmodel * self.nstep + index_nstep) % size == rank:
                    dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                    if os.path.exists(f'{workpath}/{dply_model}'):
                        mpi_print(f'Found the deployed model: {dply_model}', rank)
                        calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}')
                        )
                    else:
                        mpi_print(f'Cannot find the model: {dply_model}', rank)
                        signal = 1
                        signal = comm.bcast(signal, root=rank)

        if signal == 1:
            mpi_print('Not enough trained models', rank)
            sys.exit()

        if rank == 0:
            outputfile = open(f'result-test_{self.wndex}.txt', 'w')
            outputfile.write('index   \tUncertAbs\tRealErrorAbs\tRealE\tPredictE\n')
            outputfile.close()

        index = 1
        for id_E, id_F, id_R, id_z, id_CELL, id_PBC in zip(data_test['E'], data_test['F'], data_test['R'],
                                                          data_test['z'], data_test['CELL'], data_test['PBC']):
            struc = Atoms(id_z, positions=id_R, cell=id_CELL, pbc=id_PBC)
            prd_E = []
            prd_F = []
            zndex = 0
            for index_nmodel in range(self.nmodel):
                for index_nstep in range(self.nstep):
                    if (index_nmodel * self.nstep + index_nstep) % size == rank:
                        struc.calc = calc_MLIP[zndex]
                        prd_E.append(struc.get_potential_energy() + self.E_gs)
                        prd_F.append(struc.get_forces())
                        zndex += 1
            prd_E = comm.allgather(prd_E)
            prd_E_avg = np.average([jtem for item in prd_E if len(item) != 0 for jtem in item], axis=0)
            prd_E_std = np.std([jtem for item in prd_E if len(item) != 0 for jtem in item], axis=0)

            prd_F = comm.allgather(prd_F)
            prd_F_avg = np.average([jtem for item in prd_F if len(item) != 0 for jtem in item], axis=0)

            realerror_E = np.absolute(prd_E_avg - id_E)

            if rank == 0:
                trajfile = open(f'result-test_{self.wndex}.txt', 'a')
                trajfile.write(
                    str(index) + str('          \t') +
                    '{:.5e}'.format(Decimal(str(prd_E_std))) + str('\t') +
                    '{:.5e}'.format(Decimal(str(realerror_E))) + str('\t') +
                    '{:.10e}'.format(Decimal(str(id_E))) + str('\t') +
                    '{:.10e}'.format(Decimal(str(prd_E_avg))) + str('\n')
                )
                trajfile.close()

            if rank == 0:
                trajfile = open(f'result-test_{self.wndex}_force.txt', 'a')
                for kndex in range(len(prd_F_avg)):
                    for lndex in range(3):
                        trajfile.write(
                            '{:.10e}'.format(Decimal(str(id_F[kndex][lndex]))) + str('\t') +
                            '{:.10e}'.format(Decimal(str(prd_F_avg[kndex][lndex]))) + str('\n')
                        )
                trajfile.close()

            index += 1

        data = pd.read_csv(f'result-test_{self.wndex}.txt', sep="\t")

        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['font.size'] = "23"

        fig, ax1 = plt.subplots(figsize=(6.5, 6), dpi=80)

        ax1.plot(data['RealE'], data['PredictE'], color="orange", linestyle='None', marker='o')
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)

        ax1.set_xlabel('Real E')
        ax1.set_ylabel('Predicted E')
        ax1.tick_params(direction='in')
        plt.tight_layout()
        plt.show()
        fig.savefig('figure_energy.png')

        R2_E = r2_score(data['RealE'], data['PredictE'])
        MAE_E = mean_absolute_error(data['RealE'], data['PredictE'])
        UncertAbs_E = np.average(data['UncertAbs'])

        data = pd.read_csv(f'result-test_{self.wndex}_force.txt', sep="\t", header=None)

        fig, ax1 = plt.subplots(figsize=(6.5, 6), dpi=80)

        ax1.plot(data[0], data[1], color="orange", linestyle='None', marker='o')
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)

        ax1.set_xlabel('Real F')
        ax1.set_ylabel('Predicted F')
        ax1.tick_params(direction='in')
        plt.tight_layout()
        plt.show()
        fig.savefig('figure_force.png')

        R2_F = r2_score(data[0], data[1])
        MAE_F = mean_absolute_error(data[0], data[1])

        mpi_print(f'Energy: {R2_E}\t{MAE_E}\t{R2_F}\t{MAE_F}')
