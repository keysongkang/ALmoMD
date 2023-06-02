import son
from mpi4py import MPI
from read_input import *
from libs.lib_util import check_mkdir, job_dependency
from libs.lib_npz import generate_npz_DFT_init
from libs.lib_MODEL import execute_MODEL_job

def run_dft_init():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Call the initialize_inputs() function and access the updated variables
    input_variables = initialize_inputs()

    # Update the global namespace with the updated variables
    globals().update(input_variables)

    print("what are you doing?")

    metadata, traj = son.load('trajectory.son')
    traj_ther = traj[cutoff_ther:]

    index = 0
    workpath = f'./data/{temperature}K-{pressure}bar_{index}'
    check_mkdir(workpath)

    total_nMODEL = nMODEL_init
    total_nval = nval_init

    if rank == 0:
        generate_npz_DFT_init(
            traj_ther, nMODEL_init, nval_init, nstep, E_gs, index, temperature,
            init_temp, step_temp, pressure, workpath
        )
        del traj_ther

    execute_MODEL_job(
        total_nMODEL, total_nval, rmax, lmax, nfeatures,
        workpath, nstep, nmodel, comm, size, rank
    )

    # if rank == 0:
    #     job_dependency('cont')
