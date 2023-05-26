al_type = 'force'
uncert_type = 'absolute'
ensemble = 'NVTLangevin'
name = 'C-diamond'
supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
crtria_cnvg = 0.0000001
output_format = 'trajectory.son'

timestep = 1
loginterval = 10
steps_ther = 5000
steps_init = 125
nstep = 1
nmodel = 1

cutoff_ther = 100

init_temp = 2000
final_temp = 2000
step_temp = 2000
friction = 0.02
taut = 50

init_press = 0
final_press = 0
step_press = 0
compressibility = 4.57e-5
taup = 100
mask = (1, 1, 1)

ntrain_init = 5
ntrain = 5
rmax = 3.5
lmax = 2
nfeatures = 16

kB = 8.617333262 * (10 ** (-5))  # Unit: eV/K

__all__ = [
    'al_type', 'uncert_type', 'ensemble', 'name', 'supercell', 'crtria_cnvg', 'output_format',
    'timestep', 'loginterval', 'steps_ther', 'steps_init', 'nstep', 'nmodel',
    'cutoff_ther', 'init_temp', 'final_temp', 'step_temp', 'friction', 'taut',
    'init_press', 'final_press', 'step_press', 'compressibility', 'taup', 'mask',
    'ntrain_init', 'ntrain', 'rmax', 'lmax', 'nfeatures', 'kB'
]

