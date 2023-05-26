from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from m3gnet.models import Relaxer
from libs.lib_util        import single_print

import os


def check_struc(name, supercell):
    single_print(f'Checking the structral file...\n')
    if os.path.exists(name+'-relaxed.cif'):
        single_print(f'Found the relaxed structral file: {name}-relaxed.cif')
        struc_relaxed = Structure.from_file(name+'-relaxed.cif')
        struc_relaxed.make_supercell(supercell)
    else:
        single_print(f'Initiate a relaxation calculation...\n')
        struc_initial = Structure.from_file(f'{name}.cif')
        struc_relaxed = struc_relax(struc_initial, name)
        struc_relaxed.make_supercell(supercell)
        single_print(f'Finish the structural relaxation...\n')
        
    NumAtoms  = len(struc_relaxed.species)
    atoms_relaxed = AseAtomsAdaptor.get_atoms(struc_relaxed)
    
    return atoms_relaxed, NumAtoms
    
    
def struc_relax(structure, name):
    relax_results = Relaxer().relax(structure, verbose=True) # This loads the default pre-trained model
    final_struc = relax_results['final_structure']
    writing = CifWriter(final_struc)
    writing.write_file(f'{name}-relaxed.cif')
    
    return final_struc