from scripts.default_inputs import *

def initialize_inputs():
    # Read and update variables from "input.in" file
    with open('input.in') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('\t'):
                name_vari, value_vari = line.split(':')
                name_vari = name_vari.strip()
                value_vari = value_vari.strip()
                if name_vari in globals():
                    globals()[name_vari] = eval(value_vari)

    # Return the updated variables
    return globals()

