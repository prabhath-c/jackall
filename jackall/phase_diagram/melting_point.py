import json
import numpy as np
import os
import pandas
import random
from pyiron_base import job, Project

from pyiron_atomistics.thermodynamics.interfacemethod import fix_iso, half_velocity, analyse_minimized_structure, initialise_iterators

pr = Project('./Meltin_point_binary')

pot_dict = {
    'Config': {0: ['pair_style pace\n',
                   'pair_coeff  * * /cmmc/u/pchilaka/1_Work/1_My_Notebooks/4_Regular/2_Phase_Diagrams/Marvin_Potentials/ACE500_8d2_E_sqrt_W_ch_L_30_0d05_I_600/output_potential.yace Al Mg\n']},
    'Filename': {0: ''},
    'Model': {0: 'ACE'},
    'Name': {0: 'Marvins_Potential'},
    'Species': {0: ['Al', 'Mg']}
}