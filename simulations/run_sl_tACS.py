
import os
import numpy as np
from pymatreader import read_mat

os.chdir('/vol/specs04/vivek/perturb_oscillator/scripts/fitted_sl/tACS_perturbation_phase/final/tACS00021_final/')

from StuartLandautACS import simulate_SL_with_tACS

# Load Connectivity Matrices
mat = read_mat('SC_90aal_32HCP.mat')
print('Loading Connectivity Matrix...')

# Structural connectivity
connect               = mat['mat']
connect[connect < 10] = 0     # Removes negligible connections
np.fill_diagonal(connect, 0)  # Removes self-connections
connect = connect / np.mean(connect)
con_mat = connect

# Tract lengths
dist_mat = mat['mat_D']

# Ensure the directory exists
os.makedirs('simulations', exist_ok=True)

# Set gc and md values
gc_val = 11.92
md_val = 3

# Calculate connectivity matrix and speed
con_mat_SL = con_mat * gc_val
speedSL    = np.mean(dist_mat[con_mat > 0]) / md_val if md_val > 0 else -1000

RUN_SEED = 12345

#Compilation
simulate_SL_with_tACS(dist_mat, con_mat_SL, 1000, 1000, 0.1, 1, -5, speedSL, perturbed_nodes=[2], shape=0, seed=RUN_SEED)
print('Compilation Finished...')

# Function to run a single simulation
def run_single_simulation_with_tACS(dist_mat, con_mat_SL, speed, gc_val, md_val, perturbed_nodes, shape, seed):
    sm_ts    = simulate_SL_with_tACS(dist_mat, con_mat_SL, 600000, 600000, 0.1, 1, -5, speed, perturbed_nodes, shape=shape, seed=seed)
    filename = f"simulations/signal_C{gc_val:.2f}_md{md_val:.0f}_dt0.1_shape{shape}.npy"
    np.save(filename, sm_ts)

# run
#nodes to perturb = [44, 45, 58, 59, 48, 49, 66, 67]
print('Running Simulation with sinusoidal perturbation...')
run_single_simulation_with_tACS(dist_mat, con_mat_SL, speedSL, gc_val, md_val, perturbed_nodes=[44, 45, 58, 59, 48, 49, 66, 67], shape=0, seed=RUN_SEED)
print('Running Simulation with square perturbation...')
run_single_simulation_with_tACS(dist_mat, con_mat_SL, speedSL, gc_val, md_val, perturbed_nodes=[44, 45, 58, 59, 48, 49, 66, 67], shape=1, seed=RUN_SEED)
print('Running Simulation with triangular perturbation...')
run_single_simulation_with_tACS(dist_mat, con_mat_SL, speedSL, gc_val, md_val, perturbed_nodes=[44, 45, 58, 59, 48, 49, 66, 67], shape=2, seed=RUN_SEED)
print('Running Simulation with pulsed perturbation...')
run_single_simulation_with_tACS(dist_mat, con_mat_SL, speedSL, gc_val, md_val, perturbed_nodes=[44, 45, 58, 59, 48, 49, 66, 67], shape=3, seed=RUN_SEED)
print('Running Simulation with sawtooth perturbation...')
run_single_simulation_with_tACS(dist_mat, con_mat_SL, speedSL, gc_val, md_val, perturbed_nodes=[44, 45, 58, 59, 48, 49, 66, 67], shape=4, seed=RUN_SEED)
