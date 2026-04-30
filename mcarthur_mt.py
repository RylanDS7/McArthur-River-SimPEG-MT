# code by Johnathon Kuttai - github.com/JKutt

from simpeg import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
import numpy as np
import matplotlib.pyplot as plt
from pymatsolver import Pardiso
from simpeg.electromagnetics import natural_source as nsem
import matplotlib.pyplot as plt
import utm
import mtpy as mt
from mt_metadata import TF_XML
from pathlib import Path
import pickle
# Python Version
import sys
print(sys.version)


# ---------------------------------------------------------------------------

# helper functions

#

# Function to create a 2D rotation matrix
def rotation_matrix(theta_deg):
    """
    Create a 2x2 rotation matrix to rotate the impedance tensor.
    
    Parameters:
    theta_deg : float
        Rotation angle in degrees.
    
    Returns:
    R : ndarray
        2x2 rotation matrix.
    """
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    return R


# Function to rotate the impedance tensor
def rotate_impedance_tensor(Z, theta_deg):
    """
    Rotate the impedance tensor by a given angle.
    
    Parameters:
    Z : ndarray
        Original 2x2 impedance tensor (Zxx, Zxy, Zyx, Zyy).
    theta_deg : float
        Rotation angle in degrees.
    
    Returns:
    Z_rot : ndarray
        Rotated 2x2 impedance tensor.
    """
    # Compute rotation matrix
    R = rotation_matrix(theta_deg)
    
    # Apply the rotation: Z_rot = R(theta) @ Z @ R^T(-theta)
    Z_rot = np.dot(R, np.dot(Z, R.T))
    
    return Z_rot


# Function to rotate a list of points around a defined center
def rotate_points(points, center, theta_deg):
    """
    Rotate a list of 2D points around a defined center point by a given angle.
    
    Parameters:
    points : list of tuples or ndarray
        List of (x, y) coordinates to rotate.
    center : tuple
        The (x, y) coordinates of the center of rotation.
    theta_deg : float
        Rotation angle in degrees.
    
    Returns:
    rotated_points : ndarray
        List of rotated (x', y') coordinates.
    """
    # Convert points and center to numpy arrays
    points = np.array(points)
    center = np.array(center)
    
    # Create the rotation matrix
    R = rotation_matrix(theta_deg)
    
    # Translate points to the origin (subtract the center)
    translated_points = points - center
    
    # Apply the rotation matrix to the translated points
    rotated_points = np.dot(translated_points, R.T)
    
    # Translate the points back to the original center
    rotated_points += center
    
    return rotated_points


# ---------------------------------------------------------------------------

# load the MT data and do some preprocessing

#

print('[INFO] Preprocessing MT data...')

directory_path = Path("./data")
mtc = mt.MTCollection()
mtc.open_collection(Path().cwd().joinpath("test_collection100.h5"))

station_names = mtc.dataframe.station.tolist()

for file_path in directory_path.iterdir():
    station_name = file_path.stem
    if station_name in station_names: # don't load duplicate data
        continue
    if file_path.is_file():
        mt_object = mt.MT()
        mt_object.read(file_path)
        mt_object.survey_metadata.id = "grid"
        mtc.add_tf(mt_object)

mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == "grid"]

print("Converting to mtData")
mtd = mtc.to_mt_data()
mtc.close_collection()

# station_plot = mtc.plot_stations(pad=0.0005)

# collect the data into a nice list and convert the data and locations
_impUnitEDI2SI = 4 * np.pi * 1e-4

rx_locs = []
rx_locs_tipper = []
elevation = []
elevation_tipper = []

for key in mtd.keys():
    # print(mtd[key].has_tipper(), key)
    rx_locs += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2]]
    elevation += [mtd[key].elevation]

    if mtd[key].has_tipper():
        rx_locs_tipper += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2]]
        elevation_tipper += [mtd[key].elevation]

# since this is a 2D inversion rotate the coordinates of the location to inline
rotated_points = rotate_points(rx_locs, rx_locs[-4], -40)
rotated_points_tipper = rotate_points(rx_locs_tipper, rx_locs[-4], -40)

rx_locs2d = np.vstack([rotated_points[:, 0], elevation]).T
rx_locs2d_tipper = np.vstack([rotated_points_tipper[:, 0], elevation_tipper]).T

mtd.compute_model_errors()

# collect the data into something simpeg can use

data_col_te = {}
data_col_tm = {}

for key in mtd.keys():

    for ii, freq in enumerate(mtd[key].Z.frequency):
        
        data_col_te[freq] = {

            'real': [],
            'imag': [],

        }
        data_col_tm[freq] = {

            'real': [],
            'imag': [],

        }

for key in mtd.keys():

    for ii, freq in enumerate(mtd[key].Z.frequency):
        # print(f"{key} freq: {freq}")
        zrot = rotate_impedance_tensor(mtd[key].impedance[ii].values, 45.0)
        data_col_tm[freq]['real'] += [zrot[0, 1].real * _impUnitEDI2SI]
        # real data yx imag
        data_col_tm[freq]['imag'] += [zrot[0, 1].imag * _impUnitEDI2SI]
        
        data_col_te[freq]['real'] += [zrot[1, 0].real * _impUnitEDI2SI]
        # real data xy imag
        data_col_te[freq]['imag'] += [zrot[1, 0].imag * _impUnitEDI2SI]

# now determine the frequencies that all stations share
frequencies_2_use = []

for freq in data_col_te.keys():

    if len(data_col_te[freq]['real']) >= 17:
        frequencies_2_use += [freq]


# ---------------------------------------------------------------------------

# Setup the simpeg objects for survey and simulations for each TE & TM modes

#

print('[INFO] creating Tensor Mesh...')
mesh = discretize.TensorMesh(
[
    [(16000,1),(8000,1),(4000,1), (2000,2), (1000,2),(500,2),(250,2),(175,2),(125,2),(100,5),(80,20),
     (75,75),(80,20),(100,5),(125,2),(175,2),(250,2),(500,2),(1000,2), (2000,2), (4000,1), (8000,1), (16000, 1)], #[(min cell size,left padding cells, growth factor),(min cell size, amount of cells @ that size),(min cell size,right padding cells, growth factor)]
    [(16000,1),(8000,1),(4000,1),(2000,1),(1000,1),(750,1),(500,1),(375,2),(225,3),(175,5),(125,5),(100,5),(90,10),(80,15),(75,12)]
], x0=[rx_locs2d[:, 0].min() - 38500, -37900])

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs2d)


# ---------------------------------------------------------------------------

# Setup the simpeg objects for survey and simulations for each TE & TM modes

#

rx_list_te = [

    nsem.receivers.PointNaturalSource(
        rx_locs2d, orientation="xy", component="real"
    ),
    nsem.receivers.PointNaturalSource(
        rx_locs2d, orientation="xy", component="imag"
    ),

]

rx_list_tm = [
    nsem.receivers.PointNaturalSource(
        rx_locs2d, orientation="yx", component="real"
    ),
    nsem.receivers.PointNaturalSource(
        rx_locs2d, orientation="yx", component="imag"
    ),
]

src_list_te = [nsem.sources.Planewave(rx_list_te, frequency=f) for f in frequencies_2_use]
src_list_tm = [nsem.sources.Planewave(rx_list_tm, frequency=f) for f in frequencies_2_use]

# do the data

data_vec_te = []
data_vec_tm = []

data_vec_tx = []
data_vec_ty = []

for freq in frequencies_2_use:

    data_vec_te += [data_col_te[freq]['real']]
    data_vec_te += [data_col_te[freq]['imag']]
    data_vec_tm += [data_col_tm[freq]['real']]
    data_vec_tm += [data_col_tm[freq]['imag']]

data_vec_te = np.hstack(data_vec_te)
data_vec_tm = np.hstack(data_vec_tm)

# setup the survey
survey_te = nsem.Survey(src_list_te)

data_obj_te = data.Data(survey_te, data_vec_te)

survey_tm = nsem.Survey(src_list_tm)

data_obj_tm = data.Data(survey_tm, data_vec_tm)

# now the simulations
sim_type = "h"
fixed_boundary=True

actmap = maps.InjectActiveCells(

    mesh, indActive=active_cells, valInactive=np.log(1e-8)

)

m0 = (np.ones(mesh.nC) * np.log(1/1e4))[active_cells]

sim_kwargs = {"sigmaMap": maps.ExpMap() * actmap}
test_mod = m0

# create the simulation
sim_tm = nsem.simulation.Simulation2DMagneticField(
    mesh,
    survey=survey_tm,
    **sim_kwargs,
    solver=Pardiso,
)

sim_te = nsem.simulation.Simulation2DElectricField(
    mesh,
    survey=survey_te,
    **sim_kwargs,
    solver=Pardiso,
)


# ---------------------------------------------------------------------------

# create the data misfits for each mode

#

std_te = 0.03
std_tm = 0.1

print('[INFO] Getting things started on inversion...')

# TM mode
dmisfit_tm = data_misfit.L2DataMisfit(data=data_obj_tm, simulation=sim_tm)

# TE mode
data_obj_te.standard_deviation = np.abs(data_vec_te) * std_te

dmisfit_te = data_misfit.L2DataMisfit(data=data_obj_te, simulation=sim_te)

# assign the weights
dmisfit_te.W = 1. / (np.abs(data_obj_te.dobs) * std_te + np.percentile(np.abs(data_obj_te.dobs), 5, method='lower'))
dmisfit_tm.W = 1. / (np.abs(data_obj_tm.dobs) * std_tm + np.percentile(np.abs(data_obj_tm.dobs), 10, method='lower'))
dmisfit_combo = dmisfit_tm + dmisfit_te

coolingFactor = 2
coolingRate = 2
beta0_ratio = 1e0

# check for percentile floor

# Map for a regularization
regmap = maps.IdentityMap(nP=int(active_cells.sum()))

reg_tetm = regularization.WeightedLeastSquares(mesh, active_cells=active_cells, mapping=regmap)

# set alpha length scales
reg_tetm.alpha_s = 1e-8
reg_tetm.alpha_x = 1
reg_tetm.alpha_y = 1
reg_tetm.alpha_z = 1

opt_tetm = optimization.ProjectedGNCG(maxIter=20, upper=np.inf, lower=-np.inf)
invProb_tetm = inverse_problem.BaseInvProblem(dmisfit_combo, reg_tetm, opt_tetm)
beta = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
target = directives.TargetMisfit()
savedict = directives.SaveOutputEveryIteration()

directiveList = [
    beta, 
    betaest, 
    target,
    savedict,
]

inv_tetm = inversion.BaseInversion(
    invProb_tetm, directiveList=directiveList)
# opt.LSshorten = 0.5
opt_tetm.remember('xc')

# Run Inversion
minv_tetm = inv_tetm.run(m0)


# -----------------------------------------------------------------------

# Plot inversion results

#

rho_est = actmap * minv_tetm
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# conductivity
mtrue = np.log10(1/np.exp(rho_est))
mtrue[~active_cells] = np.nan
clim = [np.log10(20), np.log10(10000)]

dat = mesh.plot_image(
    (mtrue),
    ax=ax,
    # grid=True,
    clim=clim,
    pcolorOpts={"cmap": "Spectral"}
)

ax.set_title('Resistivity')
plt.colorbar(
    dat[0],
    cmap='Spectral', 
    label=r'Resistivity ($\Omega$m)', 
    ticks=[clim[0],clim[1]], 
    format="$10^{%.1f}$", 
    shrink=0.6
).ax.tick_params(labelsize=14)


ax.set_aspect('equal')
ax.plot(
    rx_locs2d[:, 0],
    rx_locs2d[:, 1], 'k.'
)
ax.set_title("TE+TM mode - Weighted Least Squares Inversion")
ax.set_xlabel("station (m)")
ax.set_ylabel("elevation (m)")
ax.set_xlim([494500, 499500])
ax.set_ylim([-1500, 800])
fig.savefig(f'final_model.png')
np.save(f'model_final.npy', opt_tetm.xc)
        
data_model = sim_te.dpred(minv_tetm)

fig, ax = plt.subplots(3, 6, figsize=(12, 8))

ax = ax.flatten()
for ii in range(rx_locs2d.shape[0]):
    # ax[ii].loglog(frequencies_2_use, -uniform_bg.reshape(34, len(frequencies_2_use), order='F')[ii, :])
    ax[ii].loglog(frequencies_2_use, -data_obj_te.dobs.reshape(34, len(frequencies_2_use), order='F')[ii, :], 'r-o')
    ax[ii].loglog(frequencies_2_use, -data_model.reshape(34, len(frequencies_2_use), order='F')[ii, :], 'g-o')
    ax[ii].set_xlabel('frequency (Hz)')
    ax[ii].set_ylabel('Z (V/m)')

fig.savefig(f'tm-mode-datafit.png')

# uniform_bg = sim_tm.dpred(m0)
data_model = sim_tm.dpred(minv_tetm)

fig, ax = plt.subplots(3, 6, figsize=(12, 8))

ax = ax.flatten()
for ii in range(rx_locs2d.shape[0]):
    # ax[ii].loglog(frequencies_2_use, uniform_bg.reshape(34, len(frequencies_2_use), order='F')[ii, :])
    ax[ii].loglog(frequencies_2_use, data_obj_tm.dobs.reshape(34, len(frequencies_2_use), order='F')[ii, :], 'r-o')
    ax[ii].loglog(frequencies_2_use, data_model.reshape(34, len(frequencies_2_use), order='F')[ii, :], 'g-o')
    ax[ii].set_xlabel('frequency (Hz)')
    ax[ii].set_ylabel('Z (V/m)')
fig.savefig(f'te-mode-datafit.png')

print(f"inversion completed")