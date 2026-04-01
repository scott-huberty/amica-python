"""Parameters taken from Fortran program that we do not want to expose via the API."""
import numpy as np

fix_init = False
do_approx_sphere = False
mineig = 1.0e-12
minlog = -1500          # XXX: -np.inf ?
epsdble = 1.0e-16
lratefact = 0.5         # multiplicative factor by which to decrease lrate
minlrate = 1.0e-08      # lrate after which to stop updating lrate
doscaling = True        # rescale unmixing matrix rows to unit norm
dorho = True
rho0 = 1.5              # initial shape parameter value
minrho = 1.0            # minimum shape parameter value, def=1.0
maxrho = 2.0            # maximum shape parameter value, def=2.0
rholratefact = 0.5      # multiplicative factor by which to dec rholrate, def=0.5
invsigmin = 1.0e-08     # minimum value of inverse scale parameters
invsigmax = 100.0       # maximum value of inverse scale parameters
use_min_dll = True      # stop when LL improvement is < min_dll
use_grad_norm = True    # stop when gradient norm (ndtmpsum) is < min_nd
min_dll = 1.000000e-09  # tol
min_nd = 1.0e-7         # tol
maxincs = 5  # Consecutive iters where LL increase is less than tol before exiting
maxdecs = 3
maxrej = 3              # Unused
rejstart = 2            # Unused
rejint = 3              # Unused
share_comps = False     # Unused
share_start = 100       # Unused
share_iter = 100        # Unused
outstep = 20            # Unused
restartiter = 10        # Unused
numrestarts = 0         # Unused
maxrestarts = 3         # Unused
load_gm = False         # Unused
load_A = False          # Unused
load_mu = False         # Unused
load_sbeta = False      # Unused
load_beta = False       # Unused
load_rho = False        # Unused
load_c = False          # Unused
load_alpha = False      # Unused
do_opt_block = False    # Unused

LOG_2 = np.log(2.0).item()
LOG_SQRT_PI = np.log(np.sqrt(np.pi)).item()
