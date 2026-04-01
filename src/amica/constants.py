"""Parameters taken from Fortran program that we do not want to expose via the API."""
import numpy as np

fix_init = False
mineig = 1.0e-12
dorho = True
rho0 = 1.5
minlog = -1500  # XXX: -np.inf ?
epsdble = 1.0e-16
maxrej = 3
rejstart = 2
rejint = 3
doscaling = True
share_comps = False
share_start = 100
share_iter = 100
minrho = 1.0
maxrho = 2.0
invsigmin = 1.0e-08
invsigmax = 100.0
use_min_dll = True
use_grad_norm = True
min_dll = 1.000000e-09
maxincs = 5  # Consecutive iters where loglik increase is less than tol before exiting
maxdecs = 3
outstep = 20
restartiter = 10
numrestarts = 0
maxrestarts = 3
minlrate = 1.0e-08
min_nd = 1.0e-7 # tol
lratefact = 0.5
rholratefact = 0.5
load_gm = False
load_A = False
load_mu = False
load_sbeta = False
load_beta = False
load_rho = False
load_c = False
load_alpha = False
do_opt_block = False
do_approx_sphere = False

LOG_2 = np.log(2.0).item()
LOG_SQRT_PI = np.log(np.sqrt(np.pi)).item()
