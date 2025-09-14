from uedge import *
from uedge.hdf5 import *
from uedge.rundt import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.analysis as an
import uedge_mvu.tstep as ut
import UEDGE_utils.analysis as ana
import UEDGE_utils.plot as utplt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from runcase import *
import os


setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=0.36, kyi=0.36, difni=0.5,nonuniform = True, kye_sol=0.42)
setBoundaryConditions(ncore=6.2e19, pcoree=2.8e6, pcorei=2.8e6, recycp=0.95, owall_puff=0)
setimpmodel(impmodel=True)


bbb.cion=3
bbb.oldseec=0
bbb.restart=1
bbb.nusp_imp = 3
bbb.icntnunk=0
bbb.kye=0.36
bbb.kyi = 0.36
setDChi(kye=0.36, kyi=0.36, difni=0.5,nonuniform = True, kye_sol=0.42)



hdf5_restore("./final.hdf5")



bbb.issfon=1
bbb.ftol = 1e-5
bbb.dtreal=1e-10
bbb.exmain()



try:
    lambda_exp_fit_out, lambda_eich_fit_out = ana.eich_exp_shahinul_odiv_final()
    lambda_q = lambda_eich_fit_out  # Already in mm
    print(f"    ?? Initial lambda_q = {lambda_q:.3f} mm")
except Exception as e:
    print("Error during initial lambda_q computation:", e)
    sys.exit(1)



pcore = bbb.pcoree + bbb.pcorei
print("Initial Power is:", pcore)


pcore_max = 2.1e6
pcore_step = 2e5  # W
lambda_q_target_range = (1.9, 2.05)

kye_init = 0.5
setDChi(kye=0.36, kyi=0.36, difni=0.5, nonuniform=True, kye_sol =kye_init)

kye_step = 0.02
max_kye_attempts = 50


logfile = "lambdaq_log.csv"
if not os.path.exists(logfile):
    with open(logfile, "w") as f:
        f.write("pcore,kye,lambda_q,converged\n")

while pcore >= pcore_max:
    print("\n====================================")
    pcore -= pcore_step

    bbb.pcoree = pcore / 2
    bbb.pcorei = pcore / 2

    kye_val = kye_init
    kye_attempt = 0
    converged = False
    lambda_q = np.nan

    while kye_attempt < max_kye_attempts:
        bbb.kye=0.36
        bbb.kyi = 0.36
        setDChi(kye=0.36, kyi=0.36, difni=0.5, nonuniform=True, kye_sol =kye_val)
        bbb.dtreal = 1e-10

        try:
            print("Running bbb.exmain()")
            bbb.exmain()

            print("Advancing with rundt()")
            rundt(dtreal=1e-9)

            if bbb.iterm == 1:
                print("Solver converged.")
                try:
                    lambda_exp_fit_out, lambda_eich_fit_out = ana.eich_exp_shahinul_odiv_final()
                    lambda_q = lambda_eich_fit_out
                except Exception as e:
                    kye_val += kye_step
                    kye_attempt += 1
                    continue

                if lambda_q < lambda_q_target_range[0]:
                    kye_val += kye_step
                    print("    lambda_q too low - increasing kye")
                elif lambda_q > lambda_q_target_range[1]:
                    kye_val = max(0.01, kye_val - kye_step)
                    print("    lambda_q too high - decreasing kye")
                else:
                    fname = f"tmp_pcore_{pcore:.3e}_kye_{kye_val:.2f}.h5"
                    print(f"    lambda_q within target range. Saving to: {fname}")
                    hdf5_save(fname)
                    converged = True
                    break
            else:
                print("    Solver did not converge - retrying...")

        except Exception as e:
            print("    Exception during solver step:", e)

        kye_attempt += 1

    # Log result
    with open(logfile, "a") as f:
        lam_val = lambda_q if converged else np.nan
        f.write(f"{pcore:.2e},{kye_val:.3f},{lam_val:.3f},{converged}\n")

    if not converged:
        print(f"Could not converge within kye attempts for pcore = {pcore:.2e}")
        break
