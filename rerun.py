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




setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=0.46, kyi=0.46, difni=0.5,nonuniform = True, kye_sol=0.5)
setBoundaryConditions(ncore=6.2e19, pcoree=3.0e6, pcorei=3.0e6, recycp=0.95, owall_puff=0)
setimpmodel(impmodel=True)


bbb.cion=3
bbb.oldseec=0
bbb.restart=1
bbb.nusp_imp = 3
bbb.icntnunk=0
bbb.kye=0.46
bbb.kyi = 0.46
setDChi(kye=0.36, kyi=0.36, difni=0.5,nonuniform = True, kye_sol=0.5)



hdf5_restore("./final.hdf5")

bbb.issfon=0; bbb.ftol = 1e20
bbb.dtreal=1e-10
bbb.exmain()

#rundt(dtreal=1e-10)

try:
    lambda_exp_fit_out, lambda_eich_fit_out = ana.eich_exp_shahinul_odiv_final()
    lambda_q = lambda_exp_fit_out  # Already in mm
    print(f"    ?? Initial lambda_q = {lambda_q:.3f} mm")
except Exception as e:
    print("Error during initial lambda_q computation:", e)
    sys.exit(1)


