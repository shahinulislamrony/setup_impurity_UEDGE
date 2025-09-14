from uedge import *
from uedge.hdf5 import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.tstep as mt
from uedge.rundt import* 
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import os 
import numpy as np
from runcase import *

setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=0.5, kyi=0.5, difni=0.35,nonuniform = False)
setBoundaryConditions(ncore = 7.78e19, pcoree=2.6e6, pcorei=2.6e6, recycp=0.95, owall_puff=0.0, dis = 1.8)
setimpmodel(impmodel=True,sput_factor=0.35)


bbb.lenpfac=120

bbb.cion=3
bbb.oldseec=0
bbb.restart=1
#bbb.nusp_imp = 3
bbb.albdso = 0.99
bbb.albdsi = 0.99
bbb.fphysylb  = 0.35

hdf5_restore("./final.hdf5")
#hdf5_restore("./Li_all_actived.hdf5")

bbb.isbcwdt=1
bbb.dtreal=1e-10
bbb.ftol=1e-6

bbb.exmain()

ncore = bbb.ncore[0]

print("*************")
print("Initial ncore is:", ncore)

rn = bbb.recycp[0]
print("*************")
print("Rn is:", rn)

while (rn >= 0.6):

    rn=rn-0.01
    print("Trying rn=", rn)

    bbb.recycp[0] = rn
    print("RN is : ", bbb.recycp[0])
   

    bbb.dtreal=1e-10; bbb.isbcwdt=1;
    print("run with dt", bbb.dtreal)
    bbb.exmain()

    bbb.dtreal=1e-9; bbb.isbcwdt=1;
    print("run with dt", bbb.dtreal)
    bbb.exmain()

    
    print("Now run rundt")
    
    rundt(dtreal=1e-8)

    
    if bbb.iterm == 1:
        fname = f"tmp_rn_{rn:.3e}.h5"
        print("Saving in file:", fname)
        hdf5_save(fname)
    else:
        # Handle failure to converge
        print(f"Failed to converge for rn = {rn:.3e}")
        mu.paws("Failed to converge")
         
