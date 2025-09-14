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
setBoundaryConditions(ncore = 7.78e19, pcoree=2.6e6, pcorei=2.6e6, recycp=0.95, owall_puff=1750.0, dis = 1.75)
setimpmodel(impmodel=True,sput_factor=0.35)
#drift(btry=65)


bbb.lenpfac=120
bbb.cion=3
bbb.oldseec=0
bbb.restart=1
bbb.nusp_imp = 3
bbb.albdso = 0.99
bbb.albdsi = 0.99
bbb.fphysylb  = 0.35

hdf5_restore("./final.hdf5")


bbb.dtreal= 1e-10
bbb.ftol=1e-5
bbb.exmain()

ncore = bbb.ncore[0]

pcore=bbb.pcoree+bbb.pcorei
temid=np.max(bbb.tes[bbb.ixmp,com.iysptrx]/bbb.ev)
tearr=np.array(temid)
pcarr=np.array(pcore)

print("pcore is :", pcore)

while (pcore >= 1e6):

    pcore=pcore-6e5
    print("Trying pcor=", pcore)

    bbb.pcoree = pcore/2
    bbb.pcorei = pcore/2
    print("Pcore e: ", bbb.pcoree)
    print("Pcore i: ", bbb.pcorei)

    bbb.dtreal=1e-10
    print("run with dt", bbb.dtreal)
    bbb.exmain()

    bbb.dtreal=1e-9
    print("run with dt", bbb.dtreal)
    bbb.exmain()
    
    print("Now run rundt")
    
    rundt(dtreal=1e-8)

    
    if bbb.iterm == 1:
        fname = f"tmp_pcore_{pcore:.3e}.h5"
        print("Saving in file:", fname)
        hdf5_save(fname)
    else:
        # Handle failure to converge
        print(f"Failed to converge for pcore = {pcore:.3e}")
        mu.paws("Failed to converge")
         

plt.plot(pcarr/1e6,tearr); plt.plot(pcarr/1e6,tearr,"o")
plt.grid(); plt.title("Te [eV] on sepx. at outer midplane vs. Pcore [MW]")
plt.show()
