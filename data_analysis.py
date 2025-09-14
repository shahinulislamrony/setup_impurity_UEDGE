import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import UEDGE_utils.analysis as ana
import UEDGE_utils.plot as utplt
from uedge import *
from uedge.hdf5 import *
from uedge.rundt import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.analysis as mana
import uedge_mvu.tstep as ut
import UEDGE_utils.analysis as ana
from runcase import *
import pandas as pd

setGrid()
setPhysics(impFrac=0, fluxLimit=True)
setDChi(kye=1.0, kyi=1.0, difni=0.5, nonuniform=True)
setBoundaryConditions(ncore=6.2e19, pcoree=2.0e6, pcorei=2.0e6, recycp=0.98)
setimpmodel(impmodel=True)

bbb.cion = 3
bbb.oldseec = 0
bbb.restart = 1
bbb.nusp_imp = 3
bbb.icntnunk = 0

hdf5_restore("./final.hdf5")

bbb.ftol = 1e20
bbb.dtreal = 1e-10
bbb.issfon = 0
bbb.isbcwdt = 1
bbb.exmain()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt = 40e-3  # 20 ms

current_dir = os.getcwd()
hdf5_dir = os.path.join(current_dir, "run_last_iterations")
csv_dir = os.path.join(current_dir, "fngxrb_use")

import os
import numpy as np
import pandas as pd

import os
import numpy as np
import pandas as pd

current_dir = os.getcwd()
hdf5_dir = os.path.join(current_dir, "run_last_iterations")
csv_dir = os.path.join(current_dir, "fngxrb_use")

file_list = sorted([
    f for f in os.listdir(hdf5_dir)
    if f.startswith("run_last_") and f.endswith(".hdf5")
], key=lambda x: int(x.split("_")[-1].split(".")[0]))

nx = len(file_list)
output_data = []

for i in range(nx):
    hdf5_fname = f"run_last_{i}.hdf5"
    csv_fname = f"fngxrb_use_{i}.csv"
    hdf5_path = os.path.join(hdf5_dir, hdf5_fname)
    csv_path = os.path.join(csv_dir, csv_fname)
    print(f"Processing {hdf5_fname}")

    # Compute time for this iteration
    time_value = dt * (i + 1)

    try:
        hdf5_restore(hdf5_path)
        bbb.fngxrb_use[:, 1, 0] = np.loadtxt(csv_path, delimiter=',')
        bbb.exmain()
        bbb.plateflux()
        bbb.pradpltwl()

        # --- Standard power/particle balance ---
        imp_target_Odiv = np.sum(bbb.pwr_pltz[:, 1]*com.sxnp[com.nx+1,:]) / 1e6
        imp_target_Idiv = np.sum(bbb.pwr_pltz[:, 0]*com.sxnp[0,:]) / 1e6
        imp_wall = np.sum(bbb.pwr_wallz*com.sy[:,com.ny+1]) / 1e6
        imp_PFR = np.sum(bbb.pwr_pfwallz[com.ixpt2[0]:com.nx+1] * com.sy[com.ixpt2[0]:com.nx+1, 0])/1e6

        Kinetic = 0.5 * bbb.mi[0] * bbb.up[:, :, 0]**2 * bbb.fnix[:, :, 0]
        pradhyd = np.sum(bbb.pradhyd * com.vol)
        Prad_imp = np.sum(bbb.prad[:, :] * com.vol)
        Total_prad = (np.sum(bbb.erliz + bbb.erlrc) + Prad_imp) / 1e6

        pwrx = bbb.feex + bbb.feix
        pwry = bbb.feey + bbb.feiy
        pbindy = bbb.fniy[:, :, 0] * bbb.ebind * bbb.ev
        pbindx = bbb.fnix[:, :, 0] * bbb.ebind * bbb.ev

        pcore = np.sum(pwry[:, 0]) / 1e6
        pInnerTarget = (np.sum((abs(pwrx) + abs(pbindx))[0, :]) + np.sum(abs(Kinetic[0, :]))) / 1e6
        pOuterTarget = (np.sum((pwrx + pbindx)[com.nx, :]) + np.sum(abs(Kinetic[com.nx, :]))) / 1e6
        pCFWall = (np.sum((pwry + pbindy)[:, com.ny]) + np.sum(abs(Kinetic[:, com.ny]))) / 1e6

        prad_all = (np.sum(abs(bbb.erliz) + abs(bbb.erlrc)) + Prad_imp) / 1e6
        P_SOL = np.sum(bbb.feey[:, com.iysptrx] + bbb.feiy[:, com.iysptrx]) / 1e6

        pPFWallInner = np.sum((-pwry - pbindy)[: com.ixpt1[0] + 1, 0]) / 1e6
        pPFWallOuter = np.sum((-pwry - pbindy)[com.ixpt2[0] + 1 :, 0]) / 1e6
        pPFR = abs(pPFWallOuter) + abs(pPFWallInner)

        fniy_core = np.sum(np.abs(bbb.fniy[:, 0, 0])) / 1e22
        fniy_wall = np.sum(np.abs(bbb.fniy[:, com.ny, 0])) / 1e22
        fnix_odiv = np.sum(np.abs(bbb.fnix[com.nx, :, 0])) / 1e22
        fnix_idiv = np.sum(np.abs(bbb.fnix[0, :, 0])) / 1e22

        S_ion_D = (np.sum(np.abs(bbb.psor[:, :, 0]))) / 1e22
        recombination = (np.sum(np.abs(bbb.psorrg[:, :, 0]))) / 1e22

        # --- Lithium heat flux extraction and sum ---
        data = ana.get_surface_heatflux_components()
        q_Li1 = abs(data['q_Li1_MW'])
        q_Li2 = abs(data['q_Li2_MW'])
        q_Li3 = abs(data['q_Li3_MW'])
        total_q_Li = (q_Li1 + q_Li2 + q_Li3).sum() / 1e6  # MW
        pOuterTarget += total_q_Li
        data = ana.get_surface_heatflux_components(target='inner')
        q_Li1 = data['q_Li1_MW']
        q_Li2 = data['q_Li2_MW']
        q_Li3 = data['q_Li3_MW']
        total_q_Li_in = (q_Li1 + q_Li2 + q_Li3).sum() / 1e6  # MW
        pInnerTarget += total_q_Li_in

        # --- Lithium particle balance ---
        Li_source = (
            np.sum(bbb.fngxrb_use[:, 1, 0]) +
            np.sum(bbb.sputflxrb) +
            np.sum(np.abs(bbb.sputflxlb))
        ) / 1e21  # 1e21 particles/s

        # Ionization and recombination
        Li_ionization = np.sum(np.abs(bbb.psor[:, :, 2])) / 1e21  # Li 1+ ionization
        Li_recombination = np.sum(np.abs(bbb.psorrg[:, :, 1])) / 1e21  # Li 1+ recombination

        # Neutral strike at O-div
        Li_neutral_strike_odiv = np.sum(np.abs(bbb.fngx[com.nx, :, 1])) / 1e21  # Li0
        Li_neutral_strike_idiv = np.sum(np.abs(bbb.fngx[0, :, 1])) / 1e21

        # Ion strike at O-div (Li+1,2,3)
        Li_ion_strike_odiv = np.sum(bbb.fnix[com.nx, :, 2:5]) / 1e21
        Li_ion_strike_idiv = np.sum(np.abs(bbb.fnix[0, :, 2:5])) / 1e21
        Li_ion_strike_wall = np.sum(np.abs(bbb.fniy[:, com.ny, 2:5])) / 1e21
        # Li source 
        Li_source_odiv =  (np.sum(bbb.fngxrb_use[:, 1, 0]) +np.sum(bbb.sputflxrb))/1e21 
        Li_source_idiv =   (np.sum(np.abs(bbb.sputflxlb)))/1e21
        Li_all_flux =  np.sum(abs(bbb.fngx[com.nx,:,1])) + np.sum(abs(bbb.fngx[0,:,1])) + np.sum(abs(bbb.fngy[:,com.ny,1])) + np.sum(abs(bbb.fngy[:,0,1]))


        # Plasma pump-out (non-recycled Li ions)
        Plasma_pump_Odiv = np.sum((1-bbb.recycp[1])*bbb.fnix[com.nx,:,2:5]) / 1e21
        Plasma_pump_Idiv = np.sum(np.abs((1-bbb.recycp[1])*bbb.fnix[0,:,2:5])) / 1e21
        Plasma_pump_wall = np.sum((1-bbb.recycw[1])*bbb.fniy[:,com.ny,2:5]) / 1e21
        total_pump = Plasma_pump_Odiv + Plasma_pump_Idiv + Plasma_pump_wall

        # --- Collect all output ---
        output_data.append([
            time_value, P_SOL, pcore, pInnerTarget, pOuterTarget, pCFWall,
            prad_all, pPFR, fniy_core, fniy_wall, fnix_odiv, fnix_idiv,
            S_ion_D, recombination,
            imp_target_Odiv, imp_target_Idiv, imp_wall, imp_PFR,
            # Li particle balance:
            Li_source, Li_ionization, Li_recombination,
            Li_neutral_strike_odiv, Li_neutral_strike_idiv,
            Li_ion_strike_odiv, Li_ion_strike_idiv, Li_ion_strike_wall,
            Plasma_pump_Odiv, Plasma_pump_Idiv, Plasma_pump_wall, total_pump, Li_source_odiv, Li_source_idiv 
        ])

    except Exception as e:
        print(f"  >> Failed: {e}")


columns = [
    "time [s]", "P_SOL [MW]", "P_core [MW]", "P_inner [MW]", "P_outer [MW]", "P_wall [MW]",
    "P_rad [MW]", "P_PFR [MW]", "Gamma_core [1e22/s]", "Gamma_wall [1e22/s]",
    "Gamma_ODiv [1e22/s]", "Gamma_IDiv [1e22/s]", "Ionization [1e22/s]", "Recombination [1e22/s]",
    "imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]",
    # Li particle balance:
    "Li_source [1e21/s]", "Li_ionization [1e21/s]", "Li_recombination [1e21/s]",
    "Li_neutral_strike_odiv [1e21/s]", "Li_neutral_strike_idiv [1e21/s]",
    "Li_ion_strike_odiv [1e21/s]", "Li_ion_strike_idiv [1e21/s]", "Li_ion_strike_wall [1e21/s]",
    "Li_pump_odiv [1e21/s]", "Li_pump_idiv [1e21/s]", "Li_pump_wall [1e21/s]", "Li_total_pump [1e21/s]", "Li_source_odiv [1e21/s]", "Li_source_idiv [1e21/s]"
]
df = pd.DataFrame(output_data, columns=columns)
df.to_csv("power_particle_balance_summary.csv", index=False)
print("\nSaved: power_particle_balance_summary.csv")

df["P_total [MW]"] = (
    df["P_outer [MW]"] +
    df["P_inner [MW]"] +
    df["P_wall [MW]"] +
    df["P_rad [MW]"] + df["P_PFR [MW]"]
)


plt.figure(figsize=(5,3))
plt.plot(df["time [s]"], df["Li_source [1e21/s]"]*1e21, label="Li Source")
plt.plot(df["time [s]"], df["Li_ionization [1e21/s]"]*1e21, label="Li Ionization")
plt.plot(df["time [s]"], df["Li_recombination [1e21/s]"]*1e21, label="Li Recombination")
plt.xlabel("Time [s]")
plt.ylabel("Rate [1e21/s]")
plt.title("Lithium Particle Balance vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Lithium_particle.png', dpi=300)
plt.show()

plt.figure(figsize=(5,3))
plt.plot(df["time [s]"], df["Li_neutral_strike_odiv [1e21/s]"], label="Li Neutral Strike ODiv")
plt.plot(df["time [s]"], df["Li_neutral_strike_idiv [1e21/s]"], label="Li Neutral Strike IDiv")
plt.plot(df["time [s]"], df["Li_ion_strike_odiv [1e21/s]"], label="Li Ion Strike ODiv")
plt.plot(df["time [s]"], df["Li_ion_strike_idiv [1e21/s]"], label="Li Ion Strike IDiv")
plt.xlabel("Time [s]")
plt.ylabel("Strike Rate [1e21/s]")
plt.title("Li Strikes at Divertor vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Lithium_ion_neu.png', dpi=300)
plt.show()

plt.figure(figsize=(5,3))
plt.plot(df["time [s]"], df["Li_source [1e21/s]"], label="Li Source")
plt.plot(df["time [s]"], df["Li_total_pump [1e21/s]"], label="Li Pump")
plt.plot(df["time [s]"], df["Li_ionization [1e21/s]"], label="Li Ionization")
plt.xlabel("Time [s]")
plt.ylabel("Pumped [1e21/s]")
plt.title("Li Pumped vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Lithium_pump.png', dpi=300)
plt.show()

plt.figure(figsize=(5,3))
plt.plot(df["time [s]"], df["P_SOL [MW]"], marker='h', label="SOL")
plt.plot(df["time [s]"], df["P_outer [MW]"], marker='o', label="Odiv")
plt.plot(df["time [s]"], df["P_inner [MW]"], marker='s', label="Idiv")
plt.plot(df["time [s]"], df["P_wall [MW]"], marker='^', label="O-wall")
plt.plot(df["time [s]"], df["P_rad [MW]"], marker='d', label="P-rad")
plt.plot(df["time [s]"], df["P_total [MW]"], marker='x', label="Total", color='k', linewidth=1.5)
ymax1 = df[["P_SOL [MW]", "P_outer [MW]", "P_inner [MW]", "P_wall [MW]", "P_rad [MW]", "P_total [MW]"]].max().max()

plt.xlabel("t$_{simulation}$ (s)")
plt.ylabel("Power [MW]")

plt.legend()
plt.xlim([0, np.max(df["time [s]"])*1.05])
plt.ylim([0, ymax1*1.05])
plt.yscale('log')
plt.ylim([1e-3, ymax1*1.05])

plt.grid(True)
plt.tight_layout()
plt.savefig('power.png', dpi=300)
plt.show()

df["power_balance_error_percent"] = (abs(df["P_total [MW]"] - df["P_SOL [MW]"])) / df["P_total [MW]"] * 100
plt.figure(figsize=(5,3))
plt.plot(df["time [s]"], df["power_balance_error_percent"], marker='x', label="Power Balance Error (%)")
plt.xlabel("t$_{simulation}$ (s)", fontsize=16)
plt.ylabel("Error (%)", fontsize=16)
plt.title(r"Power Balance: $\mathrm{Error} = \frac{|P_\mathrm{total} - P_\mathrm{SOL}|}{P_\mathrm{total}} \times 100$")
plt.legend()
plt.ylim([0, 10])
plt.grid(True)
plt.savefig('power_error.png', dpi=300)
plt.show()

plt.figure(figsize=(5,3))
plt.plot(df["time [s]"], df["imp_target_Odiv [MW]"], marker='o', label="Odiv")
plt.plot(df["time [s]"], df["imp_target_Idiv [MW]"], marker='s', label="Idiv")
plt.plot(df["time [s]"], df["imp_wall [MW]"], marker='^', label="O-wall")
#plt.plot(df["time [s]"], df["imp_PFR [MW]"], marker='d', label="PFR")
ymax2 = df[["imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]"]].max().max()

plt.xlabel("t$_{simulation}$ (s)", fontsize = 16)
plt.ylabel("Li rad flux [MW]", fontsize = 16)
plt.title("Impurity Power Channels vs Time")
plt.legend()
plt.grid(True)
plt.ylim([0, ymax2*1.05])
plt.xlim([0, np.max(df["time [s]"])*1.05])
plt.yscale('log')
plt.ylim([1e-3, ymax2*1.05])

plt.tight_layout()
plt.savefig('Lithium_rad.png', dpi=300)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

# First subplot: Power channels
axs[0].plot(df["time [s]"], df["P_SOL [MW]"], marker='h', label="SOL")
axs[0].plot(df["time [s]"], df["P_outer [MW]"], marker='o', label="Odiv")
axs[0].plot(df["time [s]"], df["P_inner [MW]"], marker='s', label="Idiv")
axs[0].plot(df["time [s]"], df["P_wall [MW]"]+df["P_PFR [MW]"], marker='^', label="O-wall")
axs[0].plot(df["time [s]"], df["P_rad [MW]"], marker='d', label="P-rad")
# If you have P_total [MW], uncomment below:
# axs[0].plot(df["time [s]"], df["P_total [MW]"], marker='x', label="Total", color='k', linewidth=1.5)

ymax1 = df[["P_SOL [MW]", "P_outer [MW]", "P_inner [MW]", "P_wall [MW]", "P_rad [MW]"]].max().max()
axs[0].set_ylim([0, ymax1*1.05])
axs[0].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[0].set_ylabel("Power [MW]", fontsize=16)
axs[0].legend(loc='best', ncol=2, fontsize=10)
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=12)

# Second subplot: Impurity power channels
axs[1].plot(df["time [s]"], df["imp_target_Odiv [MW]"], marker='o', label="Odiv")
axs[1].plot(df["time [s]"], df["imp_target_Idiv [MW]"], marker='s', label="Idiv")
axs[1].plot(df["time [s]"], df["imp_wall [MW]"], marker='^', label="O-wall")
axs[1].plot(df["time [s]"], df["imp_PFR [MW]"], marker='d', label="PFR")

ymax2 = df[["imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]"]].max().max()
axs[1].set_ylim([0, ymax2*1.05])
axs[1].set_xlabel("t$_{simulation}$ (s)", fontsize=16)
axs[1].set_ylabel("Li rad flux [MW]", fontsize=16)
axs[1].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[1].legend(loc='best', ncol=2, fontsize=10)
axs[1].grid(True)
axs[1].tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('combined_Lithium_rad.png', dpi=300)
plt.show()


fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

# First subplot: Power channels (log scale)
axs[0].plot(df["time [s]"], df["P_SOL [MW]"], marker='h', label="SOL")
axs[0].plot(df["time [s]"], df["P_outer [MW]"], marker='o', label="Odiv")
axs[0].plot(df["time [s]"], df["P_inner [MW]"], marker='s', label="Idiv")
axs[0].plot(df["time [s]"], df["P_wall [MW]"]+df["P_PFR [MW]"], marker='^', label="O-wall")
axs[0].plot(df["time [s]"], df["P_rad [MW]"], marker='d', label="P-rad")
axs[0].plot(df["time [s]"], df["P_total [MW]"], marker='x', label="Total", color='k', linewidth=1.5)

ymin1 = df[["P_SOL [MW]", "P_outer [MW]", "P_inner [MW]", "P_wall [MW]", "P_rad [MW]", "P_total [MW]"]].replace(0, np.nan).min().min()
ymax1 = df[["P_SOL [MW]", "P_outer [MW]", "P_inner [MW]", "P_wall [MW]", "P_rad [MW]", "P_total [MW]"]].max().max()
axs[0].set_ylim([ymin1*0.95, ymax1*1.05])
axs[0].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[0].set_ylabel("Power [MW]", fontsize=16)
axs[0].set_yscale('log')
axs[0].grid(True, which='both')
axs[0].tick_params(axis='both', labelsize=12)
# Legend above plot, no box
axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=10, frameon=False)

# Second subplot: Impurity power channels (log scale)
axs[1].plot(df["time [s]"], df["imp_target_Odiv [MW]"], marker='o', label="Odiv")
axs[1].plot(df["time [s]"], df["imp_target_Idiv [MW]"], marker='s', label="Idiv")
axs[1].plot(df["time [s]"], df["imp_wall [MW]"], marker='^', label="O-wall")
axs[1].plot(df["time [s]"], df["imp_PFR [MW]"], marker='d', label="PFR")

ymin2 = df[["imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]"]].replace(0, np.nan).min().min()
ymax2 = df[["imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]"]].max().max()
axs[1].set_ylim([ymin2*0.95, ymax2*1.05])
axs[1].set_xlabel("t$_{simulation}$ (s)", fontsize=16)
axs[1].set_ylabel("Li rad flux [MW]", fontsize=16)
axs[1].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[1].set_yscale('log')
axs[1].grid(True, which='both')
axs[1].tick_params(axis='both', labelsize=12)
# Legend in lower right, no box
axs[1].legend(loc='lower right', fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig('combined_Lithium_rad_log.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

# First subplot: Power channels
axs[0].plot(df["time [s]"], df["P_SOL [MW]"], marker='h', label="SOL")
axs[0].plot(df["time [s]"], df["P_outer [MW]"], marker='o', label="Odiv")
axs[0].plot(df["time [s]"], df["P_inner [MW]"], marker='s', label="Idiv")
axs[0].plot(df["time [s]"], df["P_wall [MW]"]+df["P_PFR [MW]"], marker='^', label="O-wall")
axs[0].plot(df["time [s]"], df["P_rad [MW]"], marker='d', label="P-rad")
# If you have P_total [MW], uncomment below:
axs[0].plot(df["time [s]"], df["P_total [MW]"], marker='x', label="Total", color='k', linewidth=1.5)

ymax1 = df[["P_SOL [MW]", "P_outer [MW]", "P_inner [MW]", "P_wall [MW]", "P_rad [MW]", "P_total [MW]" ]].max().max()
axs[0].set_yscale('log')
axs[0].set_ylim([1e-1, 4])
axs[0].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[0].set_ylabel("Power [MW]", fontsize=16)
axs[0].legend(loc='best', ncol=2, fontsize=10)
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=12)

# Second subplot: Impurity power channels
axs[1].plot(df["time [s]"], df["imp_target_Odiv [MW]"], marker='o', label="Odiv")
axs[1].plot(df["time [s]"], df["imp_target_Idiv [MW]"], marker='s', label="Idiv")
axs[1].plot(df["time [s]"], df["imp_wall [MW]"], marker='^', label="O-wall")
axs[1].plot(df["time [s]"], df["imp_PFR [MW]"], marker='d', label="PFR")

ymax2 = df[["imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]"]].max().max()
axs[1].set_yscale('log')
axs[1].set_ylim([1e-2, ymax2*1.05])
axs[1].set_xlabel("t$_{simulation}$ (s)", fontsize=16)
axs[1].set_ylabel("Li rad flux [MW]", fontsize=16)
axs[1].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[1].legend(loc='best', ncol=2, fontsize=10)
axs[1].grid(True)
axs[1].tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('combined_Lithium_rad_log1.png', dpi=300)
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

# First subplot: Power channels (log scale)
axs[0].plot(df["time [s]"], df["P_SOL [MW]"],  label="SOL")
axs[0].plot(df["time [s]"], df["P_outer [MW]"],  label="Odiv")
axs[0].plot(df["time [s]"], df["P_inner [MW]"],  label="Idiv")
axs[0].plot(df["time [s]"], df["P_wall [MW]"]+df["P_PFR [MW]"], label="O-wall")
axs[0].plot(df["time [s]"], df["P_rad [MW]"],  label="P-rad")
axs[0].plot(df["time [s]"], df["P_total [MW]"], label="Total", linewidth=1.5)

ymin1 = df[["P_SOL [MW]", "P_outer [MW]", "P_inner [MW]", "P_wall [MW]", "P_rad [MW]", "P_total [MW]"]].replace(0, np.nan).min().min()
ymax1 = df[["P_SOL [MW]", "P_outer [MW]", "P_inner [MW]", "P_wall [MW]", "P_rad [MW]", "P_total [MW]"]].max().max()
axs[0].set_ylim([1e-1, 4])
axs[0].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[0].set_ylabel("Power [MW]", fontsize=16)
axs[0].set_yscale('log')
axs[0].set_xlim([0, 5])
axs[0].grid(True, which='both')
axs[0].tick_params(axis='both', labelsize=12)
# Legend above plot, no box
axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=10, frameon=False)

# Second subplot: Impurity power channels (log scale)
axs[1].plot(df["time [s]"], df["imp_target_Odiv [MW]"], label="Odiv")
axs[1].plot(df["time [s]"], df["imp_target_Idiv [MW]"],  label="Idiv")
axs[1].plot(df["time [s]"], df["imp_wall [MW]"], label="O-wall")
axs[1].plot(df["time [s]"], df["imp_PFR [MW]"],  label="PFR")

ymin2 = df[["imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]"]].replace(0, np.nan).min().min()
ymax2 = df[["imp_target_Odiv [MW]", "imp_target_Idiv [MW]", "imp_wall [MW]", "imp_PFR [MW]"]].max().max()
axs[1].set_ylim([1e-3, 5e-1])
axs[1].set_xlabel("t$_{simulation}$ (s)", fontsize=16)
axs[1].set_ylabel("Li rad flux [MW]", fontsize=16)
axs[1].set_xlim([0, np.max(df["time [s]"])*1.05])
axs[1].set_yscale('log')
axs[1].set_xlim([0, 5])
axs[1].grid(True, which='both')
axs[1].tick_params(axis='both', labelsize=12)
# Legend in lower right, no box
axs[1].legend(loc='lower right', fontsize=10, frameon=False)
axs[1].legend(loc='lower right', fontsize=10, ncol=2)

plt.tight_layout()
plt.savefig('combined_Lithium_rad_log.png', dpi=300, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Assign a unique color to each variable
color_sol = 'tab:orange'
color_odiv = 'tab:blue'
color_idiv = 'tab:red'
color_owall = 'tab:green'
color_prad = 'tab:cyan'
color_total = 'tab:black'
color_pfr = 'tab:purple'

color_imp_odiv =  'tab:blue'
color_imp_idiv = 'tab:red'
color_imp_owall = 'tab:green'
color_imp_pfr = 'tab:purple'

fig, axs = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

# First subplot: Power channels (log scale)
axs[0].plot(df["time [s]"], df["P_SOL [MW]"], label="SOL", color=color_sol, linewidth=3)
axs[0].plot(df["time [s]"], df["P_outer [MW]"], label="Odiv", color=color_odiv, linewidth=3)
axs[0].plot(df["time [s]"], df["P_inner [MW]"], label="Idiv", color=color_idiv, linewidth=3)
axs[0].plot(df["time [s]"], df["P_wall [MW]"] + df["P_PFR [MW]"], label="O-wall", color=color_owall, linewidth=3)
axs[0].plot(df["time [s]"], df["P_rad [MW]"], label="P-rad", color=color_prad, linewidth=3)
axs[0].plot(df["time [s]"], df["P_total [MW]"], label="Total", color='black', linewidth=3)

axs[0].set_ylim([1e-1, 4])
axs[0].set_xlim([0, np.max(df["time [s]"]) * 1.05])
axs[0].set_ylabel("Power [MW]", fontsize=16)
axs[0].set_yscale('log')
axs[0].set_xlim([0, 5])
axs[0].grid(True, which='both')
axs[0].tick_params(axis='both', labelsize=12)
axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=10, frameon=False)

axs[1].plot(df["time [s]"], df["imp_target_Odiv [MW]"], label="Odiv", color=color_imp_odiv, linewidth=3)
axs[1].plot(df["time [s]"], df["imp_target_Idiv [MW]"], label="Idiv", color=color_imp_idiv, linewidth=3)
axs[1].plot(df["time [s]"], df["imp_wall [MW]"], label="O-wall", color=color_imp_owall, linewidth=3)
axs[1].plot(df["time [s]"], df["imp_PFR [MW]"], label="PFR", color=color_imp_pfr, linewidth=3)

axs[1].set_ylim([1e-3, 5e-1])
axs[1].set_xlabel("t$_{simulation}$ (s)", fontsize=16)
axs[1].set_ylabel("Li rad flux [MW]", fontsize=16)
axs[1].set_xlim([0, np.max(df["time [s]"]) * 1.05])
axs[1].set_yscale('log')
axs[1].set_xlim([0, 5])
axs[1].grid(True, which='both')
axs[1].tick_params(axis='both', labelsize=12)
axs[1].legend(loc='lower right', fontsize=10, ncol=2)

plt.tight_layout()
plt.savefig('combined_Lithium_rad_log.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 5))
axs[0].plot(df["time [s]"], df["Li_source_odiv [1e21/s]"], label="Odiv", linewidth= '2')
axs[0].plot(df["time [s]"], df["Li_source_idiv [1e21/s]"], label="Idiv", linewidth= '2')
axs[0].set_ylabel(r'$\phi_{Li}^{source}$ (10$^{21}$ /s)', fontsize=16)
axs[0].set_yscale('log')
axs[0].set_ylim([1e-2, 100])
axs[0].set_xlim([0, 5])
axs[0].tick_params(axis='both', labelsize=14)
axs[0].minorticks_on()
axs[0].grid(True, which='major', linestyle='-', linewidth=1.0, alpha=0.8)
axs[0].grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.3)
axs[0].legend(loc="best")
axs[0].text(0.02, 0.95, "(a)", transform=axs[0].transAxes, fontsize=16, va='top', ha='left')

axs[1].plot(df["time [s]"], df["Li_ion_strike_odiv [1e21/s]"], label="Odiv", linewidth= '2')
axs[1].plot(df["time [s]"], df["Li_ion_strike_idiv [1e21/s]"], label="Idiv", linewidth= '2')
axs[1].plot(df["time [s]"], df["Li_ion_strike_wall [1e21/s]"], label="Wall", linewidth= '2')
axs[1].set_xlabel(r't$_{simulation}$ (s)', fontsize=18)
axs[1].set_ylabel(r'$\phi_{Li}^{deposit}$ (10$^{21}$ /s)', fontsize=16)
axs[1].set_yscale('log')
axs[1].set_ylim([1e-2, 100])
axs[1].tick_params(axis='both', labelsize=14)
axs[1].minorticks_on()
axs[1].grid(True, which='major', linestyle='-', linewidth=1.0, alpha=0.8)
axs[1].grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.3)
axs[1].legend(loc="center", ncol=2)

axs[1].text(0.02, 0.95, "(b)", transform=axs[1].transAxes, fontsize=16, va='top', ha='left')

plt.tight_layout()
plt.savefig('Li_flux_x.png', dpi=600, bbox_inches='tight')
plt.show()



# Assign colors
color_odiv = 'tab:blue'
color_idiv = 'tab:red'
color_wall = 'tab:green'

fig, ax = plt.subplots(figsize=(5, 4))

# Plot incident (source) fluxes
line_inc_odiv, = ax.plot(df["time [s]"], df["Li_source_odiv [1e21/s]"], label="Odiv", color=color_odiv, linewidth=2)
line_inc_idiv, = ax.plot(df["time [s]"], df["Li_source_idiv [1e21/s]"], label="Idiv", color=color_idiv, linewidth=2)

# Plot deposit (strike) fluxes
line_dep_odiv, = ax.plot(df["time [s]"], df["Li_ion_strike_odiv [1e21/s]"], label="Odiv", color=color_odiv, linewidth=3, linestyle='--')
line_dep_idiv, = ax.plot(df["time [s]"], df["Li_ion_strike_idiv [1e21/s]"], label="Idiv", color=color_idiv, linewidth=3, linestyle='--')
line_dep_wall, = ax.plot(df["time [s]"], df["Li_ion_strike_wall [1e21/s]"], label="Wall", color=color_wall, linewidth=3, linestyle='--')

ax.set_xlabel(r't$_{simulation}$ (s)', fontsize=16)
ax.set_ylabel(r'Li flux (10$^{21}$ /s)', fontsize=16)
ax.set_yscale('log')
ax.set_ylim([1e-2, 100])
ax.set_xlim([0, 5])
ax.tick_params(axis='both', labelsize=14)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-', linewidth=1.0, alpha=0.8)
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.3)

# Create separate legend handles
incident_handles = [line_inc_odiv, line_inc_idiv]
deposit_handles = [line_dep_odiv, line_dep_idiv, line_dep_wall]

# Place legends outside the plot, at the top
legend1 = ax.legend(incident_handles, ["Odiv", "Idiv"], 
                    title="Sourcing", fontsize=12, title_fontsize=12, 
                    loc='lower left', bbox_to_anchor=(0.05, 1.02), frameon=True)
legend2 = ax.legend(deposit_handles, ["Odiv", "Idiv", "Wall"], 
                    title="Deposit", fontsize=12, title_fontsize=12, 
                    loc='lower right', bbox_to_anchor=(0.95, 1.02), frameon=True, ncol=2)

ax.add_artist(legend1)  # Add the first legend manually

plt.tight_layout()
plt.savefig('Li_flux_x_single_top_outside.png', dpi=600, bbox_inches='tight')
plt.show()