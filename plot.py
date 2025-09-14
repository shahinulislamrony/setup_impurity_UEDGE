import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from uedge import *
from uedge.hdf5 import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.tstep as mt
from uedge.rundt import *
from runcase import *

# Set up UEDGE parameters
setGrid()
setPhysics(impFrac=0, fluxLimit=True)
setDChi(kye=0.5, kyi=0.5, difni=0.35, nonuniform=False)
setBoundaryConditions(ncore=7.78e19, pcoree=2.0e6, pcorei=2.0e6, recycp=0.95, owall_puff=1750.0, dis=1.75)
setimpmodel(impmodel=True, sput_factor=0.35)
#drift(btry=9)

bbb.lenpfac=120

bbb.cion=3
bbb.oldseec=0
bbb.restart=1
bbb.nusp_imp = 3
bbb.albdso = 0.99
bbb.albdsi = 0.99
bbb.fphysylb  = 0.35

hdf5_restore("./tmp_pcore_4.000e+06.h5")

bbb.dtreal= 1e-10
bbb.ftol=1e20
bbb.issfon = 0
bbb.exmain()

print("**************************")
mu.paws("****Done 65, now entering loop")




def process_dataset(data_path, file_name):
    try:
        b0_str = file_name.replace("tmp_pcore_", "").replace(".h5", "")
        b0 = float(b0_str)
        print(f"Processing file: {file_name}")
        hdf5_restore(os.path.join(data_path, file_name))
    except Exception as e:
        print(f"Error restoring HDF5 data for {file_name}: {e}")
        return None

    try:
        bbb.pcoree = b0/2
        bbb.pcorei = b0/2
        bbb.ftol = 1e20
        bbb.issfon = 0
        bbb.exmain()
    except Exception as e:
        print(f"Error during simulation setup for {file_name}: {e}")
        return None

    try:
        nemax_odiv = np.max(bbb.ne[com.nx, :])
        nemax_idiv = np.max(bbb.ne[0, :])
        Te_max_odiv = np.max(bbb.te[com.nx, :] / bbb.ev)
        Te_max_idiv = np.max(bbb.te[0, :] / bbb.ev)

        bbb.pradpltwl()
        bbb.plateflux()

        q_data = (bbb.sdrrb + bbb.sdtrb).reshape(-1)
        q_idiv_data = (bbb.sdrlb + bbb.sdtlb).reshape(-1)

        max_q = np.max(q_data)
        max_q_idiv = np.max(q_idiv_data)
        q_int_odiv = np.sum(q_data*com.sxnp[com.nx,:])/ 1e6
        q_int_idiv = np.sum(q_idiv_data*com.sxnp[0,:])/ 1e6

        Kinetic = 0.5 * bbb.mi[0] * bbb.up[:, :, 0]**2 * bbb.fnix[:, :, 0]
        pradhyd = np.sum(bbb.pradhyd * com.vol)
        Prad_imp = np.sum(bbb.prad[:, :] * com.vol)
        Total_prad = (np.sum(bbb.erliz + bbb.erlrc) + Prad_imp)/ 1e6

        pwrx = bbb.feex + bbb.feix
        pwry = bbb.feey + bbb.feiy
        pbindy = bbb.fniy[:, :, 0] * bbb.ebind * bbb.ev

        q_core = np.sum(pwry[:, 0]) / 1e6
        q_int_wall = (np.sum((pwry + pbindy)[:, com.ny]) + np.sum(np.abs(Kinetic[:, com.ny]))) / 1e6
        q_sep = np.sum(bbb.feey[:, com.iysptrx] + bbb.feiy[:, com.iysptrx]) / 1e6

        n_Li_all_sep = (
            bbb.ni[bbb.ixmp, com.iysptrx, 2]
            + bbb.ni[bbb.ixmp, com.iysptrx, 3]
            + bbb.ni[bbb.ixmp, com.iysptrx, 4]
        )
        ne_all_sep = bbb.ne[bbb.ixmp, com.iysptrx]
        Te_omp_sep = bbb.te[bbb.ixmp, com.iysptrx]/bbb.ev
        C_Li_all_sep = n_Li_all_sep / ne_all_sep
        Emit_Li_flux_Odiv = np.sum(bbb.sputflxrb)
        Emit_Li_flux_Idiv = np.abs(np.sum(bbb.sputflxlb))
        Li_Odiv = np.sum(bbb.fnix[com.nx,:,2]) + np.sum(bbb.fnix[com.nx,:,3]) + np.sum(bbb.fnix[com.nx,:,4])
        Li_Idiv = np.sum(bbb.fnix[0,:,2]) + np.sum(bbb.fnix[0,:,3]) + np.sum(bbb.fnix[0,:,4])
        Li_Idiv = np.abs(Li_Idiv)
        Li_wall = np.sum(bbb.fnix[:,com.ny,2]) + np.sum(bbb.fniy[:,com.ny,3]) + np.sum(bbb.fniy[:,com.ny,4])

        return {
            "file_name": file_name,
            "b0": b0,
            "nemax_odiv": nemax_odiv,
            "nemax_idiv": nemax_idiv,
            "max_q": max_q,
            "max_q_idiv": max_q_idiv,
            "q_core": q_core,
            "q_int_wall": q_int_wall,
            "q_sep": q_sep,
            "prad_all": Total_prad,
            "Te_max_odiv": Te_max_odiv,
            "Te_max_idiv": Te_max_idiv,
            "C_Li_omp": C_Li_all_sep,
            "ne_sepm_omp": ne_all_sep,
            "q_int_idiv": q_int_idiv,
            "q_int_odiv": q_int_odiv,
            "n_Li_all_sep": n_Li_all_sep,
            "Te_omp_sep": Te_omp_sep,
            "Emit_Li_flux_Odiv": Emit_Li_flux_Odiv,
            "Emit_Li_flux_Idiv": Emit_Li_flux_Idiv,
            "Li_Odiv": Li_Odiv,
            "Li_Idiv": Li_Idiv,
            "Li_wall": Li_wall,
                             
        }

    except Exception as e:
        print(f"Error extracting data for {file_name}: {e}")
        return None

def process_all_datasets(data_path, ncore_list):
    results = []
    file_names = [f"tmp_pcore_{b0:.3e}.h5" for b0 in ncore_list]

    for file_name in file_names:
        result = process_dataset(data_path, file_name)
        if result is not None:
            results.append(result)

    return results

def plot_results(all_results):
    ncore_scan = []
    nemax_odiv = []
    nemax_idiv = []
    max_q = []
    max_q_idiv = []
    Te_max_odiv = []
    Te_max_idiv = []
    q_wall = []
    q_sep = []
    q_core = []
    C_Li_omp = []
    ne_sepm_omp = []
    prad_all = []
    q_int_idiv = []
    q_int_odiv = []
    Te_sepm_omp = []
    nli_sepm_omp = []
    Li_Odiv =[]
    Li_Idiv =[]
    Li_wall =[]
    Emit_Li_flux_Odiv = []
    Emit_Li_flux_Idiv = []
    

    for res in all_results:
        ncore_scan.append(res["b0"])
        nemax_odiv.append(res["nemax_odiv"])
        nemax_idiv.append(res["nemax_idiv"])
        max_q.append(res["max_q"])
        max_q_idiv.append(res["max_q_idiv"])
        Te_max_odiv.append(res["Te_max_odiv"])
        Te_max_idiv.append(res["Te_max_idiv"])
        C_Li_omp.append(res["C_Li_omp"])
        ne_sepm_omp.append(res["ne_sepm_omp"])
        nli_sepm_omp.append(res["n_Li_all_sep"])
        Te_sepm_omp.append(res["Te_omp_sep"])
        q_core.append(res["q_core"])
        prad_all.append(res["prad_all"])
        q_sep.append(res["q_sep"])
        q_wall.append(res["q_int_wall"])
        q_int_idiv.append(res["q_int_idiv"])
        q_int_odiv.append(res["q_int_odiv"])
        Emit_Li_flux_Odiv.append(res["Emit_Li_flux_Odiv"])
        Emit_Li_flux_Idiv.append(res["Emit_Li_flux_Idiv"])
        Li_Odiv.append(res["Li_Odiv"])
        Li_Idiv.append(res["Li_Idiv"])
        Li_wall.append(res["Li_wall"])

    # ?? Check array lengths before plotting
    print("\n--- Sanity Check: Data Lengths ---")
    print(f"ncore_scan: {len(ncore_scan)}")
    print(f"q_core:     {len(q_core)}")
    print(f"q_sep:      {len(q_sep)}")
    print(f"q_wall:     {len(q_wall)}")
    print("----------------------------------")

    # ?? Print actual q_sep values
    print("\n--- q_sep values ---")
    for b0, val in zip(ncore_scan, q_sep):
        print(f"b0 = {b0:.3f}, q_sep = {val:.4f} MW")

    fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    axs[0, 0].plot(np.divide(ncore_scan, 1e6), np.array(nemax_odiv) / 1e20, marker="o", linestyle="--", label="Odiv")
    axs[0, 0].plot(np.divide(ncore_scan, 1e6), np.array(nemax_idiv) / 1e20, marker="*", linestyle="-", label="Idiv")
    axs[0, 0].set_ylabel("$n_{e,max}$ ($10^{20}$ m$^{-3}$)", fontsize=14)
    axs[0, 0].set_ylim([0, 8])
    axs[0, 0].legend(fontsize=14, loc="lower center")
    axs[0, 0].grid(True)

    axs[0, 1].plot(np.divide(ncore_scan, 1e6), np.divide(max_q, 1e6), marker="o", linestyle="--", label="Odiv")
    axs[0, 1].plot(np.divide(ncore_scan, 1e6), np.divide(max_q_idiv, 1e6), marker="*", linestyle="-", label="Idiv")
    axs[0, 1].set_ylabel("$q_\\perp^{max}$ (MW/m$^2$)", fontsize=14)
    axs[0, 1].set_ylim([0, 18])
    axs[0, 1].grid(True)

    axs[0, 2].plot(np.divide(ncore_scan, 1e6), Te_max_odiv, marker="o", linestyle="--", label="Odiv")
    axs[0, 2].plot(np.divide(ncore_scan, 1e6), Te_max_idiv, marker="*", linestyle="-", label="Idiv")  # FIXED here
    axs[0, 2].set_ylabel("$T_e^{max}$ (eV)", fontsize=14)
    axs[0, 2].set_ylim([0, 120])
    axs[0, 2].grid(True)

    axs[1, 0].plot(np.divide(ncore_scan, 1e6), np.divide(nli_sepm_omp,1e16), marker="o", linestyle="--") 
    axs[1, 0].set_xlabel("Pcore (MW)",fontsize=14)
    axs[1, 0].set_ylabel("n$_{Li,sep}^{omp}$ ($10^{16}$ m$^{-3}$)", fontsize=14)
    axs[1, 0].set_ylim([0, 2])
    axs[1, 0].grid(True)


    axs[1, 1].plot(np.divide(ncore_scan, 1e6), np.array(ne_sepm_omp) / 1e19, marker="o", linestyle="--")
    axs[1, 1].set_xlabel("Pcore (MW)",fontsize=14)
    axs[1, 1].set_ylabel("$n_{e,sep}^{omp}$ ($10^{19}$ m$^{-3}$)", fontsize=14)
    axs[1, 1].set_ylim([0, 6.2])
    axs[1, 1].grid(True)


    axs[1, 2].plot(np.divide(ncore_scan, 1e6), np.array(Te_sepm_omp), marker="o", linestyle="--")
    axs[1, 2].set_xlabel("Pcore (MW)",fontsize=14)
    axs[1, 2].set_ylabel("$T_{e,sep}^{omp}$ (eV)", fontsize=14)
    axs[1, 2].set_ylim([0, 200])
    axs[1, 2].grid(True)


    plt.tight_layout()
    plt.savefig("All.png", dpi=300)
    plt.show()


    plt.figure(figsize=(6, 4))
    #plt.plot(np.divide(ncore_scan, 1e6), (q_core), marker="o", linestyle="--", color="b", label='core')
    plt.plot(np.divide(ncore_scan, 1e6), (q_sep), marker="*", linestyle="--", color="k", label='sep')
    plt.plot(np.divide(ncore_scan, 1e6), (q_int_idiv), marker="d", linestyle="--", color="r", label='idiv')
    plt.plot(np.divide(ncore_scan, 1e6), (q_int_odiv), marker="s", linestyle="-", color="g", label='odiv')
    plt.plot(np.divide(ncore_scan, 1e6), (q_wall), marker="o", linestyle=":", color="c", label='wall')
    plt.plot(np.divide(ncore_scan, 1e6), (prad_all), marker="<", linestyle="-.", color="m", label='rad')
    plt.xlabel("Pcore (MW)",fontsize=14)
    plt.ylabel("Power (MW)", fontsize = 14)
    plt.ylim([0.0, 10])
    #plt.yscale('log')
    plt.title("Power balance")
    plt.legend(fontsize=10, ncol=3, loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('power.png', dpi=300)
    plt.show() 

    plt.figure(figsize=(5, 3))
    plt.plot(np.divide(ncore_scan, 1e6), np.divide(max_q,1e6), marker="o", linestyle="--", color="b", label='Odiv')
    plt.plot(np.divide(ncore_scan, 1e6), np.divide(max_q_idiv,1e6), marker="*", linestyle="--", color="k", label='Idiv')
    plt.xlabel("b0",fontsize=14)
    plt.ylabel("q$_{max} (MW/m^2)$", fontsize=14)
    plt.title("Power balance",fontsize=14)
    plt.legend(fontsize=12, ncol=2)
    plt.grid(True)
    plt.ylim([0, 18])
    plt.tight_layout()
    plt.savefig('q_idiv_odiv.png', dpi=300)
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.plot(np.divide(ncore_scan, 1e6), np.divide(Li_Odiv,1e20), marker="o", linestyle="--", color="b", label='Odiv')
    plt.plot(np.divide(ncore_scan, 1e6), np.divide(Li_Idiv,1e20), marker="*", linestyle="--", color="k", label='Idiv')
    plt.plot(np.divide(ncore_scan, 1e6), np.divide(Li_wall,1e20), marker="*", linestyle="--", color="k", label='Wall')
    plt.plot(np.divide(ncore_scan, 1e6), np.divide(Emit_Li_flux_Odiv,1e20), marker="o", linestyle="--", color="b", label='Odiv')
    plt.plot(np.divide(ncore_scan, 1e6), np.divide(Emit_Li_flux_Idiv,1e20), marker="o", linestyle="--", color="b", label='Idiv')
    plt.xlabel("Pcore (MW)",fontsize=14)
    plt.ylabel("$\phi_{Li}^{deposit} (1e20 atom/s)$", fontsize=14)
    plt.title("Power balance",fontsize=14)
    plt.legend(fontsize=12, ncol=2)
    plt.grid(True)
    #plt.ylim([0, 12])
    plt.tight_layout()
    plt.savefig('Li_flux.png', dpi=300)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(np.divide(ncore_scan, 1e6), np.divide(Emit_Li_flux_Odiv, 1e20), marker="o", linestyle="--", color="b", label='Odiv')
    ax1.plot(np.divide(ncore_scan, 1e6), np.divide(Emit_Li_flux_Idiv, 1e20), marker="s", linestyle="--", color="r", label='Idiv')
    ax1.set_ylabel("$\phi_{Li}^{emit} \\, (10^{20} \\, atoms/s)$", fontsize=12, color='b')
    ax1.set_xlabel("Pcore (MW)", fontsize=12)
    ax1.set_ylim([0, 20])
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(np.divide(ncore_scan, 1e6), np.divide(Li_Odiv, 1e20), marker="^", linestyle="--", color="g", label='Odiv')
    ax2.plot(np.divide(ncore_scan, 1e6), np.divide(Li_Idiv, 1e20), marker="*", linestyle="--", color="m", label='Idiv')
    ax2.plot(np.divide(ncore_scan, 1e6), np.divide(Li_wall, 1e20), marker="x", linestyle="--", color="k", label='Wall')
    ax2.set_ylabel("$\phi_{Li}^{deposit} \\, (10^{20} \\, atoms/s)$", fontsize=12, color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylim([0, 20])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=10, ncol=2)
    legend1 = ax1.legend(loc='lower left', fontsize=9, title='Emitted')
    legend2 = ax2.legend(loc='lower right', fontsize=9, title='Deposited')
    plt.title("Li Emitted and Deposited flux vs b0", fontsize=14)
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig('Li_flux_dual_axis.png', dpi=300)
    plt.show()

    # ?? Debug plot of q_sep only
    plt.figure(figsize=(5, 3))
    plt.plot(ncore_scan, q_sep, marker='*', color='black', label='q_sep')
    plt.title("DEBUG: q_sep vs b0")
    plt.xlabel("b0",fontsize=14)
    plt.ylabel("q_sep (MW)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save the results to CSV
    data_dict = {
        'b0': ncore_scan,  # Use ncore_scan, which is the list of b0 values
        'q_core': q_core,
        'q_sep': q_sep,
        'q_int_idiv': q_int_idiv,
        'q_int_odiv': q_int_odiv,
        'q_wall': q_wall,
        'prad_all': prad_all,
        'max_q': max_q,
        'max_q_idiv': max_q_idiv,
        'Te_max_odiv': Te_max_odiv,
        'Te_max_idiv': Te_max_idiv,
        'nemax_odiv': nemax_odiv,
        'C_Li_omp': C_Li_omp,
        'ne_sepm_omp': ne_sepm_omp
    }

    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame(data_dict)

    # Save DataFrame to CSV
    df.to_csv('power_balance_data.csv', index=False)

    print("Data has been saved to 'power_balance_data.csv'")





# Main execution
if __name__ == "__main__":
    data_path = "."
    start_ncore = 1e6
    end_ncore = 10e6

    ncore_list = []
    ncore = start_ncore
    while ncore <= end_ncore:
        ncore_list.append(ncore)
        if ncore < 5:
            ncore += 0.6e6
        elif ncore == 5:
            ncore += 0.6e6
        else:
            ncore += 0.6e6

    all_results = process_all_datasets(data_path, ncore_list)
    plot_results(all_results)
