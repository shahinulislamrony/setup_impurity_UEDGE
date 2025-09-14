import sys
import math
import os 
import numpy as np
from uedge.rundt import *
import matplotlib.pyplot as plt
from uedge import *
from uedge.hdf5 import *
from uedge.rundt import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.analysis as mana
import uedge_mvu.tstep as ut
import UEDGE_utils.analysis as ana
import pandas as pd
from runcase import *
from uedge.rundt import UeRun



setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=0.5, kyi=0.5, difni=0.35,nonuniform = False)
setBoundaryConditions(ncore = 7.78e19, pcoree=2.6e6, pcorei=2.6e6, recycp=0.95, owall_puff=0.0, dis = 1.75)
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



bbb.ftol=1e-5;bbb.dtreal=1e-10
bbb.issfon=1; bbb.isbcwdt=1
bbb.exmain()

print("Completed reading a converged solution for Li atoms and Li ions solution")



molar_mass_lithium = 6.94  # g/mol for lithium
avogadro_number = 6.022e23  # atoms/mol

def lithium_atoms_per_second(mass_in_grams):
    moles_of_lithium = mass_in_grams / molar_mass_lithium
    total_atoms = moles_of_lithium * avogadro_number
    return total_atoms

phi_Li_source_odiv = []
phi_Li_source_idiv = []
phi_Li_source_wall = []
phi_Li_source_pfr = []
Li_rad = []
phi_Li_odiv = []
phi_Li_wall =[]
phi_Li_idiv = []

pump_Li_odiv = []
pump_Li_wall = []
pump_Li_idiv = []
Li_ionization = []
Te_max_odiv = []
ne_max_odiv = []
zeff_OMP_sep = []
C_Li_omp_sep = []
ni_omp_sep = []
Te_omp_sep = []
ne_omp_sep = []
C_Li_all_sep_avg = []


Tsurf_max = []
phi_Li =[]
qmax = []
Error = 1
t_run = 5 
dt_each = 0.01
n = int (t_run / dt_each)
i = 0
it = []
previous_q_data = None  
Total_int_flux = 0.0

while i <= n:
    bbb.pradpltwl()
    bbb.plateflux()
    q_rad = bbb.pwr_pltz[:,1]+bbb.pwr_plth[:,1]
    q_data = bbb.sdrrb + bbb.sdtrb
    q_data = q_data.reshape(-1)
    print('size of the data is :', len(q_data))
    print(type(bbb.sdrrb), type(bbb.sdtrb))
    print(bbb.sdrrb.shape, bbb.sdtrb.shape)
    q_data = np.round(q_data, 2)
    np.save('q_data.npy', q_data)

    q_max2 = np.max(q_data)
    print('q_perp is done')
    print('Max q is :', q_max2)

    # Call the heat code
    print('---Calling heat code----')
    try:
        exec(open("heat_code.py").read())  
    except Exception as e:
        print(f"Error executing heat_code.py: {e}")
        break
    
    Tsurf = final_temperature
    T_max = np.max(Tsurf)
    print('Temp length is :', len(Tsurf))
    print('Peak temp is :', T_max)
    Tsurf_max.append(T_max)

    output_dir = "Tsurf_Li"
    os.makedirs(output_dir, exist_ok=True) 
    fname = os.path.join(output_dir, "T_surfit_" + "{:.1f}".format(i) + ".csv")
    print("Saving Tsurf in file: ", fname)
    np.savetxt(fname, Tsurf, delimiter=",")

    q_Li= q_Li_surf
    output_dir = "q_Li_surface"
    os.makedirs(output_dir, exist_ok=True) 
    fname = os.path.join(output_dir, "q_Li_surface_" + "{:.1f}".format(i) + ".csv")
    print("Saving Tsurf in file: ", fname)
    np.savetxt(fname, q_Li, delimiter=",")

    Gamma_net= Gamma_net
    output_dir = "Gamma_net"
    os.makedirs(output_dir, exist_ok=True) 
    fname = os.path.join(output_dir, "Gamma_Li_surface_" + "{:.1f}".format(i) + ".csv")
    print("Saving Tsurf in file: ", fname)
    np.savetxt(fname, Gamma_net, delimiter=",")

    
    output_dir = "Gamma_Li"
    os.makedirs(output_dir, exist_ok=True) 
    fname = os.path.join(output_dir, "Evap_flux_" + "{:.1f}".format(i) + ".csv")
    print("Saving Evap in file: ", fname)
    np.savetxt(fname, fluxEvap, delimiter=",")

    fname = os.path.join(output_dir, "PhysSput_flux_" + "{:.1f}".format(i) + ".csv")
    print("Saving Phys_Sput in file: ", fname)
    np.savetxt(fname, fluxPhysSput, delimiter=",")

    fname = os.path.join(output_dir, "Adstom_flux_" + "{:.1f}".format(i) + ".csv")
    print("Saving Ad-atom in file: ", fname)
    np.savetxt(fname, fluxAd, delimiter=",")

    fname = os.path.join(output_dir, "Total_Li_flux_" + "{:.1f}".format(i) + ".csv")
    print("Saving Li_flux in file: ", fname)
    np.savetxt(fname, tot, delimiter=",")

    
    print('----Heat code completed and output obtained')
    print('Length of the surface temperature is:', len(Tsurf))
    print('---Heat code is done----')
  

    print("Total Li flux is :", len(tot))
    print("phi_Li^Odiv :", np.sum(tot*com.sxnp[com.nx,:]))
    print('-----Li upper limit sets to 1e22---')
    Li_int_flux = np.sum(tot*com.sxnp[com.nx,:])
    phi_Li.append(Li_int_flux)

    Total_int_flux += Li_int_flux

    Li_upper = lithium_atoms_per_second(0.4)
    #if Total_int_flux > Li_upper:
    #    print(f'Li fluxes {Total_int_flux:.2f} reaches to upper limit {Li_upper:.2f}')
    #    break
    
    
    #bbb.fngxrb_use[:,1,0] = - tot*com.sxnp[com.nx,:]
    #print("Sum of fngxrb_use",np.sum(bbb.fngxrb_use[:,1,0]))
    #if Factor == 0:
    #    bbb.fphysyrb = "0.35"
    #else:
    #    bbb.fphysyrb = f"{0.35 + (Factor - 1):.4f}"
        
    #print('The factor is :', bbb.fphysyrb[0,0])

    bbb.fngxrb_use[:,1,0] = tot*com.sxnp[com.nx,:]
    print('sum of evaporation and add atom is :', np.sum(bbb.fngxrb_use))

    Phy_sput_Li = np.sum(bbb.sputflxrb[:,1,0])
    print('Li sput phys is :', Phy_sput_Li)
    
    print("Completed heat code, now running UEDGE code with the updated Li flux")
    
    bbb.restart = 1
    bbb.itermx = 10
    bbb.dtreal = 1e-10
    bbb.ftol = 1e-5
    bbb.issfon = 1
    bbb.isbcwdt=1
    bbb.exmain()
    
    
    print("******Check done*****")

    print("***** ut-step for 1 ms")

    #print(f"Tmax is {T_max}, so use the uestep")
    ut.uestep(10e-3, reset=True)
    savefile="run_last"+".hdf5"
    hdf5_save(savefile)
    print("Saving lasted out in file: ", savefile)

    """

    if T_max <= 380:
        print(f"Tmax is {T_max}, so use the uestep")
        ut.uestep(10e-3, reset=True)
        savefile="run_last"+".hdf5"
        hdf5_save(savefile)
        print("Saving lasted out in file: ", savefile)
        
    else:
        print(f"Tmax is {T_max}, so use rundt")
        bbb.t_stop = 10e-3
        bbb.dt_tot = 10e-3
        rundt(dtreal=10e-3, t_stop = 10e-3)
    """
    

    print("******done, now save data*****")
    

    if bbb.iterm == 1:
        np.save('final.npy', final_temperature) 
        bbb.pradpltwl()
        bbb.plateflux()
        q_rad = bbb.pwr_pltz[:,1]+bbb.pwr_plth[:,1]
        q_data = bbb.sdrrb + bbb.sdtrb
        q_data = q_data.reshape(-1)
        q_max = np.max(q_data)
        qmax.append(q_max)
        print("Max q_perp is :", q_max)

        output_dir = "q_perp"
        os.makedirs(output_dir, exist_ok=True) 
        fname = os.path.join(output_dir, "q_perpit_" + "{:.1f}".format(i) + ".csv")
        print("Saving q_perp in file: ", fname)
        np.savetxt(fname, q_data)

        #fname = os.path.join(output_dir, "q_surface_" + "{:.1f}".format(i) + ".csv")
        #print("Saving q_perp in file: ", fname)
        #np.savetxt(fname, q_surface)
        
     
        savefile="run_last"+".hdf5"
        hdf5_save(savefile)
        print("Saving lasted out in file: ", savefile)

        
        
        Te_max = np.max(bbb.te[com.nx,:]/bbb.ev)
        ne_max = np.max(bbb.ne[com.nx,:])
        Te_max_odiv.append(Te_max)
        ne_max_odiv.append(ne_max)
        zeff_omp = bbb.zeff[bbb.ixmp, com.iysptrx+1]
        zeff_OMP_sep.append(zeff_omp)
        n_Li = (bbb.ni[bbb.ixmp, com.iysptrx+1,2]+bbb.ni[bbb.ixmp, com.iysptrx+1,3]+bbb.ni[bbb.ixmp, com.iysptrx+1,4])
        C_Li_OMP = n_Li/bbb.ne[bbb.ixmp, com.iysptrx+1]
        C_Li_omp_sep.append(C_Li_OMP)
        ni_omp = bbb.ni[bbb.ixmp,com.iysptrx+1,0]
        ni_omp_sep.append(ni_omp)
        ne_omp = bbb.ne[bbb.ixmp, com.iysptrx+1]
        ne_omp_sep.append(ne_omp)
        Te_omp = bbb.te[bbb.ixmp,com.iysptrx+1]/bbb.ev
        Te_omp_sep.append(Te_omp)


        output_dir = "n_Li1"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir,"n_Li1_" + "{:.1f}".format(i) + ".csv")
        print("Saving in_Li+ in file: ", fname)
        np.savetxt(fname, bbb.ni[:,:,2])

        output_dir = "n_atom"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir,"n_0_" + "{:.1f}".format(i) + ".csv")
        print("Saving in_Li0 in file: ", fname)
        np.savetxt(fname, bbb.ng[:,:,1])

        output_dir = "Phi_D1_odiv"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir,"Phi_D1_" + "{:.1f}".format(i) + ".csv")
        print("Saving in D+ in file: ", fname)
        np.savetxt(fname, bbb.fnix[com.nx,:,0])


        output_dir = "Phi_Li1_odiv"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir,"Phi_Li1_" + "{:.1f}".format(i) + ".csv")
        print("Saving in_Li+ in file: ", fname)
        np.savetxt(fname, bbb.fnix[com.nx,:,2])

        output_dir = "Phi_Li2_odiv"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir,"Phi_Li2_" + "{:.1f}".format(i) + ".csv")
        print("Saving in_Li2+ in file: ", fname)
        np.savetxt(fname, bbb.fnix[com.nx,:,3])

        output_dir = "Phi_Li3_odiv"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir,"Phi_Li3_" + "{:.1f}".format(i) + ".csv")
        print("Saving in_Li3+ in file: ", fname)
        np.savetxt(fname, bbb.fnix[com.nx,:,4])

        output_dir = "Phi_Li3_wall"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir,"Phi_Li3_" + "{:.1f}".format(i) + ".csv")
        print("Saving in_Li3+ in file: ", fname)
        np.savetxt(fname, bbb.fniy[:,com.ny,4])

        output_dir = "T_e"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "T_e_" + "{:.1f}".format(i) + ".csv")
        print("Saving T_e in file: ", fname)
        np.savetxt(fname, bbb.te/bbb.ev)
        
        output_dir = "n_Li2"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "n_Li2_" + "{:.1f}".format(i) + ".csv")
        print("Saving n_Li2+ in file: ", fname)
        np.savetxt(fname, bbb.ni[:,:,3])

        output_dir = "n_Li3"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "n_Li3_" + "{:.1f}".format(i) + ".csv")
        print("Saving n_Li3+ in file: ", fname)
        np.savetxt(fname, bbb.ni[:,:,4])

        output_dir = "n_e"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "n_e_" + "{:.1f}".format(i) + ".csv")
        print("Saving n_e in file: ", fname)
        np.save(fname, bbb.ne)

        n_Li_all_sep = (bbb.ni[:, com.iysptrx,2]+bbb.ni[:, com.iysptrx,3]+bbb.ni[:, com.iysptrx,4])
        ne_all_sep = bbb.ne[:,com.iysptrx]
        C_Li_all_sep = (n_Li_all_sep/ne_all_sep)

        output_dir = "C_Li"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "C_Li_sep_all_" + "{:.1f}".format(i) + ".csv")
        print("Saving C_Li in file: ", fname)
        np.savetxt(fname,  C_Li_all_sep)
        
        C_Li_all_sep_avg.append(np.average(C_Li_all_sep))
        
        
        n_Li_rad = (bbb.ni[bbb.ixmp, :,2]+bbb.ni[bbb.ixmp, :,3]+bbb.ni[bbb.ixmp,:,4])
        C_Li_OMP_rad = n_Li/bbb.ne[bbb.ixmp, :]
        zeff_omp_rad = bbb.zeff[bbb.ixmp,:]

        output_dir = "C_Li_omp"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "CLi_prof" + "{:.1f}".format(i) + ".csv")
        print("Saving C_Li  in file: ", fname)
        np.savetxt(fname,C_Li_OMP_rad, delimiter=",")

        output_dir = "Z_eff_omp"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "Zeff_prof" + "{:.1f}".format(i) + ".csv")
        print("Saving Zeff in file: ", fname)
        np.savetxt(fname,zeff_omp_rad , delimiter=",")


        Li_rad2D = bbb.prad[:, :]

        output_dir = "Li_rad"
        os.makedirs(output_dir, exist_ok=True)  
        fname = os.path.join(output_dir, "Li_rad" + "{:.1f}".format(i) + ".csv")
        print("Saving Zeff in file: ", fname)
        np.savetxt(fname,Li_rad2D , delimiter=",")
        
        
        Li_source_Odiv = np.sum(bbb.sputflxrb)
        Li_source_idiv = np.sum(bbb.sputflxlb)
        Li_source_pfr = np.sum(bbb.sputflxpf)
        Li_source_Owall = np.sum(bbb.sputflxw)
        Li_rad_val = np.sum(bbb.prad[:, :] * com.vol)
        Phi_D_ion_odiv = np.sum(bbb.fnix[com.nx, :, 0])
        D_neutral = np.sum(bbb.fnix[com.nx, :, 1]) 
        Li_ion_Odiv = np.sum(bbb.fnix[com.nx, :, 2:5])
        Li_ion_Idiv = np.sum(np.abs(bbb.fnix[0,:,2:5]))
        Li_ion_Wall = np.sum(np.abs(bbb.fniy[:,com.ny,2:5]))
        Li_pump_Odiv = np.sum((1-bbb.recycp[1])*bbb.fnix[com.nx,:,2:5])
        Li_pump_Idiv = np.sum((1-bbb.recycp[1])*bbb.fnix[0,:,2:5])
        Li_pump_wall = np.sum((1-bbb.recycw[1])*bbb.fniy[:,com.ny,2:5])

        Li_ionization_val = np.sum(np.sum(np.abs(bbb.psor[:,:,2:5])))
        phi_Li_source_odiv.append(Li_source_Odiv)
        phi_Li_source_idiv.append(Li_source_idiv)
        phi_Li_source_pfr.append(Li_source_pfr)
        phi_Li_source_wall.append(Li_source_Owall)
        Li_rad.append(Li_rad_val)
        phi_Li_odiv.append(Li_ion_Odiv)
        phi_Li_wall.append(Li_ion_Wall)
        phi_Li_idiv.append(Li_ion_Idiv)
        pump_Li_odiv.append(Li_pump_Odiv)
        pump_Li_wall.append(Li_pump_wall)
        pump_Li_idiv.append(Li_pump_Idiv)
        Li_ionization.append(Li_ionization_val)
                    
        
    
   
    it.append(i)
    counter = i
    i += 1
    print("Iteration:", i)
    

qmax = np.array(qmax)
tsurf = np.array(Tsurf_max)
Phi_Li = np.array(phi_Li)

savefile="final_iteration"+".hdf5"
hdf5_save(savefile)

# Save results to files for plotting 
np.savetxt("qmax.csv", qmax, delimiter=",")
np.savetxt("Tsurf.csv", tsurf, delimiter=",")
np.savetxt("It.csv", it, delimiter=",")
np.savetxt("Phi_Li.csv", Phi_Li, delimiter = ",")

print('The factor is :', bbb.fphysyrb[0,0])


data = {
    "phi_Li_source_odiv": phi_Li_source_odiv,
    "phi_Li_source_idiv": phi_Li_source_idiv,
    "phi_Li_source_pfr": phi_Li_source_pfr,
    "phi_Li_source_wall": phi_Li_source_wall,
    "Li_rad": Li_rad,
    "phi_Li_odiv": phi_Li_odiv,
    "phi_Li_wall": phi_Li_wall,
    "phi_Li_idiv": phi_Li_idiv,
    "pump_Li_odiv": pump_Li_odiv,
    "pump_Li_wall": pump_Li_wall,
    "pump_Li_idiv": pump_Li_idiv,
    "Li_ionization": Li_ionization,
    "Te_max_odiv"  : Te_max_odiv,
    "ne_max_odiv"  : ne_max_odiv,
    "zeff_omp_sep" : zeff_OMP_sep,
    "C_Li_omp_sep" : C_Li_omp_sep,
     "ni_omp_sep" :  ni_omp_sep,
     "Te_omp_sep" :  Te_omp_sep,
     "ne_omp_sep" :  ne_omp_sep,
     "C_Li_sep_all" :  C_Li_all_sep_avg,
}


df = pd.DataFrame(data)
csv_filename = "Li_all.csv"  
df.to_csv(csv_filename, index=False)
print(f"Data successfully saved to {csv_filename}")


plt.figure()
plt.plot(qmax/1e6, tsurf, '--r', marker='*', markersize=14)
plt.xlabel('q$_{\perp, max}^{odiv}$ (MW/m$^2$)', fontsize=20)
plt.ylabel('T$_{surf}^{max}$ ($^\circ$C)', fontsize=20)
plt.title('qmax vs Tmax per iteration')
#plt.legend()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.grid()
#plt.ylim([0,2])
#plt.xlim([0 ,6])
plt.tight_layout()
plt.savefig('q_max_Temp_it.png', dpi=300)
plt.show()




#num_cases = counter
#q_data = []
#for i in range(1, num_cases + 1):
#    fname = f"q_perpit_{i}.csv"
#    print(f"Loading data from file: {fname}")
#    q_data_array = np.loadtxt(fname)
#    q_data.append(q_data_array)

plt.figure(figsize=(12, 8))

for i, data in enumerate(q_data):
    plt.plot(com.yyrb, q_data, label=f'Case {i + 1}')

plt.xlabel('com.yyrb')
plt.ylabel('q Value')
plt.title('Data for Each Case vs. com.rrtb')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


