import sys
import numpy as np
import matplotlib.pyplot as plt
from uedge import *
from uedge.hdf5 import *
from uedge.rundt import *



def setGrid():

    bbb.gengrid=0
    bbb.mhdgeo = 1          #use MHD equilibrium
    com.geometry = "snull"
    com.nxpt=1 #-how many X-points in the domain
    isnonog=1
    methg=66

    bbb.gallot("Xpoint_indices",0)
    grd.readgrid("gridue",com.runid)

    com.nx=com.nxm
    com.ny=com.nym
    com.isnonog = 1

    # Finite-difference algorithms (upwind, central diff, etc.)
    bbb.methn = 33          #ion continuty eqn
    bbb.methu = 33          #ion parallel momentum eqn
    bbb.methe = 33          #electron energy eqn
    bbb.methi = 33          #ion energy eqn
    bbb.methg = 66          #neutral gas continuity eqn


def setPhysics(impFrac=0.0, b0=1.0, fluxLimit=False):    
    bbb.isteon=1
    bbb.istion=1
    bbb.isnion=1
    bbb.isnion[2:]=0
    bbb.isupon=1
    bbb.isupgon=0
    bbb.isngon=0
    bbb.kxe = 1.0	     #elec thermal conduc scale factor;now default
    bbb.lmfplim = 1.e3	     #elec thermal conduc reduc 1/(1+mfp/lmfplim)
    bbb.cion=3

    # Inertial neutrals
    com.nhsp = 2
    bbb.ziin[1] = 0.
    bbb.isngon[0] = 0
    bbb.isupgon[0] = 1


    # Fixed Impurities
    if (impFrac>1e-8):
        bbb.isimpon = 2
        bbb.afracs = impFrac
    else:
        bbb.isimpon = 0

        
   
    bbb.b0=b0
        

    if (fluxLimit):
        bbb.flalfe = 1.0 # electron parallel thermal conduct. coeff
        bbb.flalfi = 1.0 # ion parallel thermal conduct. coeff
        bbb.flalfv = 0.5  # ion parallel viscosity coeff
        bbb.flalfgx = 1.0  # neut. gas in poloidal direction
        bbb.flalfgy = 1.0 # neut. gas in radial direction
        bbb.flalftgx = 1.0 # neut power in poloidal direction
        bbb.flalftgy = 1.0 # neut power in radial direction
        bbb.lgmax = 2e-1  # max scale for gas particle diffusion
        bbb.lgtmax = 2e-1 # max scale for gas thermal diffusion
        bbb.flalftgx = 1.0		#limit x thermal transport
        bbb.flalftgy = 1.0		#limit y thermal transport
        bbb.flalfvgx = 1.0		#limit x neut visc transport
        bbb.flalfvgy = 1.0		#limit y neut visc transport
        bbb.flalfvgxy = 1.0		#limit x-y nonorthog neut visc transport
        bbb.isplflxlv = 1  #=0, flalfv not active at ix=0 & nx;=1 active all ix
        bbb.isplflxlgx = 1 #=0, flalfgx not active at ix=0 & nx;=1 active all ix
        bbb.isplflxlgxy = 1 #=0, flalfgxy not active at ix=0 & nx;=1 active all ixv
        bbb.isplflxlvgx = 1 #=0, flalfvgx not active at ix=0 & nx;=1 active all ix
        bbb.isplflxlvgxy = 1 #=0, flalfvgxy not active at ix=0 & nx;=1 active all ix
        bbb.iswflxlvgy = 1 #=0, flalfvgy not active at iy=0 & ny;=1 active all iy
        bbb.isplflxltgx = 1 #=0, flalfvgx not active at ix=0 & nx;=1 active all ix
        bbb.isplflxltgxy = 1 #=0, flalfvgxy not active at ix=0 & nx;=1 active all ix
        bbb.iswflxltgy = 1 #=0, flalfvgy not active at iy=0 & ny;=1 active all iy


        
    
    #-set flat initial profiles                                                                                 
    bbb.allocate()                                                                                             
    bbb.tes=10*bbb.ev                                
    bbb.tis=10*bbb.ev                                                                                          
    bbb.nis=7e19                                                                                               
    bbb.ngs=7e19                                                                                               
    bbb.ups=0
    
    
def setBoundaryConditions(ncore=6e19, pcoree=2.5e6, pcorei=2.5e6, recycp=0.98, recycw=1.0,owall_puff=0.0, pfr_puff=0.0, dis = 1.75):

    bbb.isnicore[0] = 1     #=3 gives uniform density and I=curcore
    bbb.ncore[0] = ncore     #hydrogen ion density on core
    bbb.isybdryog = 1       #=1 uses orthog diff at iy=0 and iy=ny bndry

    bbb.isnwcono = 3                #=3 for (1/n)dn/dy = 1/lyni
    bbb.nwomin[0] = 1.e13 # 1.e14 # 1.e12 # 
    bbb.nwimin[0] = 1.e13 # 1.e14 # 1.e12 #
    bbb.lyni[1] = 0.02              #iy=ny+1 density radial scale length (m)
    bbb.lyni[0] = 0.02             #iy=0 density radial scale length

    #bbb.isnwconi = 3                # switch for private-flux wall
    

    bbb.iflcore = 1         #flag; =0, fixed Te,i; =1, fixed power on core
    bbb.tcoree = 200.       #core Te if iflcore=0
    bbb.tcorei = 200.       #core Ti if iflcore=0
    bbb.pcoree = pcoree # .625e6      #core elec power if iflcore=1
    bbb.pcorei = pcorei # .625e6      #core ion  power if iflcore=1

    
    #bbb.isextrtw=0
    #bbb.isextrtpf=0
    bbb.istewc = 3          # =3 ditto for Te on vessel wall
    bbb.istiwc = 3          # =3 ditto for Ti on vessel wall
    bbb.lyte = 0.008 # 0.02  # scale length for Te bc
    bbb.lyti = 0.008 # 0.02  # scale length for Ti bc
   

    bbb.isupcore = 1          #=1 sets d(up)/dy=0
    bbb.isupss= 0
    bbb.isupwo = 2          # =2 sets d(up)/dy=0
    bbb.isupwi = 2          # =2 sets d(up)/dy=0
    bbb.isplflxl = 0 # 1    #=0 for no flux limit (te & ti) at plate
    bbb.isngcore[0]=0       #set neutral density gradient at core bndry
    bbb.recycm = -0.9		# mom BC at plates:up(,,2) = -recycm*up(,,1)

    bbb.matwso[0] = 1               # =1 --> make the outer wall recycling.
    bbb.matwsi[0] = 1               # =1 --> make the inner wall recycling.
    
    bbb.recycp[0] = recycp    # outer and inner wall recycling, R_N
    bbb.recycw[0] = recycw    # recycling coeff. at wall
    
    bbb.isrecmon = 1
    bbb.kxe = 1.0		#elec thermal conduc scale factor;now default
    bbb.lmfplim = 1.e3		#elec thermal conduc reduc 1/(1+mfp/lmfplim)

    # set plate albedoes
    bbb.albedolb[0,0]=0.99
    bbb.albedorb[0,0]=0.99

    # sheath parameters:
    bbb.bcei=2.5
    bbb.bcee=4.0


    # Neutral gas properties
    com.ngsp=1
    bbb.ineudif=2		# pressure driven neutral diffusion
    bbb.ngbackg = 1.e13 # 1.e15 # 1.e12 #  # background gas level (1/m**3)
    bbb.isupgon[0]=1
    bbb.isngon[0]=0
    com.nhsp=2
    bbb.ziin[com.nhsp-1]=0
    bbb.gcfacgx = 1.            # sets plate convective gas flux
    bbb.gcfacgy = 1.            # sets wall convective gas flux


    #Divertor gas puff
    bbb.nwsor = 2  	            # number of wall sources
    bbb.matwso[0] = 1               # =1 --> make the outer wall recycling.
    bbb.matwsi[0] = 1               # =1 --> make the inner wall recycling.
    bbb.albdsi = 1.0
    bbb.albdso = 1.0
    bbb.recycw = 1.0
    
   #-outer wall
    bbb.igaso[1]= owall_puff  #-total puff strength [A]
    bbb.xgaso[1] = dis  #-source center poloidal location [m]
    bbb.wgaso[1] = 0.1 #-source width [m]
    bbb.issorlb[1] = 0  #-measure poloidal distance from inner plate
    bbb.igaso = 0.0
#-inner wall
    #bbb.igasi[1]= 0e-1  #-total pump strength [A]
    #bbb.xgasi[1] = 0.3  #-source center poloidal location [m]     
    #bbb.wgasi[1] = 0.3  #-source width [m]
    #bbb.issorlb[1] = 0  #-measure poloidal distance from outer plate

     # plate boundary conditions
    bbb.isplflxl=0  # turn off flux limits at plate

def setimpmodel(impmodel=False, sput_factor=1.0):

    if (impmodel):
        bbb.isimpon = 6
        bbb.isofric = 1		#Use general bbb.friction force expression
        bbb.cfparcur = 1.	#scale fac for bbb.fqp=parcurrent from fmombal
        com.ngsp = 2
        com.nzsp[0]=3
	#bbb.nusp_imp = 3
        bbb.isngon[1] = 1		
        bbb.isnion[2:6]= 1   
        bbb.n0g[1] = 1.e17
        bbb.isupgon[1] = 0		
        bbb.recycp[1:6] = 1.e-10		
        bbb.recycw[1:6] = 1.e-10	       
        bbb.ngbackg[1]=1.e10 	  
        bbb.recycw[2:6]=1.e-10


        bbb.allocate()				
        bbb.minu[com.nhsp:com.nhsp+3] = 7.    
        bbb.ziin[com.nhsp:com.nhsp+3] = array([1, 2, 3])    
        bbb.znuclin[0:com.nhsp] = 1	       
        bbb.znuclin[com.nhsp:com.nhsp+3] = 3
	#bbb.nusp_imp = 3
        bbb.nzbackg=1.e10		      
        bbb.n0[com.nhsp:com.nhsp+3]=1.e17 
        bbb.inzb=2		      
        bbb.isbohmms=0		      
        bbb.isnicore[com.nhsp:com.nhsp+3] = 1 
        bbb.ncore[1:5] = 1e12  # Li core density BC
        
        bbb.recycc[1:5]=1.e-10	        
        #bbb.curcore[com.nhsp:com.nhsp+3] = 0.0
        bbb.isnwcono[com.nhsp:com.nhsp+3] = 3 
        bbb.isnwconi[com.nhsp:com.nhsp+3] = 3
        bbb.nwimin[com.nhsp:com.nhsp+3] = 1.e7
        bbb.nwomin[com.nhsp:com.nhsp+3] = 1.e7
        bbb.rcxighg=0.0			# force charge exchange small 
        bbb.kelighi[1] = 5.e-16       # sigma-v for elastic collisions
        bbb.kelighg[1] = 5.e-16
        com.iscxfit=2			# use reduced Maggi CX fits
        bbb.kye = 0.5		#chi_e for radial elec energy diffusion
        bbb.kyi = 0.5		#chi_i for radial ion energy diffusion
        bbb.difni[2:5] =  0.5
        bbb.difni2[2:5] = 0.5
        bbb.travis[2:5] = 1.0		#eta_a for radial ion momentum diffusion
        bbb.parvis[2:7] = 1.0	       

        


        bbb.ismctab = 2		# =1 use multi charge tables from INEL
  				# =2 Use multi charge tables from Strahl
        com.nzdf=1
        #com.mcfilename = ["Li_rates.adas"]
        bbb.ismctab = 2       # use Braams' api.rate tables
        com.mcfilename = "b2frates_Li_v4_mod3"

        #com.isrtndep=1
	#bbb.cion=3
        bbb.isch_sput[1]=1	# Haasz/Davis sputtering model
        bbb.isph_sput[1]=1	# Haasz/Davis sputtering model
        bbb.t_wall=300
        bbb.t_plat=300
        bbb.crmb=bbb.minu[0]	# set mass of bombarding particles
        fhaasz=sput_factor
        bbb.fchemylb=1.e-5
        bbb.fchemyrb=1.e-5
        bbb.fphysylb=1.0
        bbb.fphysyrb=sput_factor
        bbb.fchemygwi=1.e-5
        bbb.fchemygwo=1.e-5
        bbb.isybdryog=1


        #bbb.nwsor = 2
        bbb.igspsoro[bbb.nwsor-1] = 2
        bbb.igspsori[bbb.nwsor-1] = 2
        bbb.albdso[bbb.nwsor-1] = 0.99
        bbb.albdsi[bbb.nwsor-1] = 0.99


def setDChi(kye=0.25, kyi=0.25, difni=0.25, nonuniform=False, kye_sol = 0.15):
    
    if (nonuniform == False):
        print("Setting uniform transport coefficients")
        bbb.kye=kye
        bbb.kyi=kyi
        bbb.difni=difni
        bbb.dif_use=0.0
        bbb.kye_use=0.0
        bbb.kyi_use=0.0
        bbb.vy_use=0.0
        
    else:
        print("NON-uniform transport coefficients")
        bbb.isbohmcalc = 0
        
        #bbb.facbee=1.0
        #bbb.facbei=1.0
        #bbb.facbni=1.0

        bbb.dif_use[0:com.nx+2,0:com.ny+2] = bbb.difni[0] # base 0,   (0:nx+1,0:ny+1,1:nisp)
        bbb.kye_use[0:com.nx+2,0:com.ny+2] = bbb.kye # base 0,   (0:nx+1,0:ny+1)
        bbb.kyi_use[0:com.nx+2,0:com.ny+2] = bbb.kyi # base 0,   (0:nx+1,0:ny+1)

        bbb.difni[0]=0.0
        bbb.kye=0.0
        bbb.kyi=0.0


        bbb.nphygeo()

#Dn 
        profparam=5e-3
        Dn_core = 0.33
        Dn_SOL_max = 0.33
        rprof = np.zeros(com.ny+2)
        rprof[0:com.iysptrx] = Dn_core
        L_sol = 4e-3  # SOL decay length
        rprof[com.iysptrx : com.ny + 2] =0.33# Dn_SOL_max * np.exp(-(com.yyc[com.iysptrx : com.ny + 2] - com.yyc[com.iysptrx]) / L_sol)
        rprof[com.iysptrx - 1 : com.iysptrx + 2] = Dn_SOL_max
        
        
#Ion energy, chi_i
        profparam3=5.0e-3
        rprof3 = np.zeros(com.ny+2)
        rprof3[com.iysptrx+2:com.ny+2] = kye_sol * np.exp((com.yyc[com.iysptrx+2:com.ny+2] - com.yyc[com.iysptrx]) / profparam3) 
        rprof3[0:com.iysptrx] = 1.5
        rprof3[com.iysptrx-1:com.iysptrx+2] = kye_sol 

#-clip it to a constant value
#real 

        dmax=0.5
        dmax2=6
        dmax3=2.5
        for iii in range(0,com.ny+2):
            rprof[iii]=min(rprof[iii],dmax)
            rprof3[iii]=min(rprof3[iii],dmax3)

        for iii in range(0,com.nx+2):
            bbb.dif_use[iii,0:com.ny+2,0]=rprof # base 0,   (0:nx+1,0:ny+1,1:nisp)
      
            
            bbb.kye_use[iii,0:com.ny+2]= rprof3 # use same for e and i rprof2
            bbb.kyi_use[iii,0:com.ny+2]= rprof3

      
