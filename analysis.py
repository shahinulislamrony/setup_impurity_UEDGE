'''
General methods to analyze a case after it's finished running.
'''
import itertools
import scipy.optimize
from scipy.special import erfc, erfcx
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import shapely.geometry
from uedge import bbb, com, flx, grd, svr, aph, api
import Forthon
from lmfit import Model, Parameters
from scipy.signal import medfilt
from IPython import get_ipython
from IPython.core.magic import register_line_magic
ipython = get_ipython()


def getPkg(var: str) -> str:
    """Get the UEDGE package that the given variable resides in."""
    for pkg in ['bbb', 'com', 'flx', 'grd', 'svr', 'aph', 'api']:
        if var in globals()[pkg].varlist():
            return pkg
    return None
        

def getObject(var: str):
    """Get the object with the given name (e.g. 'ni') from the right UEDGE package"""
    obj = None
    pkg = getPkg(var)
    if pkg:
        pkg = globals()[pkg]
        try:
            obj = pkg.getpyobject(var)
        except Exception as e:
            print(e)
    return obj
    
    
def objectDescription(obj) -> str:
    """Return a string describing the given UEDGE object's value"""
    if type(obj) == np.ndarray:
        if 'S' in str(obj.dtype):
            description = obj[0].decode('utf-8')
        else:
            if obj.size < 50:
                with np.printoptions(precision=4, linewidth=60):
                    description = str(obj)
            else:
                description = 'mean %7.2g, min %7.2g, max %7.2g' % (np.mean(obj), np.min(obj), np.max(obj))
    else:
        description = str(obj)
    return description


if ipython:
    @register_line_magic
    def uhelp(var: str):
        """UEDGE documentation/getting/setting utility
        
        Usage:
        u isnicore -> show documentation and value of isnicore
        u isnicore[0] -> show value of isnicore[0]
        u isnicore[0] 1 -> set isnicore[0] to 1
        """
        if var == '':
            help(uhelp)
            return
        elif ('[' in var) or (' ' in var and var.split(' ')[1] != ''):
            # If input is isnicore[0] 1, then set isnicore[0]
            # If input is isnicore[0], show value of isnicore[0]
            varm(var)
        else:
            # If input is isnicore, show documentation and value of isnicore
            var = var.strip() # remove any stray spaces around it
            Forthon.doc(var)
            stats(var)
    # Create a shorter alias for %docm: ud
    ipython.run_line_magic('alias_magic', 'u uhelp')


def varm(varAndVal: str):
    """Get/set UEDGE variable in iPython e.g. %varm dif_use 1 sets bbb.dif_use = 1"""
    val = None
    slc = ''
    vs = varAndVal.split(' ')
    var = vs[0]
    if '[' in var:
        var, slc = var.split('[')
        slc = '[' + slc
    if len(vs) >= 2:
        val = ' '.join(vs[1:])
    obj = getObject(var)
    if obj is not None:
        pkg = getPkg(var)
        if val is not None:
            # If input is isnicore[0] 1, then set isnicore[0]
            exec('%s.%s%s = %s' % (pkg, var, slc, val))
            stats(var)
        else:
            # If input is isnicore[0], show value of isnicore[0]
            with np.printoptions(precision=4, linewidth=60):
                # exec("print('%s.%s%s:', str(obj%s.shape), '\\n', obj%s)" % (pkg, var, slc, slc, slc))
                exec("print('%s.%s%s: %%s\\n%%s' %% (str(obj%s.shape), obj%s))" % (pkg, var, slc, slc, slc))
    else:
        print('Invalid UEDGE variable')


def stats(var: str):
    """Get statistics on UEDGE variable in iPython e.g. %stats dif_use"""
    # Show current value
    obj = getObject(var)
    if obj is not None:
        pkg = getPkg(var)
        d = objectDescription(obj)
        if hasattr(obj, 'shape'):
            print('%s.%s: %s' % (pkg, var, str(obj.shape)))
            if len(obj.shape) == 3:
                # print('[:,:,:]: %s' % d)
                for i in range(obj.shape[2]):
                    print('[:,:,%d]: %s' % (i, objectDescription(obj[:,:,i])))
            else:
                print(d)
        else:
            print('%s.%s: %s' % (pkg, var, d))
    else:
        print('Invalid UEDGE variable')
        
        
def Tsep():
    tisep = (bbb.ti[bbb.ixmp,com.iysptrx]+bbb.ti[bbb.ixmp,com.iysptrx+1])/2/bbb.ev
    tesep = (bbb.te[bbb.ixmp,com.iysptrx]+bbb.te[bbb.ixmp,com.iysptrx+1])/2/bbb.ev
    return 'Tisep = %.3g eV, Tesep = %.3g eV' % (tisep, tesep)
        
        
def iximp():
    """Return inner midplane poloidal index"""
    if com.isudsym: 
        # up-down symmetric double null: jump in poloidal index at midplane
        # bbb.ixmp is outer midplane guard cell right around magnetic axis
        return bbb.ixmp-1
    else: 
        # tested only in lower single null configuration
        # bbb.ixmp is first cell above magnetic axis
        # return first (in ix) cell with center above east face of outer midplane cell
        for ix in range(bbb.ixmp):
            if com.isixcore[ix] and com.zm[ix,0,0] > com.zm[bbb.ixmp,0,2]:
                return ix


def ixilast():
    """Return index of last (in ix) inboard cell"""
    if com.isudsym:
        # up-down symmetric double null
        return bbb.ixmp-1
    else:
        # double null full domain
        if com.ixrb[0]+1 < com.nx:
            return com.ixrb[0]+1
        else:
            # lower single null (ixrb[0]=nx)
            for ix in range(bbb.ixmp):
                # cell before first where east point Z less than west point
                if com.isixcore[ix] and com.zm[ix,0,2] < com.zm[ix,0,1]:
                    return ix-1
    

def neighbor(ix, iy, direction):
    """Return physically neighboring cell in given direction. Works at X-points and in all configurations.
    
    Args:
        ix: (int)
        iy: (int)
        direction: (str) N/E/S/W
    
    Returns:
        (tuple) (nix, niy) of neighbor or None if (ix, iy) is at a boundary
    """
    if direction == 'E':
        if com.isudsym and ix == bbb.ixmp-1:
            return (None, None)
        if ix-1 in com.ixrb:
            return (None, None)
        return (bbb.ixp1[ix, iy], iy)
    elif direction == 'W':
        if com.isudsym and ix == bbb.ixmp:
            return (None, None)
        if ix in com.ixlb:
            return (None, None)
        return (bbb.ixm1[ix, iy], iy)
    elif direction == 'N':
        if iy == com.ny+1:
            return (None, None)
        return (ix, bbb.iyp1a[ix, iy])
    elif direction == 'S':
        if iy == 0:
            return (None, None)
        return (ix, bbb.iym1a[ix, iy])


def cellFaceVertices(face, ix, iy):
    '''
    Return [[v1r, v1z], [v2r, v2z]] for the N/S/E/W corners of cell ix, iy.
    '''
    v1v2 = {'N': (3, 4),
            'S': (1, 2),
            'E': (2, 4),
            'W': (1, 3)}
    v1, v2 = v1v2[face]
    return [[com.rm[ix, iy, v1], com.zm[ix, iy, v1]], 
            [com.rm[ix, iy, v2], com.zm[ix, iy, v2]]]
    

def nonGuard(var):
    '''
    Return values in non-guard cells as a flat array.
    '''
    v = []
    guardX = [0, com.nx+1]
    guardY = [0, com.ny+1]
    if 'dnbot' in com.geometry[0].decode('utf-8'):
        guardX.extend([bbb.ixmp-1, bbb.ixmp])
    for ix in range(com.nx+2):
        for iy in range(com.ny+2):
            if not (ix in guardX or iy in guardY):
                v.append(var[ix, iy])
    return np.array(v)
    
    
def toGrid(function):
    '''
    Compute for all grid cells any function that takes arguments ix, iy. 
    '''
    out = np.zeros((com.nx+2, com.ny+2))
    for ix in range(com.nx+2):
        for iy in range(com.ny+2):
            out[ix, iy] = function(ix, iy)
    return out    


def badCells():
    '''
    Return all (ix, iy) of cells which are not valid polygons, e.g. twisted, all vertices 0, etc.
    '''
    corners = (1,3,4,2,1)
    bad = []
    for ix in range(com.nx+2):
        for iy in range(com.ny+2):
            rcenter = com.rm[ix, iy, 0]
            zcenter = com.zm[ix, iy, 0]
            p = shapely.geometry.Polygon(zip(com.rm[ix,iy,corners]-rcenter, com.zm[ix,iy,corners]-zcenter))
            if not p.is_valid:
                bad.append((ix, iy))
    return bad
    
    
def overlappingCells():
    """Return list of (ix, iy) of valid cells with overlap."""
    corners = (1,3,4,2,1)
    polys = []
    for ix in range(com.nx+2):
        arr = []
        for iy in range(com.ny+2):
            arr.append(shapely.geometry.Polygon(zip(com.rm[ix,iy,corners], com.zm[ix,iy,corners])))
        polys.append(arr)
    overlappingCells = set()
    for ix in range(com.nx+1):
        for iy in range(com.ny+1):
            c = polys[ix][iy]
            if not c.is_valid:
                continue
            eix, eiy = neighbor(ix, iy, 'E')
            if eix != None:
                e = polys[eix][eiy]
                if e.is_valid and c.overlaps(e):
                    overlapFraction = c.intersection(e).area/(c.area+e.area)
                    if overlapFraction > 1e-9:
                        overlappingCells.add((ix, iy))
                        continue
            nix, niy = neighbor(ix, iy, 'N')
            if nix != None:
                n = polys[nix][niy]
                if n.is_valid and c.overlaps(n):
                    overlapFraction = c.intersection(n).area/(c.area+n.area)
                    if overlapFraction > 1e-9:
                        overlappingCells.add((ix, iy))
    return list(overlappingCells)
    

def powerLost():
    '''
    Return total power lost to outer boundaries and volumetric sinks.
    '''
    pwrx = bbb.feex+bbb.feix
    pwry = bbb.feey+bbb.feiy
    pbindx = bbb.fnix[:,:,0]*bbb.ebind*bbb.ev
    pbindy = bbb.fniy[:,:,0]*bbb.ebind*bbb.ev
    prad = np.sum(bbb.erliz+bbb.erlrc)
    if bbb.isimpon != 0:
        irad = np.sum(bbb.prad*com.vol)
    else:
        irad = 0
    pInnerTarget = np.sum((-pwrx-pbindx)[0,:])
    pOuterTarget = np.sum((pwrx+pbindx)[com.nx,:])
    pCFWall = np.sum((pwry+pbindy)[:,com.ny])
    pPFWallInner = np.sum((-pwry-pbindy)[:com.ixpt1[0]+1,0])
    pPFWallOuter = np.sum((-pwry-pbindy)[com.ixpt2[0]+1:,0])
    return pInnerTarget + pOuterTarget + pCFWall + pPFWallInner + pPFWallOuter + prad + irad 
    
    
def powerLostBreakdown():
    '''
    Return powers lost to outer boundaries and volumetric sinks.
    '''
    pwrx = bbb.feex+bbb.feix
    pwry = bbb.feey+bbb.feiy
    pbindx = bbb.fnix[:,:,0]*bbb.ebind*bbb.ev
    pbindy = bbb.fniy[:,:,0]*bbb.ebind*bbb.ev
    prad = np.sum(bbb.erliz+bbb.erlrc)
    if bbb.isimpon != 0:
        irad = np.sum(bbb.prad*com.vol)
    else:
        irad = 0
    pke = PionParallelKE()
    pInnerTarget = np.sum((-pwrx-pbindx-pke)[0,:])
    pOuterTarget = np.sum((pwrx+pbindx+pke)[com.nx,:])
    pCFWall = np.sum((pwry+pbindy)[:,com.ny])
    pPFWallInner = np.sum((-pwry-pbindy)[:com.ixpt1[0]+1,0])
    pPFWallOuter = np.sum((-pwry-pbindy)[com.ixpt2[0]+1:,0])
    return pInnerTarget, pOuterTarget, pCFWall, pPFWallInner, pPFWallOuter, prad, irad 


def gridPowerBalance():
    '''
    Sum of power sources and sinks for each cell in grid.
    '''
    return toGrid(lambda ix, iy: cellPowerBalance(ix, iy, verbose=False))
    
    
def gridPowerSumAbs():
    '''
    Sum of absolute values of power sources and sinks for each cell in grid. Useful to assess 
    relative importance of gridPowerBalance() results.
    '''
    sumAbs = np.abs(bbb.erliz) + np.abs(bbb.erlrc)
    if bbb.isimpon != 0:
        sumAbs += np.abs(bbb.prad*com.vol)
    sumAbs += np.abs(gridSourcePoloidal(bbb.feex+bbb.feix))
    sumAbs += np.abs(gridSourceRadial(bbb.feey+bbb.feiy))
    return sumAbs
    
    
def cellPowerBalance(ix, iy, verbose=True):
    '''
    Sum of power sources and sinks in specified cell.
    '''
    hrad = -bbb.erliz[ix,iy] - bbb.erlrc[ix,iy]
    if bbb.isimpon == 0:
        irad = 0
    else:
        irad = -bbb.prad[ix,iy]*com.vol[ix,iy]
    pflux = cellSourcePoloidal(bbb.feex+bbb.feix, ix, iy)
    rflux = cellSourceRadial(bbb.feey+bbb.feiy, ix, iy)
    net = pflux + rflux + hrad + irad
    if verbose:
        mismatch = abs(net)/gridPowerSumAbs()[ix,iy]
        print('%9.3g W H radiation loss due to ionization, recombination (-erliz-erlrc)' % hrad)
        print('%9.3g W impurity radiation loss (-prad*vol)' % irad)
        print('%9.3g W net poloidal flux (from feex+feix)' % pflux)
        print('%9.3g W net radial flux (from feey+feiy)' % rflux)
        print('%9.3g W total' % net)
        print('%9.3g relative mismatch abs(sum(components))/sum(abs(components))' % mismatch)
    return net
    
    
def gridParticleBalance(species=0):
    '''
    Sum of density sources and sinks for each cell in grid.
    '''
    return toGrid(lambda ix, iy: cellParticleBalance(ix, iy, species=species, verbose=False))
    
    
def gridParticleSumAbs(species=0):
    '''
    Sum of absolute values of density sources and sinks for each cell in grid. Useful to assess 
    relative importance of gridParticleBalance() results.
    '''
    sumAbs = np.abs(bbb.psor[:,:,species])
    sumAbs += np.abs(gridSourcePoloidal(bbb.fnix[:,:,species]))
    sumAbs += np.abs(gridSourceRadial(bbb.fniy[:,:,species]))
    return sumAbs
    
    
def cellParticleBalance(ix, iy, species=0, verbose=True):
    '''
    Sum of density sources and sinks in specified cell.
    '''
    ionizsrc = bbb.psor[ix,iy,species]
    recmsrc = -bbb.psorrg[ix,iy,species]
    pflux = cellSourcePoloidal(bbb.fnix[:,:,species], ix, iy)
    rflux = cellSourceRadial(bbb.fniy[:,:,species], ix, iy)
    net = pflux + rflux + ionizsrc + recmsrc
    if verbose:
        mismatch = abs(net)/(abs(ionizsrc)+abs(recmsrc)+abs(pflux)+abs(rflux))
        print('%9.3g particles/s ionization (psor)' % ionizsrc)
        print('%9.3g particles/s recombination (psorrg)' % ionizsrc)
        print('%9.3g particles/s net poloidal flux (from fnix)' % pflux)
        print('%9.3g particles/s net radial flux (from fniy)' % rflux)
        print('%9.3g particles/s total' % net)
        print('%9.3g relative mismatch abs(sum(components))/sum(abs(components))' % mismatch)
    return net


def gridSourcePoloidal(var):
    '''
    Poloidal flow in minus flow out for the given flux-type variable, over the whole grid.
    '''
    return toGrid(lambda ix, iy: cellSourcePoloidal(var, ix, iy))
    
    
def gridSourceRadial(var):
    '''
    Radial flow in minus flow out for the given flux-type variable, over the whole grid.
    '''
    return toGrid(lambda ix, iy: cellSourceRadial(var, ix, iy))
    
    
def cellSourcePoloidal(var, ix, iy):
    '''
    Poloidal flow in minus flow out for the given flux-type variable at cell ix, iy. Includes
    special considerations for any geometry X-point and midplane.
    '''
    nix, niy = neighbor(ix, iy, 'W')
    if nix != None:
        ns = var[nix, niy]
    else:
        ns = 0
    return ns-var[ix,iy]


def cellSourceRadial(var, ix, iy):
    '''
    Radial flow in minus flow out for the given flux-type variable at cell ix, iy.
    '''
    if iy == 0:
        return -var[ix,iy]
    return var[ix,iy-1]-var[ix,iy]


def distancesRadialLayers(y1, y2):
    '''
    Return array of distances between cell centers at y=y1 and y=y2.
    '''
    return np.sqrt((com.rm[:,y1,0]-com.rm[:,y2,0])**2+(com.zm[:,y1,0]-com.zm[:,y2,0])**2)


def innerBoundaryIfGradLength(var, gl=.01): 
    '''
    Ideal values at inner boundary for given variable and gradient length.
    '''
    var = var.copy()
    dx = distancesRadialLayers(0, 1)
    return var[:,1]*(2*gl/dx-1)/(1+2*gl/dx)

    
def outerBoundaryIfGradLength(var, gl=.01):
    '''
    Ideal values at outer boundary for given variable and gradient length.
    '''
    var = var.copy()
    dx = distancesRadialLayers(com.ny, com.ny+1)
    return var[:,com.ny]*(2*gl/dx-1)/(1+2*gl/dx)


def gradLengthsInner(var):
    '''
    Return gradient lengths at the inner boundary for given var.
    '''
    dx = distancesRadialLayers(0, 1)
    return (var[:,1]+var[:,0])/2.*dx/(var[:,1]-var[:,0])


def gradLengthsOuter(var):
    '''
    Return gradient lengths at the outer boundary for given var.
    '''
    dx = distancesRadialLayers(com.ny, com.ny+1)
    return (var[:,com.ny]+var[:,com.ny+1])/2.*dx/(var[:,com.ny]-var[:,com.ny+1])  
    
    
def powerLeg(leg='outer'):
    pwrx = bbb.feex+bbb.feix
    pwry = bbb.feey+bbb.feiy
    nx = com.nx
    ny = com.ny
    ebind = bbb.ebind
    ev = bbb.ev
    if leg == 'outer':
        plateIndex = 1
        xsign = 1 # poloidal fluxes are measured on east face of cell
        ixpt = com.ixpt2[0]
        xlegEntrance = ixpt+1
        xtarget = nx
        xleg = slice(ixpt+1, xtarget+1)
        print('--- OUTER DIVERTOR LEG POWER STATS ---')
    elif leg == 'inner':
        plateIndex = 0
        xsign = -1 # poloidal fluxes are measured on east face of cell
        ixpt = com.ixpt1[0]
        xlegEntrance = ixpt-1
        xtarget = 0
        xleg = slice(xtarget+1, ixpt+1)
        print('--- INNER DIVERTOR LEG POWER STATS ---')
    ysol = slice(com.iysptrx+1, ny+1)

    #-calculate photon fluxes on material surfaces
    bbb.pradpltwl()
    
    convCondLeg = xsign*sum(pwrx[xlegEntrance,:])/1e3
    print('Convection and conduction power entering leg: %.4g kW' % convCondLeg)
    recombLeg = xsign*sum((bbb.fnix[xlegEntrance,:,0])*ebind*ev)/1e3
    print('Recombination power entering leg: %.4g kW\n' % recombLeg)

    convCondTarget = xsign*sum(pwrx[xtarget,ysol])/1e3
    print('Convection and conduction power to target: %.4g kW' % convCondTarget)
    recombTarget = xsign*sum((bbb.fnix[xtarget,ysol,0])*ebind*ev)/1e3
    print('Recombination power to target: %.4g kW' % recombTarget)
    HPhotTarget = sum(bbb.pwr_plth[:,plateIndex]*com.sxnp[xtarget,:])/1e3
    print('Hydrogen photon power to target: %.4g kW' % HPhotTarget)
    impPhotTarget = sum(bbb.pwr_pltz[:,plateIndex]*com.sxnp[xtarget,:])/1e3
    print('Impurity photon power to target: %.4g kW\n' % impPhotTarget)

    HRadLeg = sum((bbb.erliz+bbb.erlrc)[xleg,:])/1e3
    print('Hydrogen radiated power lost in leg: %.4g kW' % HRadLeg)
    if (bbb.isimpon > 0):
        impRad = sum((bbb.prad*com.vol)[xleg,:])/1e3
    else:
        impRad = 0.0
    print('Impurity radiated power lost in leg: %.4g kW\n' % impRad)
    convCondLegCF = sum(pwry[xleg,ny])/1e3
    print('Convection and conduction power to leg common flux wall: %.4g kW' % convCondLegCF)
    recombLegCF = sum((bbb.fniy[xleg,ny,0])*ebind*ev)/1e3
    print('Recombination power to leg common flux wall: %.4g kW' % recombLegCF)
    HPhotLegCF = sum(bbb.pwr_wallh[xleg]*com.sy[xleg,ny])/1e3
    print('Hydrogen photon power to leg common flux wall: %.4g kW' % HPhotLegCF)
    impPhotLegCF = sum(bbb.pwr_wallz[xleg]*com.sy[xleg,ny])/1e3
    print('Impurity photon power to leg common flux wall: %.4g kW\n' % impPhotLegCF)

    convCondLegPF = -sum(pwry[xleg,0])/1e3
    print('Convection and conduction power to leg private flux wall: %.4g kW' % convCondLegPF)
    recombLegPF = -sum((bbb.fniy[xleg,0,0])*ebind*ev)/1e3
    print('Recombination power to leg private flux wall: %.4g kW' % recombLegPF)
    HPhotLegPF = sum(bbb.pwr_pfwallh[xleg,0]*com.sy[xleg,1])/1e3 # note pwr_pfwallh has dim (nx,1)
    print('Hydrogen photon power to leg private flux wall: %.4g kW' % HPhotLegPF)
    impPhotLegPF = sum(bbb.pwr_pfwallz[xleg,0]*com.sy[xleg,1])/1e3 # # note pwr_pfwallz has dim (nx,1)
    print('Impurity photon power to leg private flux wall: %.4g kW' % impPhotLegPF)


def fit_lamda_q_inner_outer(com,bbb):
    iy = com.iysptrx+1 # first point outside LCFS
    ixo = com.nx#com.ixpt2[0] # LFS cell west of X-point
    ixi = com.ixpt1[0] # HFS cell west of X-point
    xq = com.yyc[iy:]  # centers of cells mapped to OMP
    qparo = bbb.fetx[ixo,iy:]/com.sx[ixo,iy:]/com.rr[ixo+1,iy:]
    qpari = -bbb.fetx[ixi,iy:]/com.sx[ixi,iy:]/com.rr[ixi,iy:]

    expfun = lambda x, A, lamda_q_inv: A*np.exp(-x*lamda_q_inv) # fitting function in a form that curve_fit likes
    omax = np.argmax(qparo) # only fit stuff to right of max
    imax = np.argmax(qpari) 
    qofit, _ = scipy.optimize.curve_fit(expfun, xq[omax:], qparo[omax:])
    qifit, _ = scipy.optimize.curve_fit(expfun, xq[imax:], qpari[imax:])
    return 1./qofit[1], 1./qifit[1]
    
    
def getrrf():
    bpol_local = 0.5*(com.bpol[:,:,2] + com.bpol[:,:,4])
    bphi_local = 0.5*(com.bphi[:,:,2] + com.bphi[:,:,4])
    btot_local = np.sqrt(bpol_local**2+bphi_local**2)
    return bpol_local/btot_local
    
    
def qpar():
    '''Parallel projection of feex+feix eventually minus drifts'''
    pass
   
    
def qparXptInner():
    pass
    

def qparXptOuter():
    pass
    
    
def qsurfparInner():
    '''Total power to surface projected in parallel direction'''
    rrf = getrrf()
    psurf = PsurfInner()
    ix = 0
    return psurf/com.sx[ix,:]/getrrf()[ix,:]
    
    
def qsurfparOuter():
    '''Total power to surface projected in parallel direction'''    
    rrf = getrrf()
    psurf = PsurfOuter()
    ix = com.nx
    return psurf/com.sx[ix,:]/rrf[ix,:]


def powerDnbot():
    # For convenience in translating from BASIS to python
    nx = com.nx
    ny = com.ny
    ixmp = bbb.ixmp
    fniy = bbb.fniy
    fnix = bbb.fnix
    erliz = bbb.erliz
    erlrc = bbb.erlrc
    prad = bbb.prad
    vol = com.vol
    ev = bbb.ev
    ixpt1 = com.ixpt1[0]
    ixpt2 = com.ixpt2[0]
    iysptrx = com.iysptrx
    xoleg = np.s_[ixpt2+1:nx+1]  # x indices of divertor outer leg cells
    xileg = np.s_[1:ixpt1+1]     # x indices of divertor inner leg cells
    ysol = np.s_[iysptrx+1:ny+1] # y indices outside LCFS

    pwrx = bbb.feex + bbb.feix # includes neutrals
    pwry = bbb.feey + bbb.feiy # includes neutrals
    allsum = 0.0

    # Power to inner wall above X-point 
    powerFromCore = (sum(pwry[ixpt1+1:ixmp,0])+sum(pwry[ixmp:ixpt2+1,0]))/1e3
    print("Power leaving core: %.4g kW" % powerFromCore)

    # Power to inner wall above X-point
    s = np.s_[ixpt1:ixmp]
    powerToInnerWall = sum(pwry[s,ny])/1e3
    print("Power to inner wall: %.4g kW" % powerToInnerWall)
    powerRecombToInnerWall = sum(fniy[s,ny,0]*13.6*ev)/1e3
    print("Recomb. power to inner wall: %.4g kW" % powerRecombToInnerWall)
    allsum = allsum + powerToInnerWall + powerRecombToInnerWall

    # Power to outer wall above X-point 
    s = np.s_[ixmp:ixpt2+2]
    powerToOuterWall = sum(pwry[s,ny])/1e3
    print("Power to outer wall: %.4g kW" % powerToOuterWall)
    powerRecombToOuterWall = sum(fniy[s,ny,0]*13.6*ev)/1e3
    print("Recomb. power to outer wall: %.4g kW" % powerRecombToOuterWall)
    allsum = allsum + powerToOuterWall + powerRecombToOuterWall

    # Power into outer leg
    powerEnteringOuterLeg = sum(pwrx[ixpt2,ysol])/1e3
    print("Power entering outer leg: %.4g kW" % powerEnteringOuterLeg)
    powerRecombEnteringOuterLeg = sum(fnix[ixpt2,ysol,0]*13.6*ev)/1e3
    print("Recomb. power entering outer leg: %.4g kW" % powerRecombEnteringOuterLeg)
    allsum = allsum + powerEnteringOuterLeg + powerRecombEnteringOuterLeg

    # Power into inner leg
    powerEnteringInnerLeg = sum(pwrx[ixpt1,ysol])/1e3
    print("Power entering inner leg: %.4g kW" % powerEnteringInnerLeg)
    powerRecombEnteringInnerLeg = sum(fnix[ixpt1,ysol,0]*13.6*ev)/1e3
    print("Recomb. power entering inner leg: %.4g kW" % powerRecombEnteringInnerLeg)
    allsum = allsum - powerEnteringInnerLeg - powerRecombEnteringInnerLeg

    # Power radiated above X-point
    sx = np.s_[ixpt1+1:ixpt2]
    sy = np.s_[0:ny+1]
    if bbb.isimpon > 0: 
        impRadAboveXpt = sum((prad*vol)[sx,sy])/1e3
        print("Impurity radiated power above X-point: %.4g kW" % impRadAboveXpt)
        allsum = allsum + impRadAboveXpt
    else:
        print("Impurity radiated power above X-point: 0.0 kW")
    HRadAboveXpt = sum((erliz+erlrc)[sx,sy])/1e3
    print("H radiated power above X-point: %.4g kW" % HRadAboveXpt)
    allsum = allsum + HRadAboveXpt

    # Wrap up
    print("Core power setting: %.4g kW" % ((bbb.pcoree+bbb.pcorei)/1e3))
    print("Sum of above losses: %.4g kW" % allsum)

    # H power radiated in divertor legs
    HRadInnerLeg = sum((erliz+erlrc)[xileg,sy])/1e3
    print("H radiated power inner leg: %.4g kW" % HRadInnerLeg)
    HRadOuterLeg = sum((erliz+erlrc)[xoleg,sy])/1e3
    print("H radiated power outer leg: %.4g kW" % HRadOuterLeg)

    bbb.pradpltwl()

    print('Max H radiation power on inner target: %.4g kW/m^2' % (max(bbb.pwr_plth[:,0])/1e3))
    print('Max H radiation power on outer target: %.4g kW/m^2' % (max(bbb.pwr_plth[:,1])/1e3))
    print('Max H radiation power on private flux wall: %.4g kW/m^2' % (max(bbb.pwr_pfwallh)/1e3))
    print('Max H radiation power on outer wall: %.4g kW/m^2' % (max(bbb.pwr_wallh)/1e3))
    print('Max impurity radiation power on inner target: %.4g kW/m^2' % (max(bbb.pwr_pltz[:,0])/1e3))
    print('Max impurity radiation power on outer target: %.4g kW/m^2' % (max(bbb.pwr_pltz[:,1])/1e3))
    print('Max impurity radiation power on private flux wall: %.4g kW/m^2' % (max(bbb.pwr_pfwallz)/1e3))
    print('Max impurity radiation power on outer wall: %.4g kW/m^2' % (max(bbb.pwr_wallz)/1e3))
    
    
def olegParticleBalance():
    print('%6.2g /s into leg from CF poloidally' % np.sum(bbb.fnix[com.ixpt2[0]+1,com.iysptrx+1:-1,0]))
    print('%6.2g /s into leg from PF poloidally' % np.sum(bbb.fnix[com.ixpt2[0]+1,1:com.iysptrx+1,0]))
    print('%6.2g /s into leg from CF radially' % -np.sum(bbb.fniy[com.ixpt2[0]+2:-1,com.ny,0]))
    print('%6.2g /s into leg from PF radially' % np.sum(bbb.fniy[com.ixpt2[0]+2:-1,0,0]))
    print('%6.2g /s onto target poloidally' % np.sum(bbb.fnix[com.nx,1:-1,0]))
    print('%6.2g /s from ionization in volume' % np.sum(bbb.psor[com.ixpt2[0]+2:-1,1:-1,0]))
    print('%6.2g /s from recombination in volume' % -np.sum(bbb.psorrg[com.ixpt2[0]+2:-1,1:-1,0]))
    print('%6.2g /s sum' % (np.sum(bbb.fnix[com.ixpt2[0]+1,com.iysptrx+1:-1,0])+np.sum(bbb.fnix[com.ixpt2[0]+1,1:com.iysptrx+1,0])+-np.sum(bbb.fniy[com.ixpt2[0]+2:-1,com.ny,0])+np.sum(bbb.fniy[com.ixpt2[0]+2:-1,0,0])+-np.sum(bbb.fnix[com.nx,1:-1,0])+np.sum(bbb.psor[com.ixpt2[0]+2:-1,1:-1,0])+-np.sum(bbb.psorrg[com.ixpt2[0]+2:-1,1:-1,0])))
    

def fieldLineAngle():
    """(nx+2, ny+2) array of angle in degrees between field line and cell East surface. Uses B field at center of cell---could be slightly improved by interpolating to cell East face."""
    # dr, dz is the surface tangent
    # -dz, dr is the surface normal (pointing west)
    # bpol = sqrt(br^2+bz^2) (br is in machine R dir., not plasma r)
    # a.b = |a||b|cos(theta), theta = angle between field line and surface normal
    # angle between surface tangent and field line is 90-theta
    # arcsindeg(x) = 90-arccosdeg(x)
    dr = com.rm[:,:,2]-com.rm[:,:,4]
    dz = com.zm[:,:,2]-com.zm[:,:,4]
    return np.abs(np.arcsin(-com.br[:,:,0]*dz+com.bz[:,:,0]*dr)/((dr**2+dz**2)**0.5*(com.br[:,:,0]**2+com.bz[:,:,0]**2+com.bphi[:,:,0]**2)**0.5))*180/np.pi
    
    
def PionParallelKE():
    """(nx+2, ny+2) array of power [Watts] in East direction of ion parallel kinetic energy, usually only significant at plates."""
    return 0.5*bbb.mi[0]*bbb.up[:,:,0]**2*bbb.fnix[:,:,0]
    
    
def Pparallel():
    """(nx+2, ny+2) array of parallel power [Watts] in East direction including...
    - feex: electron thermal current
    - feix: ion thermal current
    - ion parallel kinetic energy
    """
    return bbb.feex+bbb.feix+PionParallelKE()


def PsurfInner():
    """(ny+2) array of total power [Watts] in West direction along inner divertor surface"""
    bbb.pradpltwl()
    plateIndex = 0
    xsign = -1 # poloidal fluxes are measured on east face of cell
    bbb.fetx = bbb.feex+bbb.feix
    psurfi = xsign*bbb.fetx[0,:] \
             +xsign*bbb.fnix[0,:,0]*bbb.ebind*bbb.ev \
             +xsign*PionParallelKE()[0,:] \
             +bbb.pwr_plth[:,plateIndex]*com.sxnp[0,:] \
             +bbb.pwr_pltz[:,plateIndex]*com.sxnp[0,:]
    return psurfi
    
    
def PsurfOuter():
    """(ny+2) array of total power [Watts] in East direction along outer divertor surface"""
    bbb.pradpltwl()
    plateIndex = 1
    xsign = 1
    bbb.fetx = bbb.feex+bbb.feix
    psurfo = xsign*bbb.fetx[com.nx,:] \
             +xsign*bbb.fnix[com.nx,:,0]*bbb.ebind*bbb.ev \
             +xsign*PionParallelKE()[com.nx,:] \
             +bbb.pwr_plth[:,plateIndex]*com.sxnp[com.nx,:] \
             +bbb.pwr_pltz[:,plateIndex]*com.sxnp[com.nx,:]
    return psurfo


def platePeakVals(iplate=0, show=False):
    #-find peak values of plasma parameters on the plate

    #-inner plate
    ix=0
    peak_ip = {
        "ni":np.max(bbb.ni[ix,:,0]), #-[m-3]
        "ng":np.max(bbb.ni[ix,:,1]), #-[m-3]
        "te":np.max(bbb.te[ix,:]/bbb.ev), #-[eV]
        "ti":np.max(bbb.ti[ix,:]/bbb.ev), #-[eV]
        "qsurf":np.max(PsurfInner()/com.sxnp[ix,:]) #-[W/m^2]
    }


    #-outer plate
    ix=com.nx
    peak_op = {
        "ni":np.max(bbb.ni[ix,:,0]), #-[m-3]
        "ng":np.max(bbb.ni[ix,:,1]), #-[m-3]
        "te":np.max(bbb.te[ix,:]/bbb.ev), #-[eV]
        "ti":np.max(bbb.ti[ix,:]/bbb.ev), #-[eV]
        "qsurf":np.max(PsurfOuter()/com.sxnp[ix,:]) #-[W/m^2]
    }


    if (iplate==0):
        peak_plate=peak_ip
        title="Inner plate:"
    elif (iplate==1):
        peak_plate=peak_op
        title="Outer plate:"

    if (show):
        print(title)
        print("max(ni)=", '{:6.2e}'.format(peak_plate.get("ni")), " [m-3]")
        print("max(ng)=", '{:6.2e}'.format(peak_plate.get("ng")), " [m-3]")
        print("max(Te)=", '{:6.2e}'.format(peak_plate.get("te")), " [eV]")
        print("max(Ti)=", '{:6.2e}'.format(peak_plate.get("ti")), " [eV]")
        print("max(qsurf)=", '{:6.2e}'.format(peak_plate.get("qsurf")/1e6), "MW/m^2")
        
    return peak_plate
    
    
def dPara():
    '''Parallel distance, invalid in PFR
    From https://github.com/LLNL/UEDGE/blob/master/pyscripts/contrib/ue_plot.py#L1308'''
    scale = np.ones((com.nx+2,com.ny+2))
    scale = (com.bphi[:,:,0]**2+com.bpol[:,:,0]**2)**0.5

    # Calculate the distance in the poloidal direction
    x = np.ones((com.nx+2,com.ny+2))
    for i in range(com.ny+2):
        x[:,i] = np.cumsum(scale[:,i]/com.gxf[:,i])
    x[-1,:] = x[-2,:] # Fix guard cell
    # x = x-x[bbb.ixmp,:] # Normalize to OMP
    
    return x


def eichFitOuter(lqoGuess=1):
    # lqoGuess is in mm
    psurfo = PsurfOuter()
    qsurfo = psurfo[1:-1]/com.sxnp[com.nx+1,1:-1]
    intqo = np.sum(psurfo)
    def qEich(rho, q0, S, lq, qbg, rho_0):
        rho = rho - rho_0
        # This can cause overflow warning when exp is large and erfc is small
        return q0/2*np.exp((S/2/lq)**2-rho/lq)*erfc(S/2/lq-rho/S)+(qbg*1e-2)
        # This can cause 'log encountered divide by zero' error and doesn't give better fits
        # return q0/2*np.exp((S/2/lq)**2-rho/lq+np.log(erfc(S/2/lq-rho/S)))+qbg
    bounds = ([0,1e-9,1e-9,0,com.yyc[0]], [np.inf,np.inf,np.inf,np.inf,com.yyc[-1]])
    oguess = (np.max(qsurfo)-np.min(qsurfo[qsurfo > 0]), lqoGuess/1000/2, lqoGuess/1000, np.min(qsurfo[qsurfo > 0]), 0)
    try:
        qsofit, cov = scipy.optimize.curve_fit(qEich, com.yyc[1:-1], qsurfo, p0=oguess, bounds=bounds)
        lqeo, So = qsofit[2], qsofit[1] # lamda_q and S in m
        return lqeo, So, np.sqrt(cov[2,2]), np.sqrt(cov[1,1]) # Lq, S, Lq stand. err., S stand. err.
    except Exception as e:
        print('qsurf outer fit failed:', e)
        return None, None, None, None
        
        
def qBrunnerPW(rho, q0, qcf, lcn, lcf, lpn, r0):
    r = rho-r0
    return np.piecewise(r, [r < 0, r >= 0], [lambda rho: q0*np.exp(rho/lpn), lambda rho: (q0-qcf)*np.exp(-rho/lcn)+qcf*np.exp(-rho/lcf)])
    
    
def qBrunnerS(rho, q0, qcf, lcn, lcf, lpn, S, r0):
    r = rho-r0
    return 1/2*np.exp(-(r**2/S**2))*(qcf*erfcx(S/(2*lcf) - r/S) + (q0 - qcf)*erfcx(S/(2*lcn) - r/S) + q0*erfcx(S/(2*lpn) + r/S))
    
    
def qBrunnerBoth(rho, q0, qcf, lcn, lcf, lpn, S, r0):
    b = qBrunnerS(rho, q0, qcf, lcn, lcf, lpn, S, r0)
    pw = qBrunnerPW(rho, q0, qcf, lcn, lcf, lpn, r0)
    b[~np.isfinite(b)] = pw[~np.isfinite(b)]
    return b
        
        
def brunnerFitOuter(guess=None):
    psurfo = PsurfOuter()
    qsurfo = psurfo[1:-1]/com.sxnp[com.nx+1,1:-1]
    intqo = np.sum(psurfo)
    if not guess:
        # Averaged the fits from the df
        guess = [6.2e8,3.6e7,0.0012,2.8,0.00047,-0.00041]
        guess[0] = np.max(qsurfo)*2
        guess[1] = guess[0]/10
    lb = 1e-20
    ub = 1e20
    lbl = 1e-9
    ubl = 10
    bounds = ([lb,lb,lbl,lbl,lbl,com.yyc[0]], [ub,ub,ubl,ubl,ubl,com.yyc[-1]])
    guess[-1] = com.yyc[1:-1][np.argmax(qsurfo)]
    f = interp1d(com.yyc[1:-1], qsurfo, kind='linear', fill_value='extrapolate')
    irhos = np.linspace(com.yyc[1],com.yyc[-2],100)
    iqsurfo = f(irhos)
    try:
        #qsofit, cov = scipy.optimize.curve_fit(qBrunnerPW, com.yyc[1:-1][qsurfo>0], qsurfo[qsurfo>0], p0=guess, bounds=bounds, maxfev=100000)
        qsofit, cov = scipy.optimize.curve_fit(qBrunnerPW, irhos[iqsurfo>0], iqsurfo[iqsurfo>0], p0=guess, bounds=bounds, maxfev=100000)
        q0, qcf, lcn, lcf, lpn, r0 = qsofit[0], qsofit[1], qsofit[2], qsofit[3], qsofit[4], qsofit[5] # lamda_qs in m
        q0err, qcferr, lcnerr, lcferr, lpnerr, r0err = np.sqrt(cov[0,0]), np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[3,3]), np.sqrt(cov[4,4]), np.sqrt(cov[5,5]) # standard errors in m
        return q0, qcf, lcn, lcf, lpn, r0, q0err, qcferr, lcnerr, lcferr, lpnerr, r0err
    except Exception as e:
        print('qsurf outer fit failed:', e)
        return [None for i in range(12)]
        
        
def brunnerSFitOuter(guess=None):
    psurfo = PsurfOuter()
    qsurfo = psurfo[1:-1]/com.sxnp[com.nx+1,1:-1]
    intqo = np.sum(psurfo)
    if not guess:
        # Averaged the fits from the df
        guess = [6.2e8,3.6e7,0.0012,2.8,0.00047,0.044,-0.00041]
        guess[0] = np.max(qsurfo)*2
        guess[1] = guess[0]/10
    lb = 1e-20
    ub = 1e20
    lbl = 1e-12
    ubl = 10
    bounds = ([lb,lb,lbl,lbl,lbl,lbl,com.yyc[0]], [ub,ub,ubl,ubl,ubl,ubl,com.yyc[-1]])
    guess[-1] = com.yyc[1:-1][np.argmax(qsurfo)]
    f = interp1d(com.yyc[1:-1], qsurfo, kind='linear', fill_value='extrapolate')
    irhos = np.linspace(com.yyc[1],com.yyc[-2],100)
    iqsurfo = f(irhos)
    try:
        #qsofit, cov = scipy.optimize.curve_fit(qBrunnerBoth, com.yyc[1:-1][qsurfo>0], qsurfo[qsurfo>0], p0=guess, bounds=bounds, maxfev=1000000)
        qsofit, cov = scipy.optimize.curve_fit(qBrunnerBoth, irhos[iqsurfo>0], iqsurfo[iqsurfo>0], p0=guess, bounds=bounds, maxfev=1000000)
        q0, qcf, lcn, lcf, lpn, S, r0 = qsofit[0], qsofit[1], qsofit[2], qsofit[3], qsofit[4], qsofit[5], qsofit[6] # lamda_qs and S in m
        q0err, qcferr, lcnerr, lcferr, lpnerr, Serr, r0err = np.sqrt(cov[0,0]), np.sqrt(cov[1,1]), np.sqrt(cov[2,2]), np.sqrt(cov[3,3]), np.sqrt(cov[4,4]), np.sqrt(cov[5,5]), np.sqrt(cov[6,6]) # standard errors in m
        return q0, qcf, lcn, lcf, lpn, S, r0, q0err, qcferr, lcnerr, lcferr, lpnerr, Serr, r0err
    except Exception as e:
        print('qsurf outer fit failed:', e)
        return [None for i in range(14)]
        
        
def quadLqProfile(rho,q0,q1,l0,l1,l2,S,r0):
    r = rho-r0
    model = (0.5*(np.exp((0.5*S/l0)**2.0-r/l0)*q0*erfc(0.5*S/l0-r/S)
                  +np.exp((0.5*S/l1)**2.0-r/l1)*q1*erfc(0.5*S/l1-r/S)
                  +np.exp((0.5*S/l2)**2.0+r/l2)*(q0+q1)*erfc(0.5*S/l2+r/S)))

    try:
        nans,idx = np.isnan(model), lambda z: z.nonzero()[0]
        model[nans] = np.interp(idx(nans),idx(~nans),model[~nans])
    except Exception as e:
        return model
    return np.log10(model)
    

def quadLqProfileNonlog(rho,q0,q1,l0,l1,l2,S,r0):
    r = rho-r0
    model = (0.5*(np.exp((0.5*S/l0)**2.0-r/l0)*q0*erfc(0.5*S/l0-r/S)
                  +np.exp((0.5*S/l1)**2.0-r/l1)*q1*erfc(0.5*S/l1-r/S)
                  +np.exp((0.5*S/l2)**2.0+r/l2)*(q0+q1)*erfc(0.5*S/l2+r/S)))

    try:
        nans,idx = np.isnan(model), lambda z: z.nonzero()[0]
        model[nans] = np.interp(idx(nans),idx(~nans),model[~nans])
    except Exception as e:
        return model
    return model
    
    
def monotonicHF(x, y):
    '''Return part of the heat flux profile where values are decreasing going out from peak'''
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    ypi = np.argmax(y)
    xout = [x[ypi]]
    yout = [y[ypi]]
    for yi in range(ypi-1,-1,-1):
        if y[yi] < y[yi+1]:
            xout.append(x[yi])
            yout.append(y[yi])
        else:
            break
    for yi in range(ypi+1,len(y)):
        if y[yi] < y[yi-1]:
            xout.append(x[yi])
            yout.append(y[yi])
        else:
            break
    yout = np.array(yout)
    xout = np.array(xout)
    yout = yout[np.argsort(xout)]
    xout = xout[np.argsort(xout)]
    return xout, yout
        
        
def brunnerFitOriginal(guess=None, method='least_squares', par=True, debug=False, idiv=False, log=True, interp=True, monotonic=True, fixedS=None):    
    rrf = getrrf()
    if idiv:
        ppar = PsurfInner()
        ixo = 0
        q = ppar[1:-1]/com.sx[ixo,1:-1]/rrf[ixo,1:-1]
    elif par and not idiv:
        ppar = PsurfOuter()
        ixo = com.nx
        q = ppar[1:-1]/com.sx[ixo,1:-1]/rrf[ixo,1:-1]
    else:
        psurfo = PsurfOuter()
        q = psurfo[1:-1]/com.sxnp[com.nx+1,1:-1]
    if monotonic:
        x, y = monotonicHF(com.yyc[1:-1][q>0]*1000, q[q>0]/1e6)
    if interp:
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        x = np.linspace(x[0],x[-1],500)
        y = np.log10(f(x))
    else:
        x = com.yyc[1:-1][q>0]*1000
        y = np.log10(q[q>0]/1e6)
    # Note Dan is fitting qpara/1e6, yyc in mm
    
    if guess:
        # MW/m^2 and mm 
        q0, q1, l0, l1, l2, S, r0 = guess
        r0 = x[np.argmax(y)]
    else:
        q0=10**2
        q1=10**1
        l0=1.
        l1=4.
        l2=0.5
        S =0.5 # 0.5 guess in Brunner but ended up clustering at 0.01
        r0=0.
    params = Parameters()
    params.add('da', value=q1/q0, min=0e+0, max=1e-1)
    params.add('dc', value=l1/l0, min=0e+0, max=1e+0)
    params.add('dd', value=S/l2,  min=0e+0, max=1e+1)
    params.add('q0', value=q0,    min=1e+0, max=1e+4)
    params.add('q1', value=q1,    min=1e-1, max=1e+4, expr='da*q0')
    params.add('l0', value=l0,    min=1e-2, max=1e+2)
    params.add('l1', value=l1,    min=1e-2, max=1e+4, expr='l0/dc')
    params.add('l2', value=l2,    min=1e-2, max=1e+2)
    if fixedS is not None:
        params.add('S',  value=fixedS ,    min=fixedS-1e-6, max=fixedS+1e-6, expr='dd*l2', vary=False)
    else:
        params.add('S',  value=S ,    min=1e-2, max=1e+1, expr='dd*l2')
    params.add('r0', value=r0,    min=-5,   max=5)

    if log:
        # The way Dan did it, plus weights which really help
        model = Model(quadLqProfile,missing='none')
        result = model.fit(y, params, rho=x, nan_policy='propagate', method=method,weights=10**y/np.max(10**y))
    else:
        # Sucks including for drifts
        model = Model(quadLqProfileNonlog,missing='none')
        result = model.fit(10**y, params, rho=x, nan_policy='propagate', method=method)
    p = result.params.valuesdict()
    if debug:
        print(result.fit_report())
    return [
    result.params['q0'].value, 
    result.params['q1'].value, 
    result.params['l0'].value, 
    result.params['l1'].value, 
    result.params['l2'].value, 
    result.params['S'].value, 
    result.params['r0'].value, 
    result.params['q0'].stderr, 
    result.params['q1'].stderr, 
    result.params['l0'].stderr, 
    result.params['l1'].stderr, 
    result.params['l2'].stderr, 
    result.params['S'].stderr, 
    result.params['r0'].stderr]


def impStats():
    txt = 'Average %.2g%% (' % (impFraction()*100)
    vmin = nonGuard(bbb.ni[:,:,2:]).min()
    vmax = nonGuard(bbb.ni[:,:,2:]).max()
    totImps = np.sum([np.sum(nonGuard(bbb.ni[:,:,i]*com.vol)) for i in range(2,bbb.ni.shape[2])])
    numImps = bbb.ni.shape[2]-2
    for i in range(numImps):
        txt += '+%d %.2g%%' % (i+1, np.sum(nonGuard(bbb.ni[:,:,i+2]*com.vol))/totImps*100)
        if i != numImps-1:
            txt += ', '
    txt += '), min %.2g, max %.2g' % (vmin, vmax)
    return txt


def impFraction():
    '''Charged impurity count divided by hydrogen ion count.'''
    return np.sum([np.sum(nonGuard(bbb.ni[:,:,i]*com.vol)) for i in range(2,bbb.ni.shape[2])])/np.sum(nonGuard(bbb.ni[:,:,0]*com.vol))
    

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

def eich_shahinul():
    yyc = com.yyc.reshape(-1)
    bbb.plateflux()
    q_data = (bbb.sdrrb + bbb.sdtrb).reshape(-1)
    yyrb = com.yyrb.reshape(-1)
    q_surf = q_data

    def eichFunction(x, S, lq, q0, s0):
        fx = (com.bpol[bbb.ixmp, com.iysptrx, 0] * com.rm[bbb.ixmp, com.iysptrx, 0]) / \
             (com.bpol[com.nx, com.iysptrx, 0] * com.rm[com.nx, com.iysptrx, 0])
        fx = np.round(fx)
        sBar = x - s0
        t0 = (S / (2 * lq * fx)) ** 2
        t1 = sBar / (lq * fx)
        t2 = S / (2 * lq * fx)
        t3 = sBar / S
        q_back = q0 * 1e-3 
        return (q0 / 2) * np.exp(t0 - t1) * erfc(t2 - t3) + q_back

    def linearExtrapolate(xarray, yarray, x1, x2, n=10):
        newxArray, newyArray = [], []
        for i in range(len(xarray) - 1):
            if x1 <= xarray[i] <= x2:
                xextrap = np.linspace(xarray[i], xarray[i + 1], n)
                slope = (yarray[i + 1] - yarray[i]) / (xarray[i + 1] - xarray[i])
                yextrap = slope * (xextrap - xarray[i]) + yarray[i]
                newxArray.extend(xextrap)
                newyArray.extend(yextrap)
            else:
                newxArray.append(xarray[i])
                newyArray.append(yarray[i])
        return np.array(newxArray), np.array(newyArray)

    # Data extrapolation for better fit
    x_data, y_data = linearExtrapolate(yyrb, q_surf, -0.005, 0.012, n=100)

    S_guess = 0.5 * (np.max(x_data) - np.min(x_data))
    x_tail = x_data[x_data > 0]
    y_tail = y_data[x_data > 0]
    try:
        def exp_tail(x, A, inv_lq): return A * np.exp(-x * inv_lq)
        popt_exp, _ = curve_fit(exp_tail, x_tail, y_tail, p0=[np.max(y_tail), 100])
        lambda_q_guess = 1 / popt_exp[1]
    except:
        lambda_q_guess = 0.002

    q0_guess = np.max(y_data) * 1.05
    s0_guess = x_data[np.argmax(y_data)]
    p0 = [S_guess, lambda_q_guess, q0_guess, s0_guess]
    bounds = ([1e-6, 1e-5, 1e2, -0.007],
              [0.05, 0.02, 1e7, 0.014])

    for i, (p, lo, hi) in enumerate(zip(p0, bounds[0], bounds[1])):
        if not (lo <= p <= hi):
            p0[i] = max(min(p, hi * 0.99), lo * 1.01)

    # Perform Eich fit
    popt, pcov = curve_fit(eichFunction, x_data, y_data, p0=p0, bounds=bounds, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))
    lamdaq = popt[1]  # in meters

    # Plot
    #plt.figure(figsize=(6, 4))
    #x_fit = np.linspace(np.min(x_data), np.max(x_data), 300)
    #plt.plot(yyrb, q_surf / 1e6, 'ro-', label='UEDGE')
    #plt.plot(x_fit, eichFunction(x_fit, *popt) / 1e6, 'b-', linewidth=2,
    #         label=r'fit: $\lambda_q$ = %.2f mm' % (lamdaq * 1000))
    #plt.xlabel(r'$r_{div} - r_{sep}$ (m)', fontsize=14)
    #plt.ylabel(r'$q_{\perp, Odiv}$ (MW/m$^2$)', fontsize=14)
    #plt.grid(True)
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig('lambdaq_eich.png', dpi=300)
    #plt.show()

    # Print fit info
    #fx_val = (com.bpol[bbb.ixmp, com.iysptrx, 0] * com.rm[bbb.ixmp, com.iysptrx, 0]) / \
    #         (com.bpol[com.nx, com.iysptrx, 0] * com.rm[com.nx, com.iysptrx, 0])
    #print('\n--- Eich Fit Results ---')
    #print(f'fx (rounded) = {np.round(fx_val):.0f} (raw = {fx_val:.4f})')
    #print(f'S    = {popt[0] * 1000:.2f} +/- {perr[0] * 1000:.2f} mm')
    #print(f'lambda_q = {lamdaq * 1000:.2f} +/- {perr[1] * 1000:.2f} mm')

    #print(f'q0   = {popt[2] / 1e6:.2f} MW/m^2')
    #print(f's0   = {popt[3]:.4f} m')

    return lamdaq


from scipy.optimize import curve_fit
from scipy.special import erfc

def get_lambdaq_eich():
  

    fx = 1
    iy = com.iysptrx + 1
    ixo = com.ixpt2[0]
    ixi = com.ixpt1[0]

    ppar = Pparallel()
    rrf = getrrf()
    xq = com.yyc[iy:-1]
    qparo = ppar[ixo, iy:-1] / com.sx[ixo, iy:-1] / rrf[ixo, iy:-1]
    qpari = -ppar[ixi, iy:-1] / com.sx[ixi, iy:-1] / rrf[ixi, iy:-1]

    psurfo = PsurfOuter()
    qsurfo = psurfo[iy:-1] / com.sxnp[com.nx, iy:-1]

    x_tgt = com.yyc[iy:-1] * fx

    def qEich(x, q0, S, lam_inv, qbg, x0):
        return q0 / 2 * np.exp((S * lam_inv / 2)**2 - x * lam_inv) * erfc(S * lam_inv / 2 - x / S) + qbg * 1e-3

    def make_eich_guess(qdata, xdata):
        if len(qdata) == 0:
            return None
        q0_guess = np.max(qdata) - np.min(qdata)
        S_guess = max((xdata[-1] - xdata[0]) / 10, 1e-3)
        lam_inv_guess = max(1 / S_guess, 1e-3)
        qbg_guess = np.min(qdata)
        x0_guess = 0
        guess = (q0_guess, S_guess, lam_inv_guess, qbg_guess, x0_guess)
        if not np.all(np.isfinite(guess)):
            return None
        return guess

    mask_paro = np.isfinite(xq) & np.isfinite(qparo) & (qparo > 0)
    xq_fit = xq[mask_paro]
    qparo_fit = qparo[mask_paro]
    qparo_eich_fit = None
    lqeo_paro = None
    guess = make_eich_guess(qparo_fit, xq_fit)
    if guess:
        try:
            qparo_eich_fit, _ = curve_fit(qEich, xq_fit, qparo_fit, p0=guess,
                                          bounds=([0, 0, 0, 0, -10], [np.inf]*5))
            lqeo_paro = 1000 / qparo_eich_fit[2]
        except Exception as e:
            print(f"qparo Eich fit failed: {e}")

    # Eich fit for q_surf (target)
    mask_surfo = np.isfinite(x_tgt) & np.isfinite(qsurfo) & (qsurfo > 0)
    x_tgt_fit = x_tgt[mask_surfo]
    qsurfo_fit = qsurfo[mask_surfo]
    qsurfo_eich_fit = None
    lqeo_surfo = None
    guess = make_eich_guess(qsurfo_fit, x_tgt_fit)
    if guess:
        try:
            qsurfo_eich_fit, _ = curve_fit(qEich, x_tgt_fit, qsurfo_fit, p0=guess,
                                           bounds=([0, 0, 0, 0, -10], [np.inf]*5))
            lqeo_surfo = 1000 / qsurfo_eich_fit[2]
        except Exception as e:
            print(f"qsurfo Eich fit failed: {e}")

    return lqeo_paro, lqeo_surfo


import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

def get_lqeo_paro():
    fx = 1  # Flux expansion from upstream to target

    # --- q_parallel calculation ---
    ppar = Pparallel()
    rrf = getrrf()
    iy = com.iysptrx + 1
    xq = com.yyc[iy:-1]
    ixo = com.ixpt2[0]

    qparo = ppar[ixo, iy:-1] / com.sx[ixo, iy:-1] / rrf[ixo, iy:-1]

    # --- Eich fit definitions ---
    def qEich(x, q0, S, lam_inv, qbg, x0):
        return q0 / 2 * np.exp((S * lam_inv / 2)**2 - x * lam_inv) * erfc(S * lam_inv / 2 - x / S) + qbg*1e-3

    mask_paro = np.isfinite(xq) & np.isfinite(qparo) & (qparo > 0)
    xq_fit = xq[mask_paro]
    qparo_fit = qparo[mask_paro]

    def make_eich_guess(qdata, xdata):
        if len(qdata) == 0:
            return None
        q0_guess = np.max(qdata) - np.min(qdata)
        S_guess = max((xdata[-1] - xdata[0]) / 10, 1e-3)
        lam_inv_guess = max(1 / S_guess, 1e-3)
        qbg_guess = np.min(qdata)
        x0_guess = 0
        guess = (q0_guess, S_guess, lam_inv_guess, qbg_guess, x0_guess)
        if not np.all(np.isfinite(guess)):
            return None
        return guess

    qparo_eich_guess = make_eich_guess(qparo_fit, xq_fit)
    if qparo_eich_guess is None:
        return None
    try:
        qparo_eich_fit, _ = curve_fit(qEich, xq_fit, qparo_fit, p0=qparo_eich_guess,
                                      bounds=([0, 0, 0, 0, -10], [np.inf] * 5))
        lqeo_paro = 1000 / qparo_eich_fit[2]  # mm
    except Exception:
        lqeo_paro = None

    return lqeo_paro


import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

def fit_out_in_eich_exp_shahinul():
    yyc = com.yyc.reshape(-1)
    bbb.plateflux()
    q_data = (bbb.sdrrb + bbb.sdtrb).reshape(-1)
    yyrb = com.yyrb.reshape(-1)
    q_surf = q_data

    iy = com.iysptrx + 1
    xq = yyc[iy:-1]
    rrf = getrrf()
    ixo = com.nx#com.ixpt2[0]
    ixi = 0#com.ixpt1[0]
    ppar = Pparallel()

    psurfo = PsurfOuter()
    qsurfo = psurfo[iy:-1] / com.sxnp[com.nx, iy:-1]
    x_tgt = com.yyc[iy:-1] * 1

    qparo = ppar[ixo, iy:-1] / com.sx[ixo, iy:-1] / rrf[ixo, iy:-1]
    qpari = -ppar[ixi, iy:-1] / com.sx[ixi, iy:-1] / rrf[ixi, iy:-1]

    # --- Exponential Fit ---
    expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)

    try:
        omax = np.argmax(qparo)
        qofit, _ = curve_fit(expfun, xq[omax:], qparo[omax:], p0=[np.max(qparo), 1000], bounds=(0, np.inf))
        lqo = 1000 / qofit[1]
    except:
        lqo = np.nan

    try:
        imax = np.argmax(qpari)
        qifit, _ = curve_fit(expfun, xq[imax:], qpari[imax:], p0=[np.max(qpari), 1000], bounds=(0, np.inf))
        lqi = 1000 / qifit[1]
    except:
        lqi = np.nan

    # --- Eich fit for q_parallel (outer) ---
    def qEich(x, q0, S, lam_inv, qbg, x0):
        x = x - x0
        expo = np.clip((S * lam_inv / 2)**2 - x * lam_inv, -700, 700) 
        exp_term = np.exp(expo)
        erfc_arg = S * lam_inv / 2 - x / S
        erfc_term = erfc(erfc_arg) 
        product = exp_term * erfc_term
        product = np.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)
       # return q0 / 2 * np.exp((S * lam_inv / 2)**2 - x * lam_inv) * erfc(S * lam_inv / 2 - x / S) + qbg * 1e-3
        #return q0 / 2 * np.exp(expo) * erfc(erfc_arg) + qbg * 1e-3
        return q0 / 2 * product + qbg * 1e-3

    def make_eich_guess(qdata, xdata):
        if len(qdata) == 0:
            return None
        q0_guess = np.max(qdata) - np.min(qdata)
        S_guess = max((xdata[-1] - xdata[0]) / 10, 1e-3)
        lam_inv_guess = max(1 / S_guess, 1e-3)
        qbg_guess = np.min(qdata)
        x0_guess = 0
        return (q0_guess, S_guess, lam_inv_guess, qbg_guess, x0_guess)

    # --- Fit q_parallel Eich ---
    mask_paro = np.isfinite(xq) & np.isfinite(qparo) & (qparo > 0)
    qparo_guess = make_eich_guess(qparo[mask_paro], xq[mask_paro])
    if qparo_guess:
        try:
            qparo_eich_fit, _ = curve_fit(qEich, xq[mask_paro], qparo[mask_paro], p0=qparo_guess,
                                          bounds=([0, 0, 0, 0, -10], [np.inf]*5))
            lqeo_paro = 1000 / qparo_eich_fit[2]
        except:
            lqeo_paro = np.nan
    else:
        lqeo_paro = np.nan

    # --- Fit q_target Eich ---
    mask_surfo = np.isfinite(x_tgt) & np.isfinite(qsurfo) & (qsurfo > 0)
    qsurfo_guess = make_eich_guess(qsurfo[mask_surfo], x_tgt[mask_surfo])
    if qsurfo_guess:
        try:
            qsurfo_eich_fit, _ = curve_fit(qEich, x_tgt[mask_surfo], qsurfo[mask_surfo], p0=qsurfo_guess,
                                           bounds=([0, 0, 0, 0, -10], [np.inf]*5))
            lqeo_surfo = 1 / qsurfo_eich_fit[2]
        except:
            lqeo_surfo = np.nan
    else:
        lqeo_surfo = np.nan

    # --- Eich fit for q_surface at plate (cross-field) ---
    def eichFunction(x, S, lq, q0, s0):
        fx = (com.bpol[bbb.ixmp, com.iysptrx, 0] * com.rm[bbb.ixmp, com.iysptrx, 0]) / \
             (com.bpol[com.nx, com.iysptrx, 0] * com.rm[com.nx, com.iysptrx, 0])
        fx = np.round(fx)
        sBar = x - s0
        t0 = (S / (2 * lq * fx)) ** 2
        t1 = sBar / (lq * fx)
        t2 = S / (2 * lq * fx)
        t3 = sBar / S
        exponent = np.clip(t0 - t1, -700, 700)
        exp_term = np.exp(exponent)
        erfc_arg = t2 - t3
        erfc_term = erfc(erfc_arg)
        product = exp_term * erfc_term
        product = np.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)

        q_back = q0 * 1e-5
        result = (q0 / 2) * product + q_back
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def linearExtrapolate(xarray, yarray, x1, x2, n=10):
        newxArray, newyArray = [], []
        for i in range(len(xarray) - 1):
            if x1 <= xarray[i] <= x2:
                xextrap = np.linspace(xarray[i], xarray[i + 1], n)
                slope = (yarray[i + 1] - yarray[i]) / (xarray[i + 1] - xarray[i])
                yextrap = slope * (xextrap - xarray[i]) + yarray[i]
                newxArray.extend(xextrap)
                newyArray.extend(yextrap)
            else:
                newxArray.append(xarray[i])
                newyArray.append(yarray[i])
        return np.array(newxArray), np.array(newyArray)

    x_data, y_data = linearExtrapolate(yyrb, q_surf, -0.005, 0.012, n=100)

    try:
        def exp_tail(x, A, inv_lq): return A * np.exp(-x * inv_lq)
        x_tail = x_data[x_data > 0]
        y_tail = y_data[x_data > 0]
        popt_exp, _ = curve_fit(exp_tail, x_tail, y_tail, p0=[np.max(y_tail), 100])
        lambda_q_guess = 1 / popt_exp[1]
    except:
        lambda_q_guess = 0.002

    S_guess = 0.5 * (np.max(x_data) - np.min(x_data))
    q0_guess = np.max(y_data) * 1.05
    s0_guess = x_data[np.argmax(y_data)]
    p0 = [S_guess, lambda_q_guess, q0_guess, s0_guess]

    bounds = ([1e-6, 1e-5, 1e2, -0.007],
              [0.05, 0.02, 1e7, 0.014])
    #print('\nChecking bounds feasibility:')
    for i, (p, lo, hi) in enumerate(zip(p0, bounds[0], bounds[1])):
        #print(f'param {i}: p0 = {p:.5e}, lower = {lo:.5e}, upper = {hi:.5e}')
        if not (lo <= p <= hi):
            #print(f'?? WARNING: p0[{i}] is out of bounds, adjusting...')
            p0[i] = max(min(p, hi * 0.99), lo * 1.01)


    try:
        popt, _ = curve_fit(eichFunction, x_data, y_data, p0=p0, bounds=bounds, maxfev=100000)
        lq_surf = popt[1] * 1000  # in mm
    except:
        lq_surf = np.nan

    return lqo, lqi, lqeo_paro, lq_surf


def compute_lambdaq_extended(Bt=1, q95=3.7, R=0.93, Psol = 4):
    
    lambda_q_mm = 0.73 * (Bt ** -0.78) * (q95 ** 1.2) * (R ** 0.0)*Psol**(0.1)

    print("\n--- Eich ?q Scaling ---")
    print("Formula: lambda_q [mm] = 0.73 * Bt^(-0.78) * q95^(1.2) * R^(0.0)*Psol**0.1")
    print(f"?q = 0.73 * ({Bt:.3f})^(-0.78) * ({q95:.3f})^(1.2) * ({R:.3f})^(0.1)")
   # print("?q = 0.73 × Bpol^(-1.19)")  


    return lambda_q_mm

def compute_lambdaq_bpol(Bpol=0.5499):
    """
    Compute Eich ?q [mm] based on local Bpol in Tesla.
    lambda_q = 0.73 * Bpol^(-1.19)
    """
    lambda_q = 0.73 * Bpol**(-1.19)
    print("\n--- Eich ?q Estimate ---")
    return lambda_q

def compute_fx():

    # Divertor values 
    Bpol_div = com.bpol[com.nx, :, 0]
    Btor_div = com.bphi[com.nx, :, 0]
    B_div = np.sqrt(Bpol_div**2 + Btor_div**2)

    # Midplane values
    Bpol_omp = com.bpol[bbb.ixmp, :, 0]
    Btor_omp = com.bphi[bbb.ixmp, :, 0]
    B_omp = np.sqrt(Bpol_omp**2 + Btor_omp**2)

    # Pitch angle at divertor
    sin_theta = Bpol_div / B_div

    # Flux expansion
    fx = (B_div / B_omp) * (1.0 / sin_theta)
    
    fx_simple = (com.bpol[bbb.ixmp, :, 0] * com.rm[bbb.ixmp, :, 0]) / \
         (com.bpol[com.nx, :, 0] * com.rm[com.nx, :, 0])

    return fx, fx_simple

def calculate_flux_expansion():
    """
    Calculate poloidal and tilt flux expansion for the CD configuration.
    D. Moulton et al., Nucl. Fusion 64 (2024) 076049 (31pp) 

    Uses:
        com.bpol, com.bphi: 3D arrays of poloidal and toroidal fields
        bbb.ixmp, com.iysptrx, com.nx: indices
        fieldLineAngle(): function returning field angle array

    """
    bpol_u = com.bpol[bbb.ixmp,com.iysptrx+1,0]
    bphi_u = abs(com.bphi[bbb.ixmp,com.iysptrx+1,0])
    bpol_t = com.bpol[com.nx,com.iysptrx+1,0]
    bphi_t = abs(com.bphi[com.nx,com.iysptrx+1,0])  # Fixed this line
    alpha_tilt_deg = fieldLineAngle()
    alpha_tilt_target = alpha_tilt_deg[com.nx,com.iysptrx+1]
    alpha_tilt = np.radians(alpha_tilt_target)

    FX_theta = (bpol_u / bphi_u) / (bpol_t / bphi_t)
    FX_tilt = 1 / np.sin(alpha_tilt)
    return  FX_theta




def compute_lambdaq_extended(Bt=1.0, R=0.93, kappa=2.75, Psol=4.0, Ip=2.0):
    A = 1.7
    a = R/A
    q95 = 3# (5 * a**2 * Bt) / (kappa * R * Ip) for NSTX-U

    # Eich extended empirical scaling for ?q in mm
    lambda_q_mm = 0.73 * (Bt ** -0.78) * (q95 ** 1.2) * (R ** 0.1) * (Psol ** 0.1)

    return lambda_q_mm, q95


def eich_exp_shahinul_final(omp=False, save_prefix='lambdaq_result'):
 

    bbb.plateflux()
    ppar = Pparallel()
    rrf = getrrf()

    fx_out = calculate_flux_expansion()
    fx_out = np.round(fx_out)

    fx_in = (com.bpol[bbb.ixmp, :, 0] * com.rm[bbb.ixmp, :, 0]) / \
            (com.bpol[0, :, 0] * com.rm[0, :, 0])
    fx_in = np.round(fx_in[com.iysptrx + 1])

    q_para_omp = ppar[bbb.ixmp, :-1] / com.sx[bbb.ixmp, :-1] / rrf[bbb.ixmp, :-1]
    q_para_odiv = ppar[com.nx, :-1] / com.sxnp[com.nx, :-1] / rrf[com.nx, :-1]
    q_para_idiv = abs(ppar[0, :-1]) / com.sxnp[0, :-1] / rrf[0, :-1]

    q_data = bbb.sdrrb + bbb.sdtrb
    q_data_idiv = bbb.sdrlb + bbb.sdtlb
    q_perp_idiv = q_data_idiv.reshape(-1)[:-1]
    q_perp = q_data.reshape(-1)[:-1]
    yyc = com.yyc.reshape(-1)[:-1]
    yyrb = com.yyrb.reshape(-1)[:-1]
    yylb = com.yylb.reshape(-1)[:-1]

    iy_sep = com.iysptrx + 1

    s_idiv = yylb[iy_sep:]
    q_perp_odiv = q_perp[iy_sep:]
    q_perp_idiv = q_perp_idiv[iy_sep:]
    s_odiv = yyrb[iy_sep:]

    def expfit(xq, qpara):
        try:
            imax = np.argmax(qpara)
            expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)
            qofit, _ = curve_fit(expfun, xq[imax:], qpara[imax:], p0=[np.max(qpara), 1000], bounds=(0, np.inf))
            return 1000 / qofit[1]  # ?q in mm
        except Exception as e:
            print('Exponential fit failed:', e)
            return 1.0

    lqo = expfit(yyc[iy_sep:-1], q_para_odiv[iy_sep:-1])
    lqi = expfit(yyc[:iy_sep], q_para_idiv[:iy_sep])

    # === Eich Function ===
    def eichFunction(x, S, lq, q0, s0, fx_val):
        sBar = x - s0
        t0 = (S / (2 * lq * fx_val)) ** 2
        t1 = sBar / (lq * fx_val)
        t2 = S / (2 * lq * fx_val)
        t3 = sBar / S
        
        return (q0 / 2) * np.exp(t0 - t1) * erfc(t2 - t3)

    def eich_fit_wrapper(s, q, fx_val):
        interp_fun = interp1d(s, q, kind='cubic', fill_value="extrapolate")
        s_interp = np.linspace(s.min(), s.max(), 300)
        q_interp = interp_fun(s_interp)
        s0_guess = np.median(s_interp)
        q0_guess = np.max(q_interp)
        p0 = [0.003, 0.002, q0_guess, s0_guess]
        bounds = ([0.0005, 0.0005, 1e6, s_interp.min() - 0.01],
                  [0.02, 0.02, 1e9, s_interp.max() + 0.01])
        try:
            popt, _ = curve_fit(lambda x, S, lq, q0, s0: eichFunction(x, S, lq, q0, s0, fx_val),
                                s_interp, q_interp, p0=p0, bounds=bounds, maxfev=10000)
            return popt[1] * 1000  # ?q in mm
        except Exception as e:
            print("Eich fit failed:", e)
            return 0.0

    lambdaq_eich_out_mm = eich_fit_wrapper(s_odiv, q_perp_odiv, fx_out)
    lambdaq_eich_in_mm = eich_fit_wrapper(s_idiv, q_perp_idiv, fx_in)

    return lqo, lqi, lambdaq_eich_out_mm,  lambdaq_eich_in_mm
    # exp fit outer, inner, eich fit outer , inner


def eich_exp_shahinul_odiv_final(omp=False, save_prefix='lambdaq_result'):
  
    fx = calculate_flux_expansion()
    fx = np.round(fx)
   
    #print("fx", fx)

    # === Select which q_parallel to fit (OMP vs ODIV)
    bbb.plateflux()
    ppar = Pparallel()
    rrf = getrrf()
    q_para_omp =  ppar[bbb.ixmp,:-1]/com.sx[bbb.ixmp,:-1]/rrf[bbb.ixmp,:-1]
    q_para_odiv = ppar[com.nx,:-1]/com.sx[com.nx,:-1]/rrf[com.nx,:-1]
    q_data = bbb.sdrrb + bbb.sdtrb
    q_perp_odiv = q_data.reshape(-1)[:-1]
    yyc = com.yyc.reshape(-1)[:-1]
    yyrb = com.yyrb.reshape(-1)[:-1]
   

    s_omp = com.yyrb[:-1]
    q_fit = q_para_omp if omp else q_para_odiv

    s_omp = s_omp.flatten()
    q_fit = q_fit.flatten()

    interp_fun = interp1d(s_omp, q_fit, kind='cubic', fill_value="extrapolate")
    s_interp = np.linspace(s_omp.min(), s_omp.max(), 300)
    q_interp = interp_fun(s_interp)
    iy_sep = com.iysptrx+1

    # === Exponential Fit ===
    xq = yyc[iy_sep:-1]
    
    qparo = q_fit[iy_sep:-1]
    expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)

    try:
        omax = np.argmax(qparo)
        expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)
        qofit, _ = curve_fit(expfun, xq[omax:], qparo[omax:], p0=[np.max(qparo), 1000], bounds=(0, np.inf))
        lqo = 1000 / qofit[1]
    except Exception as e:
        print('q_parallel outer fit failed:', e)
        qofit = None
        lqo = 1.0

    xq_omp = xq
    
    # === Eich Function ===
    def eichFunction(x, S, lq, q0, s0):
        sBar = x - s0
        t0 = (S / (2 * lq * fx)) ** 2
        t1 = sBar / (lq * fx)
        t2 = S / (2 * lq * fx)
        t3 = sBar / S
        q_back = q0*1e-3
        return (q0 / 2) * np.exp(t0 - t1) * erfc(t2 - t3)+q_back 

    # === Fit Eich Function ===
    s_omp = yyrb
    q_omp = q_perp_odiv
    interp_fun = interp1d(s_omp, q_omp, kind='cubic', fill_value="extrapolate")
    s_interp = np.linspace(s_omp.min(), s_omp.max(), 300)
    q_interp = interp_fun(s_interp)
    s_fit = s_interp
    q_fit = q_interp
    s0_guess = np.median(s_fit)
    q0_guess = np.max(q_fit)

    p0 = [0.003, 0.002, q0_guess, s0_guess]
    bounds = (
        [0.0005, 0.0005, 1e6, s_fit.min() - 0.01],
        [0.02,   0.02,   1e9, s_fit.max() + 0.01]
    )

    try:
        popt, _ = curve_fit(eichFunction, s_fit, q_fit, p0=p0, bounds=bounds, maxfev=10000)
        S_fit, lambda_q_fit, q0_fit, s0_fit = popt
        q_fit_full = eichFunction(s_fit, *popt)

        # R²
        residuals = q_fit - q_fit_full
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((q_fit - np.mean(q_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    except Exception as e:
        print("Eich fit failed:", e)
        popt = [0, 0, 0, 0]
        lambda_q_fit = 0.0
        r_squared = 0.0



    return lqo, lambda_q_fit*1000, 


def eich_lambda_q(a = 0.5471, R_geo = 0.93, P_SOL = 4):
    """
    Calculate Eich scaling for lambda_q in mm. 
    Table 3, Eich NF 2013 for NSTX U and Mast-U

    Parameters:
        a (float): Minor radius (m)
        R_geo (float): Geometric major radius (m)
        B_pol_omp (float): Poloidal magnetic field at outer midplane (T)
        P_SOL (float): Power crossing separatrix (MW)

    Returns:
        float: ?_q in mm
    """
    epsilon = a / R_geo  # Inverse aspect ratio
    B_pol_omp = com.bpol[bbb.ixmp,com.iysptrx,0]

    lambda_q_mm = (
        1.35
        * (epsilon ** 0.42)
        * (R_geo ** 0.04)
        * (B_pol_omp ** -0.92)
        * (P_SOL ** -0.02)
    )

    return lambda_q_mm



def para_conn_length(bbb, com, do_plot=False):
    """
    Calculate the parallel magnetic connection length along flux tubes.

    Parameters:
    -----------
    bbb : object
        Object containing plasma profiles like te, ev, and index ixmp.
    com : object
        Object containing grid and magnetic field data and indices like
        bphi, bpol, rm, zm, ixpt2, iysptrx, nx.
    do_plot : bool, optional
        Whether to plot Te vs connection length for the 3rd flux tube outside separatrix.

    Returns:
    --------
    connlen : np.ndarray
        2D array of cumulative parallel connection lengths [nxg, nyg-1] in meters.
    """

    # Compute total magnetic field magnitude
    Bt = np.sqrt(com.bphi[:, :, 0]**2 + com.bpol[:, :, 0]**2)


    # Prepare grid data
    b = np.copy(Bt)
    bpol = np.copy(com.bpol[:, :, 0])
    rm = com.rm[:, :, 0]
    zm = com.zm[:, :, 0]

    # Avoid division by zero
    rbfbt = b / (bpol + 1e-50)
    rbfbt[rbfbt == 0] = 1

    nxg, nyg = rm.shape

    # Midpoints for R,Z of each cell along poloidal direction
    midpoints = np.zeros((2, nxg + 1, nyg - 1))
    midpoints[0, :-1] = 0.5 * (rm[:, :-1] + rm[:, 1:])
    midpoints[1, :-1] = 0.5 * (zm[:, :-1] + zm[:, 1:])
    # Right boundary midpoints
    midpoints[0, -1] = 0.5 * (rm[-1, :-1] + rm[-1, 1:])
    midpoints[1, -1] = 0.5 * (zm[-1, :-1] + zm[-1, 1:])

    # Poloidal vector differences and lengths
    polloc = midpoints[:, 1:] - midpoints[:, :-1]
    pollength = np.sqrt(polloc[0]**2 + polloc[1]**2)

    # Parallel length of each cell
    parlength = pollength * rbfbt[:, :-1]

    # Initialize connection length array
    connlen = np.zeros((nxg, nyg - 1))

    # Compute cumulative connection length along poloidal direction
    for i in range(1, nxg):
        connlen[i, :] = connlen[i - 1, :] + parlength[i, :]

    # Fix guard cells at boundaries
    connlen[:, 0] = connlen[:, 1]
    connlen[:, -1] = connlen[:, -2]

    print("Parallel magnetic connection length array shape:", connlen.shape)

    # Calculate some connection length differences for key points
    Odiv_Oxpt = connlen[com.ixpt2, com.iysptrx] - connlen[com.nx, com.iysptrx]
    OMP_Oxpt = connlen[bbb.ixmp, com.iysptrx] - connlen[com.ixpt2, com.iysptrx]
    OMP_Odiv = connlen[bbb.ixmp, com.iysptrx] - connlen[com.nx, com.iysptrx]

    #print('-------------')
    #print('Odiv to X point conn L:', abs(Odiv_Oxpt))
    #print('OMP to O xpt conn L:', abs(OMP_Oxpt))
    #print('OMP to Odiv conn L:', abs(OMP_Odiv))
    #print('---------------')

    # Save connection length and parallel length arrays as CSV
    pd.DataFrame(connlen).to_csv('connlen.csv', index=False, header=False)
    pd.DataFrame(parlength).to_csv('parlength.csv', index=False, header=False)

    if do_plot:
        kwargs = {'color': 'red', 'fontsize': 12}

        plt.plot(connlen[:, com.iysptrx + 2], bbb.te[:, com.iysptrx + 2] / bbb.ev)
        plt.xlabel("L${||}$ from Inner div (m)")
        plt.ylabel("T$_e$ (eV)")
        plt.title("3rd FT Outside of Sep")
        plt.text(connlen[bbb.ixmp, com.iysptrx + 2], 
                 bbb.te[bbb.ixmp, com.iysptrx + 2] / bbb.ev, 'OMP', **kwargs)
        plt.grid()
        plt.show()

    return connlen


def get_surface_heatflux_components(
    target='outer'
):
    """
    Return surface heat flux components and surface recombination breakdown
    for inner or outer divertor (default is outer).

    Parameters:
    -----------
    target : str
        'outer' (default) or 'inner'

    Returns:
    --------
    dict
        Dictionary with keys:
            'y_pos', 'conv_cond', 'surf_recomb', 'ion_ke', 'h_photons',
            'imp_photons', 'total_flux', 'q_D_MW', 'q_Li1_MW', 'q_Li2_MW', 'q_Li3_MW'
    """
    import numpy as np

    bbb.plateflux()
    bbb.pradpltwl()

    yylb = com.yylb[1:-1]
    yyrb = com.yyrb[1:-1]
    pwrx = bbb.feex + bbb.feix
    idx_D   = 0
    idx_Li1 = com.nhsp + 0
    idx_Li2 = com.nhsp + 1
    idx_Li3 = com.nhsp + 2
    eV_to_J = bbb.ev
    I_D  = bbb.ebind
    I0, I1, I2 = 5.39, 75.6, 122.4
    E_Li1, E_Li2, E_Li3 = I0, I0+I1, I0+I1+I2

    q_data = bbb.sdrrb + bbb.sdtrb
    q_data_odiv = q_data.reshape(-1)
    q_data_idiv = (bbb.sdrlb + bbb.sdtlb).reshape(-1)

    if target == 'outer':
        xtarget = com.nx
        plateIndex = 1
        xsign = 1
        y_pos = yyrb
        q_data_target = q_data_odiv
    elif target == 'inner':
        xtarget = 0
        plateIndex = 0
        xsign = -1
        y_pos = yylb
        q_data_target = q_data_idiv
    else:
        raise ValueError("target must be 'inner' or 'outer'")

    conv_cond = xsign * pwrx[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6

    Gamma_D   = xsign * bbb.fnix[xtarget, 1:-1, idx_D]
    Gamma_Li1 = xsign * bbb.fnix[xtarget, 1:-1, idx_Li1]
    Gamma_Li2 = xsign * bbb.fnix[xtarget, 1:-1, idx_Li2]
    Gamma_Li3 = xsign * bbb.fnix[xtarget, 1:-1, idx_Li3]

    P_D   = Gamma_D   * I_D   * eV_to_J
    P_Li1 = Gamma_Li1 * E_Li1 * eV_to_J
    P_Li2 = Gamma_Li2 * E_Li2 * eV_to_J
    P_Li3 = Gamma_Li3 * E_Li3 * eV_to_J

    area = com.sxnp[xtarget, 1:-1]
    q_D_MW   = P_D   / area * 1e-6
    q_Li1_MW = P_Li1 / area * 1e-6
    q_Li2_MW = P_Li2 / area * 1e-6
    q_Li3_MW = P_Li3 / area * 1e-6
    surf_recomb = xsign * (q_D_MW + q_Li1_MW + q_Li2_MW + q_Li3_MW)

    try:
        ion_ke = xsign * PionParallelKE()[xtarget, 1:-1] / area / 1e6
    except Exception:
        ion_ke = np.zeros_like(area)

    h_photons = bbb.pwr_plth[1:-1, plateIndex] / 1e6
    imp_photons = bbb.pwr_pltz[1:-1, plateIndex] / 1e6
    total_flux = conv_cond + surf_recomb + ion_ke + h_photons + imp_photons

    return {
        'y_pos': y_pos,
        'conv_cond': conv_cond,
        'surf_recomb': surf_recomb,
        'ion_ke': ion_ke,
        'h_photons': h_photons,
        'imp_photons': imp_photons,
        'total_flux': total_flux,
        'q_D_MW': q_D_MW,
        'q_Li1_MW': q_Li1_MW,
        'q_Li2_MW': q_Li2_MW,
        'q_Li3_MW': q_Li3_MW
    }


def plot_Li_flux_omp(fontsize=14, abs_val=True):
    Gamma_Li1 = bbb.fnix[bbb.ixmp,1:-1,2] / com.sx[bbb.ixmp,1:-1]
    Gamma_Li2 = bbb.fnix[bbb.ixmp,1:-1,3] / com.sx[bbb.ixmp,1:-1]
    Gamma_Li3 = bbb.fnix[bbb.ixmp,1:-1,4] / com.sx[bbb.ixmp,1:-1]
    Gamma_Li = Gamma_Li1 + Gamma_Li2 + Gamma_Li3

    return Gamma_Li




