import os
import datetime
import glob
import copy
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from uedge import bbb, com, api, grd
from uedge import __version__ as uedgeVersion
#from UEDGE_utils import analysis, sparc
from UEDGE_utils import analysis
from UEDGE_utils import run
from matplotlib.collections import LineCollection
from scipy.special import erfc, erfcx
from scipy.interpolate import interp1d
import scipy





def zoom(factor):
    '''
    Zoom in/out of plot view, adjusting x and y axes by the same factor.
    '''
    x1, x2 = plt.gca().get_xlim()
    plt.xlim([(x1+x2)/2-factor*(x2-x1)/2, (x1+x2)/2+factor*(x2-x1)/2])
    y1, y2 = plt.gca().get_ylim()
    plt.ylim([(y1+y2)/2-factor*(y2-y1)/2, (y1+y2)/2+factor*(y2-y1)/2])


def getPatches():
    '''
    Create cell patches for 2D plotting.
    '''
    patches = []
    for iy in np.arange(0,com.ny+2):
        for ix in np.arange(0,com.nx+2):
            rcol=com.rm[ix,iy,[1,2,4,3]]
            zcol=com.zm[ix,iy,[1,2,4,3]]
            patches.append(np.column_stack((rcol,zcol)))
    return patches
    

def plotvar(var, title='', label=None, iso=True, rzlabels=True, stats=True, message=None,
            orientation='vertical', vmin=None, vmax=None, minratio=None, cmap='viridis', log=False,
            patches=None, show=True, sym=False, showGuards=False, colorbar=True, extend=None, norm=None,linscale=1):
    '''
    Plot a quantity on the grid in 2D. 
    
    Args:
        patches: supplying previously computed patches lowers execution time
        show: set this to False if you are calling this method to create a subplot
        minratio: set vmin to this fraction of vmax (useful for log plots with large range)
        sym: vmax=-vmin
    '''
    plt.rcParams['axes.axisbelow'] = True
    
    if not patches:
        patches = getPatches()

    # reorder value in 1-D array
    vals = var.T.flatten()
    
    # Set vmin and vmax disregarding guard cells
    if not vmax:
        vmax = np.max(analysis.nonGuard(var))
    if not vmin:
        vmin = np.min(analysis.nonGuard(var))
        
    if show:
        rextent = np.max(com.rm)-np.min(com.rm)
        zextent = np.max(com.zm)-np.min(com.zm)
        fig, ax = plt.subplots(1, figsize=(4.8, 6.4))
    else:
        ax = plt.gca()
        
    if sym:
        maxval = np.max(np.abs([vmax, vmin]))
        vmax = maxval
        vmin = -maxval
        cmap = plt.cm.bwr
        plt.gca().set_facecolor('lightgray')
    else:
        plt.gca().set_facecolor('gray')
        
    _extend = 'neither' # minratio related
        
    # Need to make a copy for set_bad
    cmap = copy.copy(cmap)
    
    if not np.any(var > 0):
        log = False

    if log:
        cmap.set_bad((1,0,0,1))
        if vmin > 0:
            if minratio:
                vmin = vmax/minratio
                _extend = 'min'
            _norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if minratio:
                # linscale=np.log10(minratio)/linscale
                _norm = matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=vmax/minratio, linscale=linscale, base=10)
            else:
                _norm = matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=(vmax-vmin)/1000, linscale=linscale, base=10)
    else:
        _norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if norm:
        _norm = norm
    p = PolyCollection(patches, array=np.array(vals), cmap=cmap, norm=_norm)
    
    if (vmin > np.min(analysis.nonGuard(var))) and (vmax < np.max(analysis.nonGuard(var))):
        _extend = 'both'
    elif vmin > np.min(analysis.nonGuard(var)):
        _extend = 'min'
    elif vmax < np.max(analysis.nonGuard(var)):
        _extend = 'max'
    
    if extend:
       _extend = extend
    
    if showGuards:
        plt.scatter(com.rm[0,:,0], com.zm[0,:,0], c=var[0,:], cmap=cmap)
        plt.scatter(com.rm[com.nx+1,:,0], com.zm[com.nx+1,:,0], c=var[com.nx+1,:], cmap=cmap)
        plt.scatter(com.rm[:,0,0], com.zm[:,0,0], c=var[:,0], cmap=cmap)
        plt.scatter(com.rm[:,com.ny+1,0], com.zm[:,com.ny+1,0], c=var[:,com.ny+1], cmap=cmap)
        if 'dnbot' in com.geometry[0].decode('UTF-8'):
            plt.scatter(com.rm[bbb.ixmp-1,:,0], com.zm[bbb.ixmp-1,:,0], c=var[bbb.ixmp-1,:], cmap=cmap)
            plt.scatter(com.rm[bbb.ixmp,:,0], com.zm[bbb.ixmp,:,0], c=var[bbb.ixmp,:], cmap=cmap)

    ax.grid(False)
    plt.title(title)
    if rzlabels:
        plt.xlabel(r'$R$ [m]')
        plt.ylabel(r'$Z$ [m]')
    else:
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        
    ax.add_collection(p)
    ax.autoscale_view()    

    if colorbar:
        if sym and log and minratio:
            # maxpow = int(np.log10(vmax))
            # minpow = int(np.log10(vmax/minratio)+0.5)
            # print(minpow, maxpow)
            # ticks = [-10**p for p in range(maxpow, minpow-1, -1)]
            # ticklabels = ['$-10^{%d}$' % p for p in range(maxpow, minpow-1, -1)]
            # ticks.append(0)
            # ticklabels.append('0')
            # ticks.extend([10**p for p in range(minpow, maxpow+1)])
            # ticklabels.extend(['$10^{%d}$' % p for p in range(minpow, maxpow+1)])
            
            # ticks = np.arange(1,10)
            # cbar = plt.colorbar(p, label=label, extend=_extend, orientation=orientation, ticks=ticks)
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels(ticklabels)
            # minticks = []
            # for p in range(maxpow, minpow-1, -1):
            #     minticks.extend([i*10**p for i in range(2, 10)])
            #cbar.set_ticks(minticks)
            cbar = plt.colorbar(p, label=label, extend=_extend, orientation=orientation)
        else:
            cbar = plt.colorbar(p, label=label, extend=_extend, orientation=orientation)
    
    if iso:
        plt.axis('equal')  # regular aspect-ratio
        
    if stats:
        text =  '      max %.2g\n' % np.max(analysis.nonGuard(var))
        text += '      min %.2g\n' % np.min(analysis.nonGuard(var))
        text += ' min(abs) %.2g\n' % np.min(np.abs(analysis.nonGuard(var)))
        text += '     mean %.2g\n' % np.mean(analysis.nonGuard(var))
        text += 'mean(abs) %.2g' % np.mean(np.abs(analysis.nonGuard(var)))
    if message:
        text = message
    if stats or message:        
        plt.text(0.01, 0.01, text, fontsize=4, color='black', family='monospace',
                 horizontalalignment='left', verticalalignment='bottom', 
                 transform=plt.gca().transAxes)

    if show:
        plt.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], '--m', linewidth=1)
        plt.tight_layout()
        plt.show(block=False)



from matplotlib.collections import PolyCollection
import copy

def plotvar_shahinul(var, title='', label=None, iso=True, rzlabels=True, stats=True, message=None,
            orientation='vertical', vmin=None, vmax=None, minratio=None, cmap='viridis', log=False,
            patches=None, show=True, sym=False, showGuards=False, colorbar=True, extend=None, 
            norm=None, linscale=1, xlim=None, ylim=None):
    '''
    Plot a quantity on the grid in 2D. 
    
    Args:
        var: 2D array to plot
        patches: supplying previously computed patches lowers execution time
        show: set this to False if you are calling this method to create a subplot
        minratio: set vmin to this fraction of vmax (useful for log plots with large range)
        sym: enforce symmetric colormap limits around zero (vmax=-vmin)
        xlim: tuple (xmin, xmax) for x-axis limits
        ylim: tuple (ymin, ymax) for y-axis limits
    '''
    plt.rcParams['axes.axisbelow'] = True
    
    if not patches:
        patches = getPatches()

    # reorder values into 1-D array
    vals = var.T.flatten()
    
    # Set vmin and vmax disregarding guard cells
    if vmax is None:
        vmax = np.max(analysis.nonGuard(var))
    if vmin is None:
        vmin = np.min(analysis.nonGuard(var))
        
    if show:
        fig, ax = plt.subplots(1, figsize=(4.8, 6.4))
    else:
        ax = plt.gca()
        
    if sym:
        maxval = np.max(np.abs([vmax, vmin]))
        vmax = maxval
        vmin = -maxval
        cmap = plt.cm.bwr
        ax.set_facecolor('lightgray')
    else:
        ax.set_facecolor('gray')
        
    _extend = 'neither' # minratio related
        
    # Ensure cmap is a Colormap object
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        cmap = copy.copy(cmap)  # avoid mutating global colormap
    
    if not np.any(var > 0):
        log = False

    # Define normalization
    if log:
        cmap.set_bad((1,0,0,1))  # red for invalid
        if vmin > 0:
            if minratio:
                vmin = vmax/minratio
                _extend = 'min'
            _norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if minratio:
                _norm = matplotlib.colors.SymLogNorm(
                    vmin=vmin, vmax=vmax,
                    linthresh=vmax/minratio, linscale=linscale, base=10
                )
            else:
                _norm = matplotlib.colors.SymLogNorm(
                    vmin=vmin, vmax=vmax,
                    linthresh=(vmax-vmin)/1000, linscale=linscale, base=10
                )
    else:
        _norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    if norm:
        _norm = norm

    # Add PolyCollection
    p = PolyCollection(patches, array=np.array(vals), cmap=cmap, norm=_norm)
    
    # Decide extend for colorbar
    ng_min = np.min(analysis.nonGuard(var))
    ng_max = np.max(analysis.nonGuard(var))
    if (vmin > ng_min) and (vmax < ng_max):
        _extend = 'both'
    elif vmin > ng_min:
        _extend = 'min'
    elif vmax < ng_max:
        _extend = 'max'
    
    if extend:
        _extend = extend
    
    # Optionally plot guard cells
    if showGuards:
        plt.scatter(com.rm[0,:,0], com.zm[0,:,0], c=var[0,:], cmap=cmap)
        plt.scatter(com.rm[com.nx+1,:,0], com.zm[com.nx+1,:,0], c=var[com.nx+1,:], cmap=cmap)
        plt.scatter(com.rm[:,0,0], com.zm[:,0,0], c=var[:,0], cmap=cmap)
        plt.scatter(com.rm[:,com.ny+1,0], com.zm[:,com.ny+1,0], c=var[:,com.ny+1], cmap=cmap)
        if 'dnbot' in com.geometry[0].decode('UTF-8'):
            plt.scatter(com.rm[bbb.ixmp-1,:,0], com.zm[bbb.ixmp-1,:,0], c=var[bbb.ixmp-1,:], cmap=cmap)
            plt.scatter(com.rm[bbb.ixmp,:,0], com.zm[bbb.ixmp,:,0], c=var[bbb.ixmp,:], cmap=cmap)

    ax.grid(False)
    plt.title(title)
    if rzlabels:
        plt.xlabel(r'$R$ [m]')
        plt.ylabel(r'$Z$ [m]')
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    ax.add_collection(p)
    ax.autoscale_view()    

    if colorbar:
        cbar = plt.colorbar(p, label=label, extend=_extend, orientation=orientation)
    
    if iso:
        plt.axis('equal')  # regular aspect-ratio
        
    # Stats box
    if stats:
        text =  '      max %.2g\n' % np.max(analysis.nonGuard(var))
        text += '      min %.2g\n' % np.min(analysis.nonGuard(var))
        text += ' min(abs) %.2g\n' % np.min(np.abs(analysis.nonGuard(var)))
        text += '     mean %.2g\n' % np.mean(analysis.nonGuard(var))
        text += 'mean(abs) %.2g' % np.mean(np.abs(analysis.nonGuard(var)))
    if message:
        text = message
    if stats or message:        
        plt.text(0.01, 0.01, text, fontsize=4, color='black', family='monospace',
                 horizontalalignment='left', verticalalignment='bottom', 
                 transform=ax.transAxes)

    # Final plotting
    if show:
        plt.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], '--m', linewidth=1)
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.tight_layout()
        plt.show(block=False)

    return fig, ax if show else ax


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

def plot_flux_shahinul(flux_r, flux_z, title='', scale=50, width=0.003,
                       density=1.0, xlim=None, ylim=None, show=True, patches=None):
    """
    Plot flux direction arrows (no colorbar), using polygon centers from getPatches().

    Args:
        flux_r   : 2D array of flux in R direction
        flux_z   : 2D array of flux in Z direction
        title    : Plot title
        scale    : Arrow scaling factor (larger -> shorter arrows)
        width    : Arrow width
        density  : Fraction of arrows to keep (1 = all, <1 = subsample)
        xlim     : (xmin, xmax) optional limits
        ylim     : (ymin, ymax) optional limits
        show     : Display immediately
        patches  : optional cached patches (from getPatches())
    """

    plt.rcParams['axes.axisbelow'] = True

    if not patches:
        patches = getPatches()

    # compute centroids of polygons
    centroids_r = []
    centroids_z = []
    for poly in patches:
        arr = np.array(poly)
        r_mean = np.mean(arr[:, 0])
        z_mean = np.mean(arr[:, 1])
        centroids_r.append(r_mean)
        centroids_z.append(z_mean)

    centroids_r = np.array(centroids_r)
    centroids_z = np.array(centroids_z)

    # flatten flux arrays to match polygon order
    fr = flux_r.T.flatten()
    fz = flux_z.T.flatten()

    # Optionally subsample for clarity
    if density < 1.0:
        step = int(1.0 / density)
        centroids_r = centroids_r[::step]
        centroids_z = centroids_z[::step]
        fr = fr[::step]
        fz = fz[::step]

    # Create figure/axes
    if show:
        fig, ax = plt.subplots(1, figsize=(4.8, 6.4))
    else:
        ax = plt.gca()
        fig = ax.figure

    # Plot quiver (black arrows, clean for papers)
    ax.quiver(centroids_r, centroids_z, fr, fz,
              angles='xy', scale=scale, width=width,
              headwidth=3, headlength=4, headaxislength=3.5,
              color='k')

    ax.set_title(title)
    ax.set_xlabel(r"$R$ [m]")
    ax.set_ylabel(r"$Z$ [m]")
    ax.set_aspect('equal')
    ax.grid(False)

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    plt.tight_layout()
    if show:
        plt.show(block=False)

    return fig, ax

    
    
def plotDiffs(h5file):
    '''
    Plot differences between current variables and old ones.
    '''
    with h5py.File(h5file, 'r') as hf:
        hfb = hf.get('bbb')
        teold = np.array(hfb.get('tes'))
        tiold = np.array(hfb.get('tis'))
        niold = np.array(hfb.get('nis'))
    plt.figure(figsize=(14,5))
    plt.subplot(141)
    plotvar((bbb.ti-tiold)/tiold,sym=True,show=False,title=r'$(\Delta T_i)/T_i$')
    plt.subplot(142)
    plotvar((bbb.te-teold)/teold,sym=True,show=False,title=r'$(\Delta T_e)/T_e$')
    plt.subplot(143)
    plotvar((bbb.ni[:,:,0]-niold[:,:,0])/niold[:,:,0],sym=True,show=False,title=r'$(\Delta n_i)/n_i$')
    plt.subplot(144)
    plotvar((bbb.ni[:,:,1]-niold[:,:,1])/niold[:,:,1],sym=True,show=False,title=r'$(\Delta n_n)/n_n$')
    plt.tight_layout()
    plt.show()
    
    
def plotSources():
    kw = {'sym': True, 'show': False}
    rows = 2
    cols = 4
    plt.figure(figsize=(14,5))
    plt.subplot(rows, cols, 1)
    plotvar(bbb.fnix[:,:,0]/com.vol,title=r'fnix/vol', **kw)
    plt.subplot(rows, cols, 2)
    plotvar(bbb.fngx[:,:,0]/com.vol,title=r'fngx/vol', **kw)
    plt.subplot(rows, cols, 3)
    plotvar(bbb.fniy[:,:,0]/com.vol,title=r'fniy/vol', **kw)
    plt.subplot(rows, cols, 4)
    plotvar(bbb.fngy[:,:,0]/com.vol,title=r'fngy/vol', **kw)
    plt.subplot(rows, cols, 5)
    plotvar(bbb.psor[:,:,0]/com.vol,title=r'psor/vol', **kw)
    plt.subplot(rows, cols, 6)
    plotvar(bbb.psorrg[:,:,0]/com.vol,title=r'psorrg/vol', **kw)
    plt.subplot(rows, cols, 7)
    plotvar((bbb.psor[:,:,0]-bbb.psorrg[:,:,0]-bbb.fnix[:,:,0]-bbb.fniy[:,:,0])/com.vol,title=r'dni/dt', **kw)
    plt.subplot(rows, cols, 8)
    plotvar((-bbb.psor[:,:,0]+bbb.psorrg[:,:,0]-bbb.fngx[:,:,0]-bbb.fngy[:,:,0])/com.vol,title=r'dng/dt', **kw)
    plt.tight_layout()
    plt.show()
    
    
def readmesh(meshfile):
    """Return rm and zm of meshfile.
    
    Args:
        meshfile: (str) name of mesh file
        
    Source: https://github.com/LLNL/UEDGE/blob/master/pyscripts/uereadgrid.py
    """
    fh = open(meshfile, 'r')

    lns = fh.readlines()

    # Read the header information including metadata and grid shape
    ln1 = lns.pop(0).split()

    xxr = ln1[0]
    yyr = ln1[1]

    xsp1 = ln1[2]
    xsp2 = ln1[3]
    ysp = ln1[4]

    lns.pop(0)

    # Reshape the grid data to be linear
    data = np.zeros( (lns.__len__() * 3), np.float )
    print(data.shape)
    for i in range(0, lns.__len__()-1):
        ll = lns[i].split()
        print(ll)
        data[3*i  ] = float( ll[0].replace('D','E') )
        data[3*i+1] = float( ll[1].replace('D','E') )
        data[3*i+2] = float( ll[2].replace('D','E') )

    rml = 0
    rmh = (xxr+2) * (yyr+2) * 5
    rm = data[rml:rmh].reshape( (xxr+2, yyr+2, 5) )

    zml = rmh
    zmh = rmh + (xxr+2) * (yyr+2) * 5
    zm = data[zml:zmh].reshape( (xxr+2, yyr+2, 5) )
    return rm, zm


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plotmesh(meshfile=None, iso=True, xlim=None, ylim=None, wPlates=True, show=True, color_idiv='black', color_odiv='green', color_mesh='gray', linewidth=0.2, fig=None, showBad=False, outlineOnly=False, zorder=1):
    # Load mesh
    if meshfile:
        # Add your mesh loading code here
        pass
    else:    
        rm = com.rm
        zm = com.zm

    if fig:
        ax = fig.gca()
    else:
        fig, ax = plt.subplots(1, figsize=(4.8, 6.4))

    if outlineOnly:
        lines = []
        lines.extend([list(zip(rm[0,iy,[1,2,4,3,1]],zm[0,iy,[1,2,4,3,1]])) for iy in np.arange(0,com.ny+2)])
        lines.extend([list(zip(rm[com.nx+1,iy,[1,2,4,3,1]],zm[com.nx+1,iy,[1,2,4,3,1]])) for iy in np.arange(0,com.ny+2)])
        lines.extend([list(zip(rm[ix,0,[1,2,4,3,1]],zm[ix,0,[1,2,4,3,1]])) for ix in np.arange(0,com.nx+2)])
        lines.extend([list(zip(rm[ix,com.ny+1,[1,2,4,3,1]],zm[ix,com.ny+1,[1,2,4,3,1]])) for ix in np.arange(0,com.nx+2)])
        lc = LineCollection(lines, linewidths=linewidth, color=color_mesh, zorder=1)
        ax.add_collection(lc)
        ax.autoscale()
    else:
        lines = [list(zip(rm[ix,iy,[1,2,4,3,1]],zm[ix,iy,[1,2,4,3,1]])) for ix in np.arange(0,com.nx+2) for iy in np.arange(0,com.ny+2)]
        lc = LineCollection(lines, linewidths=linewidth, color=color_mesh, zorder=zorder)
        ax.add_collection(lc)
        ax.autoscale()

    if showBad:
        rover = []
        zover = []
        for ix, iy in analysis.overlappingCells():
            rover.append(rm[ix,iy,0])
            zover.append(zm[ix,iy,0])
        if rover:
            ax.scatter(rover, zover, c='orange', marker='o', label='Overlapping (%d)' % len(rover), zorder=2)
        rbad = []
        zbad = []
        for ix, iy in analysis.badCells():
            rbad.append(rm[ix,iy,0])
            zbad.append(zm[ix,iy,0])
        if rbad:
            ax.scatter(rbad, zbad, c='red', marker='x', label='Invalid polygon (%d)' % len(rbad), zorder=3)
        if rbad or rover:
            ax.legend()

    if iso:
        ax.axis('equal')

    if wPlates:
        O_div = np.loadtxt('/mnt/c/UEDGE_run_Shahinul/UEDGE_utils/UEDGE_utils/plate2.txt', delimiter=',') 
        In_div = np.loadtxt('/mnt/c/UEDGE_run_Shahinul/UEDGE_utils/UEDGE_utils/plate1.txt', delimiter=',')
        z = O_div[:, 0]
        r = O_div[:, 1]
        ax.plot(z, r, linestyle='-', color=color_odiv, linewidth=2)
        plt.text(com.rm[com.nx, com.ny + 1, 0]*1.1, -1.65, 'O-div (Li)', color=color_odiv, fontsize=12, fontweight='bold')

        z = In_div[:, 0]
        r = In_div[:, 1]
        ax.plot(z, r, linestyle='-', color=color_idiv, linewidth=2)
        plt.text(com.rm[0, com.ny + 1, 0]*0.5, -1.6, 'I-div (Li)', color=color_idiv, fontsize=12, fontweight='bold', rotation='vertical')

    ax.set_xlabel('R [m]', fontsize=16)
    ax.set_ylabel('Z [m]', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], '--m', linewidth=1)
    ax.plot(com.rm[:, com.ny + 1, 2], com.zm[:, com.ny + 1, 2], '-b', linewidth=1)
    ax.text(com.rm[com.nx, com.ny + 1, 0]*1.6, -1.0, 'O-wall', color='blue', fontsize=12, fontweight='bold')
    plt.text(com.rm[bbb.ixmp, com.ny + 1, 0], 0, 'OMP', fontsize=12, fontweight='bold')

    ax.grid(True)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if show:
        plt.tight_layout()
        plt.savefig('mesh.png', dpi=300)
        plt.show()  
     



def showCell(ix, iy):
    '''
    Show the location of cell ix, iy in the mesh.
    '''
    plotmesh(show=False)
    corners = (1,3,4,2,1)
    plt.plot(com.rm[ix, iy, corners], com.zm[ix, iy, corners], c='red') 
    plt.show()
    
    
def plotAreas():
    '''
    Calculate signed area of all cells to make sure corners are in same order.
    Differences in signed area might indicate that some cells are flipped.
    Positive signed area = counterclockwise.
    '''
    def PolygonArea(corners):
        """
        https://stackoverflow.com/a/24468019
        """
        n = len(corners) # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = area / 2.0
        return area
    
    areas = np.zeros((com.nx+2, com.ny+2))
    for ix in range(0, com.nx+2):
        for iy in range(0, com.ny+2):
            corners = (1,3,4,2)
            rcenter = com.rm[ix, iy, 0]
            zcenter = com.zm[ix, iy, 0]
            # Subtract rcenter and zcenter to avoid numerical precision issues
            vs = list(zip(com.rm[ix, iy, corners]-rcenter, com.zm[ix, iy, corners]-zcenter))
            areas[ix, iy] = PolygonArea(vs)
            if areas[ix, iy] > 0:
                print(ix, iy, areas[ix, iy])
    
    print('min', np.min(areas))
    print('max', np.max(areas))
    plotvar(areas, cmap=plt.cm.viridis, label='Area [m^2]')
    
    
def plotCellRotated(ix, iy, edge=1):
    '''
    Plot specified cell to determine if it is healthy and not twisted.
    Cell is rotated to make it easier to see. Increment "edge" variable in case
    it's still difficult to see.
    '''
    corners = (1,3,4,2)
    rcenter = com.rm[ix, iy, 0]
    zcenter = com.zm[ix, iy, 0]
    rc = com.rm[ix,iy,corners]-rcenter
    zc = com.zm[ix,iy,corners]-zcenter
    theta = -np.arctan2(zc[1+edge]-zc[0+edge], rc[1+edge]-rc[0+edge])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    cRot = R.dot(np.array([rc, zc]))
    rc = cRot[0,:]
    zc = cRot[1,:]
    plt.plot(rc, zc)
    plt.scatter(rc[0], zc[0])
    plt.show()
    
    
def show_qpar():
    fig, ax = plt.subplots(1)

    #-parallel heat flux
    bbb.fetx=bbb.feex+bbb.feix

    #-radial profile of qpar below entrance to the outer leg
    qpar1=(bbb.fetx[com.ixpt2[0]+1,com.iysptrx:]/com.sx[com.ixpt2[0]+1,com.iysptrx:])/com.rr[com.ixpt2[0]+1,com.iysptrx:]

    #-radial profile of qpar below entrance to the inner leg
    qpar2=(bbb.fetx[com.ixpt1[0],com.iysptrx:]/com.sx[com.ixpt1[0],com.iysptrx:])/com.rr[com.ixpt1[0],com.iysptrx:]

    ###fig1 = plt.figure()
    plt.plot(com.yyc[com.iysptrx:], qpar1)
    plt.plot(com.yyc[com.iysptrx:], qpar2, linestyle="dashed")

    plt.xlabel('R-Rsep [m]')
    plt.ylabel('qpar [W/m^2]')
    fig.suptitle('qpar at inner & outer (dash) divertor entrance')
    plt.grid(True)

    plt.show()


def plotr(v, ix=1, 
          title="UEDGE data", 
          xtitle="R-Rsep [m]", 
          ytitle="", 
          linestyle="solid",
          overplot=False):


    if (overplot == False):
        ###print("Overplot=False")
        fig,ax = plt.subplots(1)
        fig.suptitle(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
 #   else:
 #       print("Overplot=True")


    plt.plot(com.yyc,v[ix,:], linestyle=linestyle)
    plt.grid(True)

    plt.show()


def showIndices():
    '''
    Plot grid and overlay cell indices as text.
    '''
    fig, ax = plt.subplots(1)

    plt.axes().set_aspect('equal', 'datalim')


    for iy in np.arange(0,com.ny+2):
        for ix in np.arange(0,com.nx+2):
            plt.plot(com.rm[ix,iy,[1,2,4,3,1]],
                     com.zm[ix,iy,[1,2,4,3,1]], 
                     color="b", linewidth=0.5)
            plt.text(com.rm[ix, iy, 0], com.zm[ix, iy, 0], '%d,%d' % (ix, iy), fontsize=8)


    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    fig.suptitle('UEDGE mesh')
    plt.grid(True)

    plt.show()


rlabel = r'$R_{omp}-R_{sep}$ [mm]'
c0 = 'C0'
c1 = 'tomato'
c2 = 'black'
    
    
def allSame(arr):
    """
    Return true if every element in the numpy array has the same value.
    """
    arrf = arr.flatten()
    return np.all(arrf[0] == arrf)
    
    
def getConfigText():
    # Label
    txt = r'$\bf{Run\ label}$ ' + bbb.label[0].decode('utf-8')
    # Path
    path = os.getcwd()
    spath = path.split('UEDGE_private/')
    if len(spath) > 1:
        path = spath[-1]
    txt += '\n' + r'$\bf{Path}$ ' + path
    # Date created
    date = datetime.datetime.now().strftime('%I:%M %p %a %d %b %Y')
    txt += '\n' + r'$\bf{Plots\ created}$ ' + date
    # UEDGE version
    txt += '\n' + r'$\bf{UEDGE\ version}$ ' + str(uedgeVersion)
    
    # Grid 
    numBad = len(analysis.badCells())
    txt += '\n\n' + r'$\bf{Grid}$ nx = %d, ny = %d, %d cells are invalid polygons' % (com.nx, com.ny, numBad)
    if numBad > 0:
        txt += r' $\bf{(!!!)}$'
    # Core ni
    txt += '\n' + r'$\bf{Core\ n_i}$ '
    d = {0: 'set flux to curcore/sy locally in ix',
         1: r'fixed uniform %.3g m$^{-3}$' % bbb.ncore[0],
         2: 'set flux & ni over range',
         3: 'set icur = curcore-recycc*fngy, const ni',
         4: 'use impur. source terms (impur only)',
         5: 'set d(ni)/dy = -ni/lynicore at midp & ni constant poloidally'}
    if bbb.isnicore[0] in d:
        txt += d[bbb.isnicore[0]]
    # Core ng
    txt += '\n' + r'$\bf{Core\ n_n}$ '
    d = {0: 'set loc flux = -(1-albedoc)*ng*vtg/4',
         1: r'fixed uniform %.3g /m$^3$' % bbb.ngcore[0],
         2: 'invalid option',
         3: 'extrapolation, but limited'}
    if bbb.isngcore[0] in d:
        txt += d[bbb.isngcore[0]]
    else:
        txt += 'set zero derivative'
    # Core Te,Ti or Pe,Pi
    txt += '\n' + r'$\bf{Core\ T_e,T_i\ or\ P_e,P_i}$ '
    if bbb.iflcore == 0:
        txt += r'fixed $T_e$ = %.3g eV, $T_i$ = %.3g eV' % (bbb.tcoree, bbb.tcorei)
    elif bbb.iflcore == 1:
        txt += r'fixed $P_e$ = %.3g MW, $P_i$ = %.3g MW' % (bbb.pcoree/1e6, bbb.pcorei/1e6)
    # Core ion vparallel
    txt += '\n' + r'$\bf{Core\ ion\ v_\parallel\ (up)}$ '
    d = {0: 'up = upcore at core boundary',
         1: 'd(up)/dy = 0 at core boundary',
         2: 'd^2(up)/dy^2 = 0',
         3: 'fmiy = 0',
         4: 'tor. ang mom flux = lzflux & n*up/R=const',
         5: 'ave tor vel = utorave & n*up/R=const'}
    if bbb.isupcore[0] in d:
        txt += d[bbb.isupcore[0]]
    # D,chi if constant
    txt += '\n' + r'$\bf{Uniform\ coeffs}$ $D$ = %.3g m$^2/$s, $\chi_e$ = %.3g m$^2/$s, $\chi_i$ = %.3g m$^2/$s' % (bbb.difni[0], bbb.kye, bbb.kyi)
    # CF wall Te
    txt += '\n' + r'$\bf{CF\ wall\ T_e}$ '
    if allSame(bbb.tewallo):
        tewallo = 'fixed %.3g eV' % bbb.tewallo[0]
    else: 
        tewallo = 'fixed to 1D spatially varying profile (bbb.tewallo)'
    d = {0: 'zero energy flux',
         1: tewallo,
         2: 'extrapolated',
         3: r'$L_{Te}$ = %.3g m' % bbb.lyte[1],
         4: 'feey = bceew*fniy*te'}
    if bbb.istewc in d:
        txt += d[bbb.istewc]
    # PF wall Te
    txt += '\n' + r'$\bf{PF\ wall\ T_e}$ '
    if allSame(bbb.tewalli):
        tewalli = 'fixed %.3g eV' % bbb.tewalli[0]
    else: 
        tewalli = 'fixed to 1D spatially varying profile (bbb.tewalli)'
    d = {0: 'zero energy flux',
         1: tewalli,
         2: 'extrapolated',
         3: r'$L_{Te}$ = %.3g m' % bbb.lyte[0],
         4: 'feey = bceew*fniy*te'}
    if bbb.istepfc in d:
        txt += d[bbb.istepfc]
    # CF wall Ti
    txt += '\n' + r'$\bf{CF\ wall\ T_i}$ '
    if allSame(bbb.tiwallo):
        tiwallo = 'fixed %.3g eV' % bbb.tiwallo[0]
    else:
        tiwallo = 'fixed to 1D spatially varying profile (bbb.tiwallo)'
    d = {0: 'zero energy flux',
         1: tiwallo,
         2: 'extrapolated',
         3: r'$L_{Ti}$ = %.3g m' % bbb.lyti[1],
         4: 'feiy = bceiw*fniy*ti'}
    if bbb.istiwc in d:
        txt += d[bbb.istiwc]
    # PF wall Ti
    txt += '\n' + r'$\bf{PF\ wall\ T_i}$ '
    if allSame(bbb.tiwalli):
        tiwalli = 'fixed %.3g eV' % bbb.tiwalli[0]
    else:
        tiwalli = 'fixed to 1D spatially varying profile (bbb.tiwalli)'
    d = {0: 'zero energy flux',
         1: tiwalli,
         2: 'extrapolated',
         3: r'$L_{Ti}$ = %.3g m' % bbb.lyti[0],
         4: 'feiy = bceiw*fniy*ti'}
    if bbb.istipfc in d:
        txt += d[bbb.istipfc]
    # CF wall ni
    txt += '\n' + r'$\bf{CF\ wall\ n_i}$ '
    if allSame(bbb.nwallo):
        nwallo = r'fixed %.3g m$^{-3}$' % bbb.nwallo[0]
    else:
        nwallo = 'fixed to 1D spatially varying profile (bbb.nwallo)'
    z = {0: 'dn/dy = 0', 1: 'fniy = 0'}
    d = {0: z[bbb.ifluxni],
         1: nwallo,
         2: 'extrapolated',
         3: r'$L_{ni}$ = %.3g m, $n_{wall\ min}$ = %.3g m$^{-3}$' % (bbb.lyni[1], bbb.nwomin[0])}
    if bbb.isnwcono[0] in d:
        txt += d[bbb.isnwcono[0]]
    # PF wall ni
    txt += '\n' + r'$\bf{PF\ wall\ n_i}$ '
    if allSame(bbb.nwalli):
        nwalli = r'fixed %.3g m$^{-3}$' % bbb.nwalli[0]
    else:
        nwalli = 'fixed to 1D spatially varying profile (bbb.nwalli)'
    z = {0: 'dn/dy = 0', 1: 'fniy = 0'}
    d = {0: z[bbb.ifluxni],
         1: nwalli,
         2: 'extrapolated',
         3: r'$L_{ni}$ = %.3g m, $n_{wall\ min}$ = %.3g m$^{-3}$' % (bbb.lyni[0], bbb.nwimin[0])}
    if bbb.isnwconi[0] in d:
        txt += d[bbb.isnwconi[0]]
    # Flux limits
    if bbb.flalfe == bbb.flalfi == 0.21 and bbb.flalfv == 1 and np.all(bbb.lgmax == 0.05) and np.all(bbb.lgtmax == 0.05):
        flim = 'on'
    elif bbb.flalfe == bbb.flalfi == 1e20 and bbb.flalfv == 1e10 and np.all(bbb.lgmax == 1e20) and np.all(bbb.lgtmax == 1e20):
        flim = 'off'
    else:
        flim = 'unknown'
    txt += '\n' + r'$\bf{Flux\ limits}$ %s' % flim
    # Plates H recycling coefficient
    txt += '\n' + r'$\bf{Recycling\ coefficient}$ %.5g (plates), %.5g (walls)' % (bbb.recycp[0], bbb.recycw[0])
    # Neutral model
    if bbb.isngon[0] == 1 and bbb.isupgon[0] == 0 and com.nhsp == 1:
        nmodel = 'diffusive neutrals'
    elif bbb.isngon[0] == 0 and bbb.isupgon[0] == 1 and com.nhsp == 2:
        nmodel = 'inertial neutrals'
    else:
        nmodel = 'unknown'
    txt += '\n' + r'$\bf{Neutral\ model}$ %s' % nmodel
    # Impurity
    if bbb.isimpon == 2:
        txt += '\n' + r'$\bf{Impurity\ Z}$ %i' % (api.atn)
    elif bbb.isimpon == 6:
        txt += '\n' + r'$\bf{Impurity\ Z}$ %i %s' % (bbb.znuclin[2], bbb.ziin[2:com.nzsp[0]+2])
    # Impurity model
    txt += '\n' + r'$\bf{Impurity\ model}$ '
    d = {0: 'no impurity',
         2: 'fixed-fraction model',
         3: 'average-impurity-ion model (disabled)',
         4: 'INEL multi-charge-state model (disabled)',
         5: "Hirshman's reduced-ion model",
         6: 'force-balance model or nusp_imp > 0; see also isofric for full-Z drag term',
         7: 'simultaneous fixed-fraction and multi-charge-state (isimpon=6) models'}
    if bbb.isimpon in d:
        txt += d[bbb.isimpon]
    # Impurity fraction
    if bbb.isimpon == 2:
        txt += '\n' + r'$\bf{Impurity\ fraction}$ '
        if allSame(bbb.afracs):
            txt += '%.3g (spatially uniform)' % bbb.afracs[0,0]
        else:
            txt += 'spatially varying (mean = %.3g, std = %.3g, min = %.3g, max = %.3g)' % (np.mean(bbb.afracs), np.std(bbb.afracs), np.min(bbb.afracs), np.max(bbb.afracs))
    # Potential equation
    txt += '\n' + r'$\bf{Potential\ equation}$ '
    d = {0: 'off',
         1: 'on, b0 = %.3g' % bbb.b0}
    if bbb.isphion in d:
        txt += d[bbb.isphion]
            
    # Converged
    if bbb.iterm == 1:
        converged = 'yes'
    else: 
        converged = 'NOOOOOOOOO' # just to catch viewer's attention
    txt += '\n\n' + r'$\bf{Converged}$ ' + converged + (', sim. time %.3g s' % bbb.dt_tot)
    # Field line angle
    flangs = analysis.fieldLineAngle()
    txt += '\n' + r'$\bf{Field\ line\ angle}$ %.3g$\degree$ inner target, %.3g$\degree$ outer target' % (flangs[0,com.iysptrx+1], flangs[com.nx+1,com.iysptrx+1])
    # Separatrix
    nisep = (bbb.ni[bbb.ixmp,com.iysptrx,0]+bbb.ni[bbb.ixmp,com.iysptrx+1,0])/2
    nnsep = (bbb.ng[bbb.ixmp,com.iysptrx,0]+bbb.ng[bbb.ixmp,com.iysptrx+1,0])/2
    tisep = (bbb.ti[bbb.ixmp,com.iysptrx]+bbb.ti[bbb.ixmp,com.iysptrx+1])/2/bbb.ev
    tesep = (bbb.te[bbb.ixmp,com.iysptrx]+bbb.te[bbb.ixmp,com.iysptrx+1])/2/bbb.ev
    txt += '\n' + r'$\bf{Separatrix}$ $n_i$ = %.2g m$^{-3}$, $n_n$ = %.2g m$^{-3}$, $T_i$ = %.3g eV, $T_e$ = %.3g eV' % (nisep, nnsep, tisep, tesep)
    # Corner neutral pressure
    txt += '\n' + r'$\bf{Outer\ PF\ corner\ p_n}$ %.3g Pa' % (bbb.ng[:,:,0]*bbb.ti)[com.nx,1]
    # Power sharing
    powcc = bbb.feey + bbb.feiy 
    ixilast = analysis.ixilast()
    powcci = np.sum(powcc[com.ixpt1[0]+1:ixilast+1,com.iysptrx])/1e6
    powcco = np.sum(powcc[ixilast+1:com.ixpt2[0]+1,com.iysptrx])/1e6
    txt += '\n' + r'$\bf{Power\ sharing}$ 1:%.2g, $P_{LCFS\ inboard}$ = %.2g MW, $P_{LCFS\ outboard}$ = %.2g MW' % (powcco/powcci, powcci, powcco)
    # Impurity densities if multi-species
    if bbb.isimpon == 6:
        txt += '\n' + r'$\bf{n_{imp}}$ ' + analysis.impStats()
    # Impurity radiation
    if bbb.isimpon != 0:
        irad = bbb.prad/1e6*com.vol
        iradXPoint = np.sum(irad[com.ixpt1[0]:com.ixpt1[0]+2,:])+np.sum(irad[com.ixpt2[0]:com.ixpt2[0]+2,:])
        iradInnerLeg = np.sum(irad[:com.ixpt1[0],:])
        iradOuterLeg = np.sum(irad[com.ixpt2[0]+2:,:])
        iradMainChamberSOL = np.sum(irad[com.ixpt1[0]+1:bbb.ixmp,com.iysptrx+1:])+np.sum(irad[bbb.ixmp:com.ixpt2[0]+1,com.iysptrx+1:])
        iradCore = np.sum(irad[com.ixpt1[0]+1:bbb.ixmp,:com.iysptrx+1])+np.sum(irad[bbb.ixmp:com.ixpt2[0]+1,:com.iysptrx+1])
        txt += '\n' + r'$\bf{P_{rad\ imp}}$ $P_{tot}$ = %.2g MW, $P_{xpt}$ = %.2g MW, $P_{ileg}$ = %.2g MW, $P_{oleg}$ = %.2g MW,' % (np.sum(irad), iradXPoint, iradInnerLeg, iradOuterLeg) + '\n' + r'             $P_{main\ chamber\ SOL}$ = %.2g MW, $P_{core}$ = %.2g MW' % (iradMainChamberSOL, iradCore)
    # Domain power balance
    pInnerTarget, pOuterTarget, pCFWall, pPFWallInner, pPFWallOuter, prad, irad = analysis.powerLostBreakdown()
    pLoss = analysis.powerLost()
    txt += '\n' + r'$\bf{Power\ balance}$ $P_{loss}$ = %.2g MW = $P_{core}$%+.2g%%' % (pLoss/1e6, 100*pLoss/(bbb.pcoree+bbb.pcorei)-100) + '\n' + r'              ($P_{IT}$ = %.2g MW, $P_{OT}$ = %.2g MW, $P_{CFW}$ = %.2g MW, $P_{PFW}$ = %.2g MW, $P_{H}$ = %.2g MW, $P_{I}$ = %.2g MW)' % (pInnerTarget/1e6, pOuterTarget/1e6, pCFWall/1e6, (pPFWallOuter+pPFWallInner)/1e6, prad/1e6, irad/1e6)
    # Density balance
    nbalAbs = np.sum(np.abs(analysis.nonGuard(analysis.gridParticleBalance())))/np.sum(analysis.nonGuard(analysis.gridParticleSumAbs()))
    txt += '\n' + r'$\bf{Density\ balance}$ $\Sigma_{xy}|\Sigma_s(\Delta n)_s^{xy}|\left/\Sigma_{xy}\Sigma_s|(\Delta n)_s^{xy}|\right.$ = %.2g%%' % (nbalAbs*100)
    # Angle factor
    angleDegs = 2
    fi = 1./com.rr[0,com.iysptrx+1]*np.sin(angleDegs*np.pi/180.)
    fo = 1./com.rr[com.nx,com.iysptrx+1]*np.sin(angleDegs*np.pi/180.)
    # txt += '\n' + r'$\bf{Tilted\ plate\ factor\ q_{2\degree} = Fq_{pol}}$ $F_{inboard}$ = %.2g, $F_{outboard}$ = %.2g' % (fi, fo)
    return txt
    
    
def plotTransportCoeffs(patches):
    plt.rcParams['font.size'] = 10
    yyc_mm = com.yyc*1000
    iximp = analysis.iximp()
    ixomp = bbb.ixmp
    kwargs = {}
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # D line plot
    if not np.alltrue(bbb.dif_use == 0):
        plt.subplot(4,3,7)
        plt.plot(yyc_mm, bbb.dif_use[:,:,0][ixomp], c=c0, label=r'$D_{omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.dif_use[:,:,0][iximp], c=c1, label=r'$D_{imp}$', **kwargs)
        plt.title(r"$D$ [m$^2$/s]")
        plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
        for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
        plt.yscale('log')
        plt.legend()
        plt.xlabel(rlabel)
        # D 2D image
        plt.subplot(4,3,10)
        plotvar(bbb.dif_use[:,:,0],  title=r"$D$ [m$^2$/s]", log=True, patches=patches, show=False, 
                rzlabels=False, stats=False)
    # vconv line plot
    if not np.alltrue(bbb.vy_use == 0):
        plt.subplot(4,3,8)
        plt.plot(yyc_mm, bbb.vy_use[ixomp,:,0], c=c0, label=r'$v_{conv\ omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.vy_use[iximp,:,0], c=c1, label=r'$v_{conv\ imp}$', **kwargs)
        plt.title(r"$v_{conv}$ [m/s]")
        plt.legend()
        plt.xlabel(rlabel)
        plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
        for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
        # vconv 2D image
        plt.subplot(4,3,11)
        plotvar(bbb.vy_use[:,:,0],  title=r"$v_{conv}$ [m/s]", log=False, patches=patches, show=False, 
                rzlabels=False, stats=False)
    # Chi line plot
    if not (np.alltrue(bbb.kye_use == 0) and np.alltrue(bbb.kyi_use == 0)):
        plt.subplot(4,3,9)
        plt.plot(yyc_mm, bbb.kye_use[ixomp], c=c0, lw=1, label=r'$\chi_{e\ omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.kyi_use[ixomp], c=c0, lw=2, label=r'$\chi_{i\ omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.kye_use[iximp], c=c1, lw=1, label=r'$\chi_{e\ imp}$', **kwargs)
        plt.plot(yyc_mm, bbb.kyi_use[iximp], c=c1, lw=2, label=r'$\chi_{i\ imp}$', **kwargs)
        plt.title(r'$\chi$ [m$^2$/s]')
        plt.xlabel(rlabel)
        if np.any(bbb.kye_use > 0) and np.any(bbb.kyi_use > 0):
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
        for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
        # Chi 2D image
        plt.subplot(4,3,12)
        plotvar(bbb.kye_use,  title=r"$\chi$ [m$^2$/s]", log=True, patches=patches, show=False, 
                rzlabels=False, stats=False)
        plt.rcParams['font.size'] = 12
    
    
def plot2Dvars(patches):
    kwargs = {'log': True, 'patches': patches, 'show': False, 'rzlabels': False}
    kwargslin = {'log': False, 'patches': patches, 'show': False, 'rzlabels': False}
    plt.subplot(331)
    plotvar(bbb.te/bbb.ev,  title=r"$T_e$ [eV]", **kwargs)
    plt.subplot(332)
    plotvar(bbb.ti/bbb.ev,  title=r"$T_i$ [eV]", **kwargs)
    plt.subplot(334)
    plotvar(bbb.ni[:,:,0],  title=r"$n_{i}$ [m$^{-3}$]", **kwargs)
    plt.subplot(335)
    plotvar(bbb.ng[:,:,0],  title=r"$n_n$ [m$^{-3}$]", **kwargs)
    plt.subplot(337)
    plotvar(bbb.up[:,:,0],  title=r"$u_{pi}$ [m/s]", sym=True, **kwargs)
    if bbb.up.shape[2] > 1:
        plt.subplot(338)
        plotvar(bbb.up[:,:,1],  title=r"$u_{pn}$ [m/s]", sym=True, **kwargs)
    if bbb.isphion == 1:
        plt.subplot(339)
        plotvar(bbb.phi,  title=r"$\phi$ [V]", sym=True, **kwargslin)
    if bbb.isimpon == 6:
        plt.subplot(336)
        plotvar(np.sum(bbb.ni[:,:,2:], axis=2), title=r"$n_{imp}$ [m$^{-3}$]", **kwargs)
    

def plotnTprofiles(plotV0, h5):
    plt.rcParams['font.size'] = 10
    # plt.subplots_adjust(hspace = .01)
    yyc_mm = com.yyc*1000
    ixomp = bbb.ixmp
    iximp = analysis.iximp()
    #df = sparc.getV0data()
    df = df[(df['rho [mm]'] > min(yyc_mm)) & (df['rho [mm]'] < max(yyc_mm))]
    niV0 = df[' ni [10^20 m^-3]']*1e20
    TiV0 = df[' Ti [keV]']*1000
    TeV0 = df[' Te [keV]']*1000
    rhoV0mm = df['rho [mm]']
    lineNiIn =  {'c': c1, 'ls': '-',  'lw': 2}
    lineNiOut = {'c': c0, 'ls': '-',  'lw': 2}
    lineNiV0 =  {'c': c2, 'ls': '-',  'lw': 2}
    lineNgIn =  {'c': c1, 'ls': '--', 'lw': 2}
    lineNgOut = {'c': c0, 'ls': '--', 'lw': 2}
    lineTiIn =  {'c': c1, 'ls': '-',  'lw': 2}
    lineTiOut = {'c': c0, 'ls': '-',  'lw': 2}
    lineJs1In = {'c': c1, 'ls': '-',  'lw': 2}
    lineJs3In = {'c': c1, 'ls': '-.',  'lw': 2}
    lineJs1Out = {'c': c0, 'ls': '-',  'lw': 2}
    lineJs2In = {'c': c1, 'ls': '-',  'lw': 1}
    lineJs2Out = {'c': c0, 'ls': '-',  'lw': 1}
    lineJs3Out = {'c': c0, 'ls': '-.',  'lw': 2}
    lineTiV0 =  {'c': c2, 'ls': '-',  'lw': 2}
    lineTeIn =  {'c': c1, 'ls': '-',  'lw': 1}
    lineTeOut = {'c': c0, 'ls': '-',  'lw': 1}
    lineTeV0 =  {'c': c2, 'ls': '-',  'lw': 1}
    #
    plt.subplot(521)
    # plt.title('Inner midplane')
    if plotV0:
        plt.plot(rhoV0mm, niV0, label=r'$n_i$ V0', **lineNiV0)
    plt.plot(yyc_mm, bbb.ni[:,:,0][iximp], label=r'$n_{i\,imp}$', **lineNiIn)
    # plt.plot(yyc_mm, bbb.ng[:,:,0][iximp], label=r'$n_{n\,imp}$', **lineNgIn)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.yscale('log')
    #
    plt.subplot(522)
    # plt.title('Outer midplane')
    plt.plot(yyc_mm, bbb.ni[:,:,0][ixomp], label=r'$n_{i\,omp}$', **lineNiOut)
    # plt.plot(yyc_mm, bbb.ng[:,:,0][ixomp], label=r'$n_{n\,omp}$', **lineNgOut)
    if h5:
        var = 'neomid'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            yerr = h5[var+'/value_err'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            #plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
            plt.errorbar(rho[mask]*1000, val[mask], c='skyblue', ms=2, fmt='o', yerr=yerr[mask], zorder=1, elinewidth=0.7)
        var = 'neomidfit'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            plt.plot(rho[mask]*1000, val[mask], c='k', zorder=100,ls='--')
    if plotV0:
        plt.plot(rhoV0mm, niV0, label=r'$n_i$ V0', **lineNiV0)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.legend()
    # ymax = np.max(bbb.ni[bbb.ixmp,:,0])
    # plt.ylim([-0.05*ymax,ymax*1.05])
    plt.yscale('log')
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(523)
    # plt.title('Inner midplane')
    if plotV0:
        plt.plot(rhoV0mm, TiV0, label=r'$T_i$ V0', **lineTiV0)
        plt.plot(rhoV0mm, TeV0, label=r'$T_e$ V0', **lineTeV0)
    plt.plot(yyc_mm, bbb.ti[iximp]/bbb.ev, label=r'$T_{i\,imp}$', **lineTiIn)
    plt.plot(yyc_mm, bbb.te[iximp]/bbb.ev, label=r'$T_{e\,imp}$', **lineTeIn)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(524)
    # plt.title('Outer midplane')
    if h5:
        var = 'teomid'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            yerr = h5[var+'/value_err'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            # plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
            plt.errorbar(rho[mask]*1000, val[mask], c='skyblue', ms=2, yerr=yerr[mask], fmt='o', zorder=1, elinewidth=0.7)
        var = 'teomidfit'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            plt.plot(rho[mask]*1000, val[mask], c='k', zorder=100, ls='--')
    if plotV0:
        plt.plot(rhoV0mm, TiV0, label=r'$T_i$ V0', **lineTiV0)
        plt.plot(rhoV0mm, TeV0, label=r'$T_e$ V0', **lineTeV0)
    plt.plot(yyc_mm, bbb.ti[ixomp]/bbb.ev, label=r'$T_{i\,omp}$', **lineTiOut)
    plt.plot(yyc_mm, bbb.te[ixomp]/bbb.ev, label=r'$T_{e\,omp}$', **lineTeOut)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    # ymax = np.max([np.max(bbb.ti[bbb.ixmp]),np.max(bbb.te[bbb.ixmp])])/bbb.ev
    # plt.ylim([-0.05*ymax,ymax*1.05])
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(525)
    # plt.title('Inner plate')
    plt.plot(yyc_mm, bbb.ni[:,:,0][0], label=r'$n_{i\,it}$', **lineNiIn)
    plt.plot(yyc_mm, bbb.ng[:,:,0][0], label=r'$n_{n\,it}$', **lineNgIn)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(526)
    # plt.title('Outer plate')
    if h5:
        var = 'neotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
    plt.plot(yyc_mm, bbb.ni[:,:,0][com.nx+1], label=r'$n_{i\,ot}$', **lineNiOut)
    plt.plot(yyc_mm, bbb.ng[:,:,0][com.nx+1], label=r'$n_{n\,ot}$', **lineNgOut)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(527)
    # plt.title('Inner plate')
    plt.plot(yyc_mm, bbb.ti[0]/bbb.ev, label=r'$T_{i\,it}$', **lineTiIn)
    plt.plot(yyc_mm, bbb.te[0]/bbb.ev, label=r'$T_{e\,it}$', **lineTeIn)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    # plt.yscale('log')
    #
    plt.subplot(528)
    # plt.title('Outer plate')
    if h5:
        var = 'teotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
    plt.plot(yyc_mm, bbb.ti[com.nx+1]/bbb.ev, label=r'$T_{i\,ot}$', **lineTiOut)
    plt.plot(yyc_mm, bbb.te[com.nx+1]/bbb.ev, label=r'$T_{e\,ot}$', **lineTeOut)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(529)
    jsat1 = bbb.qe*bbb.ni[:,:,0]*np.sqrt((bbb.zeff*bbb.te+bbb.ti)/bbb.mi[0])
    vpolce = -bbb.cf2ef*bbb.v2ce[:,:,0]*(1-com.rr**2)**.5
    vtorce = bbb.cf2ef*bbb.v2ce[:,:,0]*com.rr
    vpolcb = -bbb.cf2bf*bbb.v2cb[:,:,0]*(1-com.rr**2)**.5
    vtorcb = bbb.cf2bf*bbb.v2cb[:,:,0]*com.rr
    upol = bbb.up[:,:,0]*com.rr
    utor = bbb.up[:,:,0]*(1-com.rr**2)**.5
    vtot = ((vpolce+vpolcb+upol)**2+(vtorce+vtorcb+utor)**2)**.5
    #jsat2 = bbb.qe*bbb.ni[:,:,0]*vtot
    # jsat2 = bbb.qe*bbb.ni[:,:,0]*bbb.up[:,:,0]
    #jsat3 = bbb.qe*bbb.ni[:,:,0]*np.sqrt((bbb.zeff*bbb.te+3*bbb.ti)/bbb.mi[0])
    jscale = 1000
    # plt.title('Inner plate')
    plt.plot(yyc_mm, jsat1[0]/jscale, label=r'$j_{sat\,it}\ \gamma_i=1$', **lineJs1In)
    jsat2 = bbb.qe*bbb.ni[:,:,0]*bbb.up[:,:,0]
    # if not np.all(bbb.fqpsatlb==0):
    #     plt.plot(yyc_mm, bbb.fqpsatlb[:,0]/com.sx[0]/com.sxnp[0]/jscale, label=r'fqpsatlb', **lineJs2In)
    # plt.plot(yyc_mm, jsat1[1]/jscale, **lineJs3In)
    plt.ylabel(r'$j_{sat}$ [kA/m$^2$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(5,2,10)
    # plt.title('Outer plate')
    if h5:
        var = 'jsotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask]/jscale, c='skyblue', s=5)
    plt.plot(yyc_mm, jsat1[com.nx+1]/jscale, label=r'$j_{sat\,ot}\ \gamma_i=1$', **lineJs1Out)
    # if not np.all(bbb.fqpsatrb==0):
    #     fqpsatrb = bbb.qe*bbb.isfdiax*( bbb.ne[com.nx+1,:]*bbb.v2ce[com.nx,:,0]*bbb.rbfbt[com.nx+1,:]*com.sx[com.nx,:] + bbb.fdiaxrb[:,0] )
    #     fqpsatrb += bbb.qe*bbb.zi[0]*bbb.ni[com.nx+1,:,0]*bbb.up[com.nx,:,0]*com.sx[com.nx,:]*com.rrv[com.nx,:]  
    #     plt.plot(yyc_mm, fqpsatrb[:,0]/com.sx[com.nx+1]/com.sxnp[com.nx+1]/jscale, label=r'fqpsatrb', **lineJs2In)
    #plt.plot(yyc_mm, jsat1[com.nx]/jscale, **lineJs3In)
    plt.ylabel(r'$j_{sat}$ [kA/m$^2$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.rcParams['font.size'] = 12
    plt.tight_layout(h_pad=0.01)
    
    
def plotAlongLegs():
    xpto = com.xfs[com.ixpt2[0]]
    xpti = com.xfs[com.ixpt1[0]]
    leg = list(range(0,com.ixpt1[0]+1))[::-1]
    xf = [(xpti-com.xfs[ix])*100 for ix in leg]
    legc = leg[:-1]
    xc = [(xpti-com.xcs[ix])*100 for ix in legc]
    xlim = [xc[0],xf[-1]+1]
    plt.figure(figsize=(8.5,11))
    plt.subplot(421)
    plt.plot(xf[1:], [np.sum(-(bbb.feex+bbb.feix)[ix,1:com.ny+1])/1e6 for ix in leg][1:], label='$P_{conv+cond}$', lw=2, c='k')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xf[1:]: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW]')
    plt.title('$P_{conv+cond}$ along inner leg')
    plt.subplot(423)
    if np.mean(bbb.afracs) > 1e-20:
        plt.plot(xc, [np.sum((bbb.prad*com.vol)[ix,1:com.ny+1])/1e6 for ix in legc], label='$P_{rad\ imp}$', c='C2')
    plt.plot(xc, [np.sum((bbb.erliz)[ix,1:com.ny+1])/1e6 for ix in legc], label='$P_{rad\ ioniz}$', c='C3')
    plt.plot(xc, [np.sum((bbb.erlrc)[ix,1:com.ny+1])/1e6 for ix in legc], label='$P_{rad\ recomb}$', c='C4')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.legend()
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW/m$^3$]')
    plt.title('$P_{rad}$ along inner leg')
    plt.subplot(425)
    plt.plot(xc, [np.max(bbb.te[ix,1:com.ny+1])/bbb.ev for ix in legc], label='$T_e$', lw=1, c='C1')
    plt.plot(xc, [np.max(bbb.ti[ix,1:com.ny+1])/bbb.ev for ix in legc], label='$T_i$', lw=2, c='C1')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.legend()
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$T$ [eV]')
    plt.title('$T_{max}$ along inner leg')
    plt.subplot(427)
    plt.plot(xc, [np.mean(bbb.ni[ix,1:com.ny+1,0]) for ix in legc], label='$n_i$', lw=2, c='C0')
    plt.plot(xc, [np.mean(bbb.ng[ix,1:com.ny+1,0]) for ix in legc], label='$n_n$', c='C0', ls='--', lw=2)
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$n$ [m$^{-3}$]')
    plt.title('$n$ along inner leg')
    plt.legend()

    leg = range(com.ixpt2[0]+1,com.nx+1)
    xf = [(com.xfs[ix]-xpto)*100 for ix in leg]
    xc = [(com.xcs[ix]-xpto)*100 for ix in leg]
    xlim = [xc[0],xf[-1]+1]
    plt.subplot(422)
    plt.plot(xf, [np.sum((bbb.feex+bbb.feix)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{conv+cond}$', lw=2, c='k')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xf: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW]')
    plt.title('$P_{conv+cond}$ along outer leg')
    plt.subplot(424)
    if np.mean(bbb.afracs) > 1e-20:
        plt.plot(xc, [np.sum((bbb.prad)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{rad\ imp}$', c='C2')
    plt.plot(xc, [np.sum((bbb.erliz/com.vol)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{rad\ ioniz}$', c='C3')
    plt.plot(xc, [np.sum((bbb.erlrc/com.vol)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{rad\ recomb}$', c='C4')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW/m$^3$]')
    plt.title('$P_{rad}$ along outer leg')
    plt.subplot(426)
    plt.plot(xc, [np.max(bbb.te[ix,1:com.ny+1])/bbb.ev for ix in leg], label='$T_e$', lw=1, c='C1')
    plt.plot(xc, [np.max(bbb.ti[ix,1:com.ny+1])/bbb.ev for ix in leg], label='$T_i$', lw=2, c='C1')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$T$ [eV]')
    plt.title('$T_{max}$ along outer leg')
    plt.legend()
    plt.subplot(428)
    plt.plot(xc, [np.mean(bbb.ni[ix,1:com.ny+1,0]) for ix in leg], label='$n_i$', lw=2, c='C0')
    plt.plot(xc, [np.mean(bbb.ng[ix,1:com.ny+1,0]) for ix in leg], label='$n_n$', c='C0', ls='--', lw=2)
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$n$ [m$^{-3}$]')
    plt.title('$n$ along outer leg')
    plt.legend()
    
    
def plotPressures():
    yyc_mm = com.yyc*1000
    iximp = analysis.iximp()
    ixomp = bbb.ixmp
    # nT inboard calculation
    nT_imp = bbb.ni[:,:,0][iximp]*(bbb.te[iximp]+bbb.ti[iximp]) + bbb.ni[:,:,0][iximp]*bbb.mi[0]*bbb.up[:,:,0][iximp]**2
    nT_idiv = bbb.ni[:,:,0][1]*(bbb.te[1]+bbb.ti[1]) + bbb.ni[:,:,0][1]*bbb.mi[0]*bbb.up[:,:,0][1]**2
    # nT inboard plot
    plt.subplot(321)
    plt.title('Inboard thermal+ram pressure')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_imp[com.iysptrx+1:-1], ls='--', c=c1, label='Midplane')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_idiv[com.iysptrx+1:-1], c=c1, label='Divertor plate')
    plt.xlabel(rlabel)
    plt.ylabel(r'Pressure [Pa]')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm[com.iysptrx+1:]: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    # nT outboard calculation
    nT_omp = bbb.ni[:,:,0][ixomp]*(bbb.te[ixomp]+bbb.ti[ixomp]) + bbb.ni[:,:,0][ixomp]*bbb.mi[0]*bbb.up[:,:,0][ixomp]**2
    nT_odiv = bbb.ni[:,:,0][com.nx]*(bbb.te[com.nx]+bbb.ti[com.nx]) + bbb.ni[:,:,0][com.nx]*bbb.mi[0]*bbb.up[:,:,0][com.nx]**2
    # nT outboard plot
    plt.subplot(322)
    plt.title('Outboard thermal+ram pressure')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_omp[com.iysptrx+1:-1], ls='--', c=c0, label='Midplane')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_odiv[com.iysptrx+1:-1], c=c0, label='Divertor plate')
    plt.xlabel(rlabel)
    plt.ylabel(r'Pressure [Pa]')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm[com.iysptrx+1:]: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    
    
def plotqFits(h5):
    # qpar calculation
    ppar = analysis.Pparallel()
    rrf = analysis.getrrf()
    iy = com.iysptrx+1
    xq = com.yyc[iy:-1]
    ixo = com.ixpt2[0]
    ixi = com.ixpt1[0]
    #-radial profile of qpar at X-point outer
    qparo = ppar[ixo,iy:-1]/com.sx[ixo,iy:-1]/rrf[ixo,iy:-1]
    intqo = np.sum(ppar[ixo+1,iy:-1]) # integral along first set of edges that *enclose* the outer divertor
    #-radial profile of qpar at X-point inner
    qpari = -ppar[ixi,iy:-1]/com.sx[ixi,iy:-1]/rrf[ixi,iy:-1]
    intqi = np.sum(-ppar[ixi-1,iy:-1])
    # lamda_q fits
    expfun = lambda x, A, lamda_q_inv: A*np.exp(-x*lamda_q_inv) # needs to be in this form for curve_fit to work
    omax = np.argmax(qparo) # only fit stuff to right of max
    try:
        qofit, _ = curve_fit(expfun, xq[omax:], qparo[omax:], p0=[np.max(qparo),1000], bounds=(0, np.inf))
        lqo = 1000/qofit[1] # lamda_q in mm
        lqoGuess = lqo
    except Exception as e:
        print('q parallel outer fit failed:', e)
        qofit = None
        lqoGuess = 1.
    imax = np.argmax(qpari) # only fit stuff to right of max
    try:
        qifit, _ = curve_fit(expfun, xq[imax:], qpari[imax:], p0=[np.max(qpari),1000], bounds=(0, np.inf))
        lqi = 1000/qifit[1] # lamda_q in mm
        lqiGuess = lqi
    except Exception as e:
        print('q parallel inner fit failed:', e)
        qifit = None
        lqiGuess = 1.
    # qpar plotting
    #plt.subplot(211)
    plt.figure(figsize=(6, 4))
    #plt.title(r'$q_\parallel$ at divertor entrance ($P_{xpt\ in}:P_{xpt\ out}$ = 1:%.1f)' % (intqo/intqi))
    plt.title(r'$q_\parallel$ at divertor entrance')

    #plt.plot(xq*1000, qparo/1e6, c=c0, label=r'X-point to outer wall, $P_{xpt}$ = %.3g MW' % (intqo/1e6))
    plt.plot(xq*1000, qparo/1e6, '*', color = 'blue', markersize = 8, label=r'UEDGE', linewidth = 2)
    #plt.plot(xq*1000, qpari/1e6, c=c1, label=r'X-point to inner wall, $P_{xpt}$ = %.3g MW' % (intqi/1e6))
    # ylim = plt.gca().get_ylim()
    if np.any(qofit):
        plt.plot(xq[omax:]*1000, expfun(xq, *qofit)[omax:]/1e6, '-', color = 'blue', linewidth=2, 
                 label='Fitted: $\lambda_q$ = %.3f mm' % lqo)
    #if np.any(qifit):
     #   plt.plot(xq[imax:]*1000, expfun(xq, *qifit)[imax:]/1e9, c=c1, ls=':', 
      #           label='Inboard exp. fit: $\lambda_q$ = %.3f mm' % lqi)
    try:
        ylim=[np.min([np.min(qparo[qparo>0]), np.min(qpari[qpari>0])])/1e6,np.max([np.max(qparo[qparo>0]), np.max(qpari[qpari>0])])/1e6]
        plt.ylim(ylim)
    except Exception as e:
        print('qpar ylim error:', e)
    plt.xlim([-0.1, com.yyc[-1]*1000])
    plt.xlabel(rlabel, fontsize = 16)
    plt.ylabel(r'$q_\parallel$ [MW/m$^2$]', fontsize = 16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    #plt.yscale('log')
    plt.tight_layout()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    
    # qsurf calculation
    #-radial profile of qpar below entrance to the outer leg
    psurfo = analysis.PsurfOuter()
    qsurfo = psurfo[1:-1]/com.sxnp[com.nx,1:-1]
    intqo = np.sum(psurfo)
    #-radial profile of qpar below entrance to the inner leg
    psurfi = analysis.PsurfInner()
    qsurfi = psurfi[1:-1]/com.sxnp[0,1:-1]
    intqi = np.sum(psurfi)
    # lamda_q fits
    def qEich(rho, q0, S, lqi, qbg, rho_0):
        rho = rho - rho_0
        # lqi is inverse lamda_q
        return q0/2*np.exp((S*lqi/2)**2-rho*lqi)*erfc(S*lqi/2-rho/S)+qbg
    bounds = ([0,0,0,0,com.yyc[0]], [np.inf,np.inf,np.inf,np.inf,com.yyc[-1]])
    oguess = (np.max(qsurfo)-np.min(qsurfo[qsurfo>0]), lqoGuess/1000/2, 1000/lqoGuess, np.min(qsurfo[qsurfo>0]), 0)
    try:
        qsofit, _ = curve_fit(qEich, com.yyc[1:-1], qsurfo, p0=oguess, bounds=bounds)
        lqeo, So = 1000/qsofit[2], qsofit[1]*1000 # lamda_q and S in mm
    except Exception as e:
        print('qsurf outer fit failed:', e)
        qsofit = None
    iguess = (np.max(qsurfi)-np.min(qsurfi[qsurfi>0]), lqiGuess/1000/2, 1000/lqiGuess, np.min(qsurfi[qsurfi>0]), 0)
    try:
        qsifit, _ = curve_fit(qEich, com.yyc[1:-1], qsurfi, p0=iguess, bounds=bounds)
        lqei, Si = 1000/qsifit[2], qsifit[1]*1000 # lamda_q and S in mm 
    except Exception as e:
        print('qsurf inner fit failed:', e)
        qsifit = None
    # qsurf plotting
    #plt.subplot(212)
   # plt.title(r'$q_{surf\ tot}$ ($P_{surf\ in}:P_{surf\ out}$ = 1:%.1f)' % (intqo/intqi))
    if h5:
        var = 'qotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask]/1e6, c='skyblue', s=5)
    plt.plot(com.yyc[1:-1]*1000, qsurfo/1e6, c=c0, label=r'Outboard plate, $P_{surf}$ = %.3g MW, $q_{peak}$ = %.3g MW/m$^2$' % (intqo/1e6, np.max(qsurfo)/1e6))
    plt.plot(com.yyc[1:-1]*1000, qsurfi/1e6, c=c1, label=r'Inboard plate, $P_{surf}$ = %.3g MW, $q_{peak}$ = %.3g MW/m$^2$' % (intqi/1e6, np.max(qsurfi)/1e6))
    #plt.yscale('log')
    #ylim = plt.gca().get_ylim()
    #if np.any(qsofit):
     #   plt.plot(com.yyc[1:-1]*1000, qEich(com.yyc[1:-1], *qsofit)/1e6, c=c0, ls=':',
     #            label=r'Outboard Eich fit: $\lambda_q$ = %.3f mm, $S$ = %.3g mm' % (lqeo, So))
    #if np.any(qsifit):
     #   plt.plot(com.yyc[1:-1]*1000, qEich(com.yyc[1:-1], *qsifit)/1e6, c=c1, ls=':',
      #           label=r'Inboard Eich fit: $\lambda_q$ = %.3f mm, $S$ = %.3g mm' % (lqei, Si))
    #plt.xlabel(rlabel)
    #plt.ylabel(r'$q_{surf}$ [MW/m$^2$]')
    #plt.legend(fontsize=8)
    #plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    #plt.ylim(ylim)
    #for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

def plotqFits_shahinul(sharex=0):
    fx = 1  # Flux expansion from upstream to target

    # --- q_parallel calculation ---
    ppar = analysis.Pparallel()
    rrf = analysis.getrrf()
    iy = com.iysptrx + 1
    ixo = com.ixpt2[0]
    ixi = com.ixpt1[0]

    xq = com.yyc[iy:-1] * 1000  # mm

    qparo = ppar[ixo, iy:-1] / com.sx[ixo, iy:-1] / rrf[ixo, iy:-1]
    qpari = -ppar[ixi, iy:-1] / com.sx[ixi, iy:-1] / rrf[ixi, iy:-1]

    # --- Target surface heat flux ---
    psurfo = analysis.PsurfOuter()
    qsurfo = psurfo[iy:-1] / com.sxnp[com.nx, iy:-1]
    x_tgt = com.yyc[iy:-1] * fx * 1000  # mm

    # --- Exponential Fit ---
    expfun = lambda x, A, lam_inv: A * np.exp(-x * lam_inv)
    omax = np.argmax(qparo)
    imax = np.argmax(qpari)

    qofit, _ = curve_fit(expfun, xq[omax:], qparo[omax:])
    qifit, _ = curve_fit(expfun, xq[imax:], qpari[imax:])
    lqo = 1 / qofit[1]  # mm
    lqi = 1 / qifit[1]  # mm

    # --- Eich Fit Function ---
    def qEich(x, q0, S, lam_inv, qbg, x0):
        x = x - x0
        return q0 / 2 * np.exp((S * lam_inv / 2)**2 - x * lam_inv) * erfc(S * lam_inv / 2 - x / S) + qbg * 1e-3

    # --- Fit Preparation ---
    mask_paro = np.isfinite(xq) & np.isfinite(qparo) & (qparo > 0)
    xq_fit = xq[mask_paro]
    qparo_fit = qparo[mask_paro]

    mask_surfo = np.isfinite(x_tgt) & np.isfinite(qsurfo) & (qsurfo > 0)
    x_tgt_fit = x_tgt[mask_surfo]
    qsurfo_fit = qsurfo[mask_surfo]

    def make_eich_guess(qdata, xdata):
        if len(qdata) == 0:
            return None
        q0_guess = np.max(qdata) - np.min(qdata)
        S_guess = max((xdata[-1] - xdata[0]) / 10, 1e-3)
        lam_inv_guess = max(1 / S_guess, 1e-3)
        qbg_guess = np.min(qdata)
        x0_guess = 0
        return (q0_guess, S_guess, lam_inv_guess, qbg_guess, x0_guess)

    # --- Eich Fit for q_parallel ---
    qparo_eich_fit, lqeo_paro = None, None
    qparo_guess = make_eich_guess(qparo_fit, xq_fit)
    if qparo_guess:
        try:
            qparo_eich_fit, _ = curve_fit(qEich, xq_fit, qparo_fit, p0=qparo_guess,
                                          bounds=([0, 0, 0, 0, -10], [np.inf]*5))
            lqeo_paro = 1 / qparo_eich_fit[2]
            So_paro = qparo_eich_fit[1]
            print(f"q_parallel Eich fit: ?q = {lqeo_paro:.2f} mm, S = {So_paro:.2f} mm")
        except Exception as e:
            print("qparo Eich fit failed:", e)

    # --- Eich Fit for q_target ---
    qsurfo_eich_fit, lqeo_surfo = None, None
    qsurfo_guess = make_eich_guess(qsurfo_fit, x_tgt_fit)
    if qsurfo_guess:
        try:
            qsurfo_eich_fit, _ = curve_fit(qEich, x_tgt_fit, qsurfo_fit, p0=qsurfo_guess,
                                           bounds=([0, 0, 0, 0, -10], [np.inf]*5))
            lqeo_surfo = 1 / qsurfo_eich_fit[2]
            So_surfo = qsurfo_eich_fit[1]
            print(f"q_target Eich fit: ?q = {lqeo_surfo:.2f} mm, S = {So_surfo:.2f} mm")
        except Exception as e:
            print("qsurfo Eich fit failed:", e)

    # --- Plotting ---
    if sharex == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
        fig.suptitle("Parallel and Target Heat Flux Fits", fontsize=16, y=1.02)
    else:
        fig, ax1 = plt.subplots(figsize=(5, 3.5))
        ax2 = None
        ax1.set_xlabel('r$_{omp}$ - r$_{sep}$ [mm]', fontsize=16)
        #plt.tight_layout()
        #plt.grid()

    # --- q_parallel Plot ---
    ax1.set_title(r'$q_\parallel$ at divertor entrance')
    ax1.plot(xq, qparo / 1e6, '*', color='blue', label='OX-point (UEDGE)', markersize=8)
    ax1.plot(xq, qpari / 1e6, 'h', color='red', label='IX-point (UEDGE)', markersize=8)
    ax1.plot(xq[omax:], expfun(xq[omax:], *qofit) / 1e6, color='blue', ls=':',
             label=f'Out exp fit: $\lambda_q$ = {lqo:.2f} mm')
    ax1.plot(xq[imax:], expfun(xq[imax:], *qifit) / 1e6, color='red', ls=':',
             label=f'In exp fit: $\lambda_q$ = {lqi:.2f} mm')
    if qparo_eich_fit is not None:
        ax1.plot(xq_fit, qEich(xq_fit, *qparo_eich_fit) / 1e6, '-', color='green',
                 label=f'Eich fit: $\lambda_q$ = {lqeo_paro:.2f} mm')
    ax1.set_ylabel(r'$q_\parallel$ [MW/m$^2$]', fontsize=16)
    ax1.set_xlim([0, 13])
    ax1.grid()
    ymax = np.max(qparo / 1e6)
    ax1.set_ylim([0, ymax*1])
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=12)

    # --- q_target Plot ---
    if sharex == 1 and ax2 is not None:
        ax2.plot(x_tgt, qsurfo / 1e6, '*', color='black', label='UEDGE', markersize=8)
        if qsurfo_eich_fit is not None:
            ax2.plot(x_tgt_fit, qEich(x_tgt_fit, *qsurfo_eich_fit) / 1e6, 'k--',
                     label=f'Eich fit: ?q = {lqeo_surfo:.2f} mm')
        ax2.set_xlabel('r$_{omp}$ - r$_{sep}$ [mm]', fontsize=14)
        ax2.set_ylabel(r'$q_{\perp}$ [MW/m$^2$]', fontsize=14)
        ax2.set_xlim([0, 13])
        ax2.grid(True, which='both', axis='y', color='#ddd')
        ax2.legend(fontsize=10)
        ax2.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"lambdaq.png", dpi=300)
    plt.show()

def plotqFits_all(h5):
    # qpar calculation
    ppar = analysis.Pparallel()
    rrf = analysis.getrrf()
    iy = com.iysptrx+1
    xq = com.yyc[iy:-1]
    ixo = com.ixpt2[0]
    ixi = com.ixpt1[0]
    #-radial profile of qpar at X-point outer
    qparo = ppar[ixo,iy:-1]/com.sx[ixo,iy:-1]/rrf[ixo,iy:-1]
    intqo = np.sum(ppar[ixo+1,iy:-1]) # integral along first set of edges that *enclose* the outer divertor
    #-radial profile of qpar at X-point inner
    qpari = -ppar[ixi,iy:-1]/com.sx[ixi,iy:-1]/rrf[ixi,iy:-1]
    intqi = np.sum(-ppar[ixi-1,iy:-1])
    # lamda_q fits
    expfun = lambda x, A, lamda_q_inv: A*np.exp(-x*lamda_q_inv) # needs to be in this form for curve_fit to work
    omax = np.argmax(qparo) # only fit stuff to right of max
    try:
        qofit, _ = curve_fit(expfun, xq[omax:], qparo[omax:], p0=[np.max(qparo),1000], bounds=(0, np.inf))
        lqo = 1000/qofit[1] # lamda_q in mm
        lqoGuess = lqo
    except Exception as e:
        print('q parallel outer fit failed:', e)
        qofit = None
        lqoGuess = 1.
    imax = np.argmax(qpari) # only fit stuff to right of max
    try:
        qifit, _ = curve_fit(expfun, xq[imax:], qpari[imax:], p0=[np.max(qpari),1000], bounds=(0, np.inf))
        lqi = 1000/qifit[1] # lamda_q in mm
        lqiGuess = lqi
    except Exception as e:
        print('q parallel inner fit failed:', e)
        qifit = None
        lqiGuess = 1.
    # qpar plotting
    #plt.subplot(312)
    plt.figure(figsize=(5, 3))
    plt.title(r'$q_\parallel$ at divertor entrance ($P_{xpt\ in}:P_{xpt\ out}$ = 1:%.1f)' % (intqo/intqi))
    plt.plot(xq*1000, qparo/1e9, c=c0, marker = "*", label=r'OX-point, $P_{xpt}$ = %.3g MW' % (intqo/1e6))
    plt.plot(xq*1000, qpari/1e9, c=c1,  marker = "h", label=r'IX-point, $P_{xpt}$ = %.3g MW' % (intqi/1e6))
    # ylim = plt.gca().get_ylim()
    if np.any(qofit):
        plt.plot(xq[omax:]*1000, expfun(xq, *qofit)[omax:]/1e9, c=c0, ls=':', 
                 label='Outboard exp. fit: $\lambda_q$ = %.3f mm' % lqo)
    if np.any(qifit):
        plt.plot(xq[imax:]*1000, expfun(xq, *qifit)[imax:]/1e9, c=c1, ls=':', 
                 label='Inboard exp. fit: $\lambda_q$ = %.3f mm' % lqi)
    try:
        ylim=[np.min([np.min(qparo[qparo>0]), np.min(qpari[qpari>0])])/1e9,np.max([np.max(qparo[qparo>0]), np.max(qpari[qpari>0])])/1e9]
        plt.ylim(ylim)
    except Exception as e:
        print('qpar ylim error:', e)
    plt.xlim([-0.1, com.yyc[-1]*1000])
    plt.xlabel(rlabel, fontsize= 16)
    plt.ylabel(r'$q_\parallel$ [GW/m$^2$]', fontsize= 16)
    plt.legend(fontsize=10)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig('lambda_q.png', dpi = 300)
    
    # qsurf calculation
    #-radial profile of qpar below entrance to the outer leg
    psurfo = analysis.PsurfOuter()
    qsurfo = psurfo[1:-1]/com.sxnp[com.nx,1:-1]
    intqo = np.sum(psurfo)
    #-radial profile of qpar below entrance to the inner leg
    psurfi = analysis.PsurfInner()
    qsurfi = psurfi[1:-1]/com.sxnp[0,1:-1]
    intqi = np.sum(psurfi)
    # lamda_q fits
    def qEich(rho, q0, S, lqi, qbg, rho_0):
        rho = rho - rho_0
        # lqi is inverse lamda_q
        return q0/2*np.exp((S*lqi/2)**2-rho*lqi)*erfc(S*lqi/2-rho/S)+qbg
    bounds = ([0,0,0,0,com.yyc[0]], [np.inf,np.inf,np.inf,np.inf,com.yyc[-1]])
    oguess = (np.max(qsurfo)-np.min(qsurfo[qsurfo>0]), lqoGuess/1000/2, 1000/lqoGuess, np.min(qsurfo[qsurfo>0]), 0)
    try:
        qsofit, _ = curve_fit(qEich, com.yyc[1:-1], qsurfo, p0=oguess, bounds=bounds)
        lqeo, So = 1000/qsofit[2], qsofit[1]*1000 # lamda_q and S in mm
    except Exception as e:
        print('qsurf outer fit failed:', e)
        qsofit = None
    iguess = (np.max(qsurfi)-np.min(qsurfi[qsurfi>0]), lqiGuess/1000/2, 1000/lqiGuess, np.min(qsurfi[qsurfi>0]), 0)
    try:
        qsifit, _ = curve_fit(qEich, com.yyc[1:-1], qsurfi, p0=iguess, bounds=bounds)
        lqei, Si = 1000/qsifit[2], qsifit[1]*1000 # lamda_q and S in mm 
    except Exception as e:
        print('qsurf inner fit failed:', e)
        qsifit = None
    # qsurf plotting
    #plt.subplot(313)
    plt.title(r'$q_{surf\ tot}$ ($P_{surf\ in}:P_{surf\ out}$ = 1:%.1f)' % (intqo/intqi))
    if h5:
        var = 'qotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask]/1e6, c='skyblue', s=5)
    plt.plot(com.yyc[1:-1]*1000, qsurfo/1e6, c=c0, label=r'Outboard plate, $P_{surf}$ = %.3g MW, $q_{peak}$ = %.3g MW/m$^2$' % (intqo/1e6, np.max(qsurfo)/1e6))
    plt.plot(com.yyc[1:-1]*1000, qsurfi/1e6, c=c1, label=r'Inboard plate, $P_{surf}$ = %.3g MW, $q_{peak}$ = %.3g MW/m$^2$' % (intqi/1e6, np.max(qsurfi)/1e6))
    plt.yscale('log')
    ylim = plt.gca().get_ylim()
    if np.any(qsofit):
        plt.plot(com.yyc[1:-1]*1000, qEich(com.yyc[1:-1], *qsofit)/1e6, c=c0, ls=':',
                 label=r'Outboard Eich fit: $\lambda_q$ = %.3f mm, $S$ = %.3g mm' % (lqeo, So))
    if np.any(qsifit):
        plt.plot(com.yyc[1:-1]*1000, qEich(com.yyc[1:-1], *qsifit)/1e6, c=c1, ls=':',
                 label=r'Inboard Eich fit: $\lambda_q$ = %.3f mm, $S$ = %.3g mm' % (lqei, Si))
    #plt.xlabel(rlabel)a
    plt.ylabel(r'$q_{surf}$ [MW/m$^2$]')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    plt.ylim(ylim)
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

def plotPowerBreakdown():
    yyc_mm = com.yyc[1:-1]*1000
    pwrx = bbb.feex+bbb.feix

    # Inner target qsurf breakdown
    plt.subplot(221)
    plt.title('Inner target')
    plateIndex = 0
    xsign = -1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt1[0]
    xtarget = 0
    pwrxtot = xsign*np.sum(pwrx[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*pwrx[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Conv.+cond. e+i+n (%.2g MW)' % pwrxtot)
    fnixtot = xsign*np.sum(bbb.fnix[xtarget,1:-1,0])*bbb.ebind*bbb.ev/1e6
    plt.plot(yyc_mm, xsign*bbb.fnix[xtarget,1:-1,0]*bbb.ebind*bbb.ev/com.sxnp[xtarget,1:-1]/1e6, label='Surface recomb. (%.2g MW)' % fnixtot)
    ketot = np.sum(xsign*analysis.PionParallelKE()[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*analysis.PionParallelKE()[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Ion kinetic energy (%.2g MW)' % ketot)
    plthtot = np.sum(bbb.pwr_plth[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_plth[1:-1,plateIndex]/1e6, ls='--', label='H photons (%.2g MW)' % plthtot)
    pltztot = np.sum(bbb.pwr_pltz[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_pltz[1:-1,plateIndex]/1e6, ls='--', label='Imp. photons (%.2g MW)' % pltztot)
    plt.xlabel(rlabel)
    plt.ylabel(r'$q_{surf}$ [MW/m$^2$]')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

    # Outer target qsurf breakdown
    plt.subplot(222)
    plt.title('Outer target')
    plateIndex = 1
    xsign = 1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt2[0]
    xlegEntrance = ixpt+1
    xtarget = com.nx
    xleg = slice(ixpt+1, xtarget+1)
    pwrxtot = xsign*np.sum(pwrx[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*pwrx[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Conv.+cond. e+i+n (%.2g MW)' % pwrxtot)
    fnixtot = xsign*np.sum(bbb.fnix[xtarget,1:-1,0])*bbb.ebind*bbb.ev/1e6
    plt.plot(yyc_mm, xsign*bbb.fnix[xtarget,1:-1,0]*bbb.ebind*bbb.ev/com.sxnp[xtarget,1:-1]/1e6, label='Surface recomb. (%.2g MW)' % fnixtot)
    ketot = np.sum(xsign*analysis.PionParallelKE()[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*analysis.PionParallelKE()[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Ion kinetic energy (%.2g MW)' % ketot)
    plthtot = np.sum(bbb.pwr_plth[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_plth[1:-1,plateIndex]/1e6, ls='--', label='H photons (%.2g MW)' % plthtot)
    pltztot = np.sum(bbb.pwr_pltz[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_pltz[1:-1,plateIndex]/1e6, ls='--', label='Imp. photons (%.2g MW)' % pltztot)
    plt.xlabel(rlabel)
    plt.ylabel(r'$q_{surf}$ [MW/m$^2$]')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)



def plotPowerBreakdown_shahinul():
    yylb = com.yylb[1:-1]
    yyrb = com.yyrb[1:-1]
    pwrx = bbb.feex + bbb.feix
    bbb.plateflux()
    bbb.pradpltwl()
    

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    #fig.suptitle('Target Surface Power Breakdown', fontsize=14)

    # --- Inner target (left subplot) ---
    ax = axs[0]
    ax.set_title('Inner target')
    plateIndex = 0
    xsign = -1
    xtarget = 0

    pwrxtot = xsign * np.sum(pwrx[xtarget, 1:-1]) / 1e6
    ax.plot(yylb, xsign * pwrx[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6, label=f'Conv.+cond. e+i+n ({pwrxtot:.2g} MW)')

    fnixtot = xsign * np.sum(bbb.fnix[xtarget, 1:-1, 0]) * bbb.ebind * bbb.ev / 1e6
    ax.plot(yylb, xsign * bbb.fnix[xtarget, 1:-1, 0] * bbb.ebind * bbb.ev / com.sxnp[xtarget, 1:-1] / 1e6,
            label=f'Surface recomb. ({fnixtot:.2g} MW)')

    ketot = np.sum(xsign * analysis.PionParallelKE()[xtarget, 1:-1]) / 1e6
    ax.plot(yylb, xsign * analysis.PionParallelKE()[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6,
            label=f'Ion kinetic energy ({ketot:.2g} MW)')

    plthtot = np.sum(bbb.pwr_plth[1:-1, plateIndex] * com.sxnp[xtarget, 1:-1]) / 1e6
    ax.plot(yylb, bbb.pwr_plth[1:-1, plateIndex] / 1e6, ls='--', label=f'H photons ({plthtot:.2g} MW)')

    pltztot = np.sum(bbb.pwr_pltz[1:-1, plateIndex] * com.sxnp[xtarget, 1:-1]) / 1e6
    ax.plot(yylb, bbb.pwr_pltz[1:-1, plateIndex] / 1e6, ls='--', label=f'Imp. photons ({pltztot:.2g} MW)')

    ax.set_xlabel(r'r$_{div}$ - r$_{sep}$ (m)', fontsize = 14)
    ax.set_ylabel(r'$q_{surf}$ [MW/m$^2$]', fontsize = 14)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', axis='y', color='#ddd')
    ax.set_axisbelow(True)
    for i in com.yyc * 1:
        ax.axvline(i, c='#ddd', lw=0.5, zorder=0)

    # --- Outer target (right subplot) ---
    ax = axs[1]
    ax.set_title('Outer target')
    plateIndex = 1
    xsign = 1
    xtarget = com.nx

    pwrxtot = xsign * np.sum(pwrx[xtarget, 1:-1]) / 1e6
    ax.plot(yyrb, xsign * pwrx[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6, label=f'Conv.+cond. e+i+n ({pwrxtot:.2g} MW)')

    fnixtot = xsign * np.sum(bbb.fnix[xtarget, 1:-1, 0]) * bbb.ebind * bbb.ev / 1e6
    ax.plot(yyrb, xsign * bbb.fnix[xtarget, 1:-1, 0] * bbb.ebind * bbb.ev / com.sxnp[xtarget, 1:-1] / 1e6,
            label=f'Surface recomb. ({fnixtot:.2g} MW)')

    ketot = np.sum(xsign * analysis.PionParallelKE()[xtarget, 1:-1]) / 1e6
    ax.plot(yyrb, xsign * analysis.PionParallelKE()[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6,
            label=f'Ion kinetic energy ({ketot:.2g} MW)')

    plthtot = np.sum(bbb.pwr_plth[1:-1, plateIndex] * com.sxnp[xtarget, 1:-1]) / 1e6
    ax.plot(yyrb, bbb.pwr_plth[1:-1, plateIndex] / 1e6, ls='--', label=f'H photons ({plthtot:.2g} MW)')

    pltztot = np.sum(bbb.pwr_pltz[1:-1, plateIndex] * com.sxnp[xtarget, 1:-1]) / 1e6
    ax.plot(yyrb, bbb.pwr_pltz[1:-1, plateIndex] / 1e6, ls='--', label=f'Imp. photons ({pltztot:.2g} MW)')

    ax.set_xlabel(r'r$_{div}$ - r$_{sep}$ (m)', fontsize = 14)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', axis='y', color='#ddd')
    ax.set_axisbelow(True)
    for i in com.yyc * 1:
        ax.axvline(i, c='#ddd', lw=0.5, zorder=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('q_surface.png', dpi = 300)
    plt.show()

def plotPowerBreakdown_shahinul_full():
    yylb = com.yylb[1:-1]
    yyrb = com.yyrb[1:-1]
    pwrx = bbb.feex + bbb.feix
    bbb.plateflux()
    bbb.pradpltwl()
    q_data = bbb.sdrrb + bbb.sdtrb
    q_data_odiv = q_data.reshape(-1)
    q_data_idiv = (bbb.sdrlb + bbb.sdtlb).reshape(-1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # --- Inner target (left subplot) ---
    ax = axs[0]
    ax.set_title('Inner target')
    plateIndex = 0
    xsign = -1
    xtarget = 0

    conv_cond = xsign * pwrx[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6
    surf_recomb = xsign * bbb.fnix[xtarget, 1:-1, 0] * bbb.ebind * bbb.ev / com.sxnp[xtarget, 1:-1] / 1e6
    ion_ke = xsign * analysis.PionParallelKE()[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6
    h_photons = bbb.pwr_plth[1:-1, plateIndex] / 1e6
    imp_photons = bbb.pwr_pltz[1:-1, plateIndex] / 1e6

    # Total at each point
    total_inner = conv_cond + surf_recomb + ion_ke + h_photons + imp_photons

    ax.plot(yylb, conv_cond, label='Conv.+cond. e+i+n')
    ax.plot(yylb, surf_recomb, label='Surface recomb.')
    ax.plot(yylb, ion_ke, label='Ion kinetic energy')
    ax.plot(yylb, h_photons, ls='--', label='H photons')
    ax.plot(yylb, imp_photons, ls='--', label='Imp. photons')

    # Plot total
    ax.plot(yylb, total_inner, 'k-', lw=2, label='Total (sum)')

    # Plot q_data_idiv
    ax.plot(yylb, q_data_idiv[1:-1]/1e6, 'm:', lw=2, label='q_data_idiv')

    ax.set_xlabel(r'r$_{div}$ - r$_{sep}$ (m)', fontsize=14)
    ax.set_ylabel(r'$q_{surf}$ [MW/m$^2$]', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', axis='y', color='#ddd')
    ax.set_axisbelow(True)
    for i in com.yyc * 1:
        ax.axvline(i, c='#ddd', lw=0.5, zorder=0)

    # --- Outer target (right subplot) ---
    ax = axs[1]
    ax.set_title('Outer target')
    plateIndex = 1
    xsign = 1
    xtarget = com.nx

    conv_cond = xsign * pwrx[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6
    surf_recomb = xsign * bbb.fnix[xtarget, 1:-1, 0] * bbb.ebind * bbb.ev / com.sxnp[xtarget, 1:-1] / 1e6
    ion_ke = xsign * analysis.PionParallelKE()[xtarget, 1:-1] / com.sxnp[xtarget, 1:-1] / 1e6
    h_photons = bbb.pwr_plth[1:-1, plateIndex] / 1e6
    imp_photons = bbb.pwr_pltz[1:-1, plateIndex] / 1e6

    total_outer = conv_cond + surf_recomb + ion_ke + h_photons + imp_photons

    ax.plot(yyrb, conv_cond, label='Conv.+cond. e+i+n')
    ax.plot(yyrb, surf_recomb, label='Surface recomb.')
    ax.plot(yyrb, ion_ke, label='Ion kinetic energy')
    ax.plot(yyrb, h_photons, ls='--', label='H photons')
    ax.plot(yyrb, imp_photons, ls='--', label='Imp. photons')

    # Plot total
    ax.plot(yyrb, total_outer, 'k-', lw=2, label='Total (sum)')

    # Plot q_data_odiv
    ax.plot(yyrb, q_data_odiv[1:-1]/1e6, 'c:', lw=2, label='q_data_odiv')

    ax.set_xlabel(r'r$_{div}$ - r$_{sep}$ (m)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', axis='y', color='#ddd')
    ax.set_axisbelow(True)
    for i in com.yyc * 1:
        ax.axvline(i, c='#ddd', lw=0.5, zorder=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('q_surface.png', dpi=300)
    plt.show()

def plotPowerSurface():
    pwrx = bbb.feex+bbb.feix
    pwry = bbb.feey+bbb.feiy
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad((1,0,0))
    fontsize = 8
    bdict = dict(boxstyle="round", alpha=0.5, color='lightgray')
    c = 'black'
    labelArgs = {}
    offset = .02

    # Inner target total qsurf
    ax = plt.subplot(223)
    labels = []
    plateIndex = 0
    xsign = -1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt1[0]
    xtarget = 0
    segments = []
    ptot = [] 
    pout = 0 # power lost in the volume of the divertor (with int prad dV rather than psurf dS)
    areas = []
    # Inner leg entrance
    pxs = []
    ix = ixpt-1
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        ptot.append(xsign*pwrx[ix,iy])
        areas.append(com.sxnp[ix,iy])
    pent = sum(pxs)
    v1, v2 = analysis.cellFaceVertices('E', ix, com.iysptrx)
    rcenter, zcenter = v1
    rcenter += offset
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='left', va='bottom', size=fontsize, bbox=bdict))
    # Inner leg common flux
    pys = []
    pbinds = []
    iy = com.ny
    for ix in range(1, ixpt):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(pwry[ix, iy])
        pbinds.append(bbb.fniy[ix, com.ny,0]*bbb.ebind*bbb.ev)
        ptot.append(pwry[ix, iy]
                    +bbb.fniy[ix, com.ny,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_wallh[ix]*com.sy[ix, iy]
                    +bbb.pwr_wallz[ix]*com.sy[ix, iy])
        areas.append(com.sy[ix, iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[1,iy,0]+com.rm[ixpt-1,iy,0])/2
    zcenter = (com.zm[1,iy,0]+com.zm[ixpt-1,iy,0])/2+offset
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='right', va='bottom', size=fontsize, bbox=bdict))
    # Inner leg private flux
    pys = []
    pbinds = []
    iy = 0
    for ix in range(1, ixpt):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(-pwry[ix,iy])
        pbinds.append(-bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(-pwry[ix,iy]
                    -bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_pfwallh[ix,0]*com.sy[ix,iy]
                    +bbb.pwr_pfwallz[ix,0]*com.sy[ix,iy])
        areas.append(com.sy[ix, iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[1,iy,0]+com.rm[ixpt-1,iy,0])/2
    zcenter = (com.zm[1,iy,0]+com.zm[ixpt-1,iy,0])/2-offset
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='left', va='top', size=fontsize, bbox=bdict))
    # Inner target
    pxs = []
    pbinds = []
    ix = 0
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        pbinds.append(xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(xsign*pwrx[ix,iy]
                    +xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_plth[iy,plateIndex]*com.sxnp[ix,iy]
                    +bbb.pwr_pltz[iy,plateIndex]*com.sxnp[ix,iy])
        areas.append(com.sxnp[ix, iy])
    pout += sum(pxs) + sum(pbinds)
    rcenter = com.rm[ix,com.iysptrx,0]-offset
    zcenter = com.zm[ix,com.iysptrx,0]
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6) + '\n' + r'$P_{\parallel bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='right', va='top', size=fontsize, bbox=bdict))
    # Total power
    segments = np.array(segments)
    ptot = np.array(ptot)
    areas = np.array(areas)
    ptot = ptot/areas
    summ = np.sum(ptot*areas)-pent
    # Total radiation
    prad = np.sum((bbb.erliz+bbb.erlrc)[1:ixpt,1:com.ny+1])
    if bbb.isimpon != 0:
        irad = np.sum((bbb.prad*com.vol)[1:ixpt,1:com.ny+1])
    else:
        irad = 0
    pout += prad + irad
    text = '\n'.join([r'$\int p_{H\ rad}\ dV=$%.2g MW' % (prad/1e6), 
                      r'$\int p_{imp\ rad}\ dV=$%.2g MW' % (irad/1e6),
                      r'$P_{\parallel\ in}=$%.2g MW' % (pent/1e6),
                      r'$P_{out}=$%.2g MW' % (pout/1e6)])
    plt.text(0.03, 0.03, text, fontsize=fontsize, c=c, bbox=bdict,
                     ha='left', va='bottom', 
                     transform=ax.transAxes)
    # Plot line collection
    norm = matplotlib.colors.LogNorm()
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(ptot/1e6)
    lc.set_linewidth(4)
    line = ax.add_collection(lc)
    plt.colorbar(line, ax=ax, orientation='horizontal', pad=0, label=r'$q_{surf\ tot}$ [MW/m$^2$]')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('gray')
    ax.grid(False)
    plt.axis('equal')
    plt.title('Inner leg')
    plt.xlabel(r'$R$ [m]')
    plt.ylabel(r'$Z$ [m]')
    # Scale view to include labels that might otherwise be cut off
    plt.draw()
    for label in labels:
        bbox = label.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        bbox_data = bbox.transformed(ax.transData.inverted()).padded(.05)
        ax.update_datalim(bbox_data.corners())
    ax.autoscale_view()

    # Outer target total qsurf
    ax = plt.subplot(224)
    labels = []
    plateIndex = 1
    xsign = 1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt2[0]
    xlegEntrance = ixpt+1
    xtarget = com.nx
    xleg = slice(ixpt+1, xtarget+1)
    segments = []
    ptot = []
    pout = 0 # power lost in the volume of the divertor (with int prad dV rather than psurf dS)
    areas = []
    # Outer leg entrance
    pxs = []
    ix = ixpt+1
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        ptot.append(xsign*pwrx[ix,iy])
        areas.append(com.sxnp[ix,iy])
    pent = sum(pxs)
    rcenter = (com.rm[ix,1,0]+com.rm[ix,com.ny,0])/2
    zcenter = (com.zm[ix,1,0]+com.zm[ix,com.ny,0])/2
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='center', va='bottom', size=fontsize, bbox=bdict))
    # Outer leg common flux
    pys = []
    pbinds = []
    iy = com.ny
    for ix in range(ixpt+2, com.nx+1):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(pwry[ix,iy])
        pbinds.append(bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(pwry[ix,iy]
                    +bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_wallh[ix]*com.sy[ix,iy]
                    +bbb.pwr_wallz[ix]*com.sy[ix,iy])
        areas.append(com.sy[ix,iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[ixpt+2,iy,0]+com.rm[com.nx,iy,0])/2+offset
    zcenter = (com.zm[ixpt+2,iy,0]+com.zm[com.nx,iy,0])/2
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='left', va='bottom', size=fontsize, bbox=bdict))
    # Outer leg private flux
    pys = []
    pbinds = []
    iy = 0
    for ix in range(ixpt+2, com.nx+1):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(-pwry[ix,iy])
        pbinds.append(-bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(-pwry[ix,iy]
                    -bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_pfwallh[ix,0]*com.sy[ix,iy]
                    +bbb.pwr_pfwallz[ix,0]*com.sy[ix,iy])
        areas.append(com.sy[ix,iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[ixpt+2,iy,0]+com.rm[com.nx,iy,0])/2-offset
    zcenter = (com.zm[ixpt+2,iy,0]+com.zm[com.nx,iy,0])/2
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='right', va='top', size=fontsize, bbox=bdict))
    # Outer target
    pxs = []
    pbinds = []
    ix = com.nx
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        pbinds.append(xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(xsign*pwrx[ix,iy]
                    +xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_plth[iy,plateIndex]*com.sxnp[ix,iy]
                    +bbb.pwr_pltz[iy,plateIndex]*com.sxnp[ix,iy])
        areas.append(com.sxnp[ix,iy])
    pout += sum(pxs) + sum(pbinds)
    rcenter = com.rm[ix+1,1,0]
    zcenter = com.zm[ix+1,1,0]-offset
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6) + '\n' + r'$P_{\parallel bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='center', va='top', size=fontsize, bbox=bdict))
    # Total power
    segments = np.array(segments)
    ptot = np.array(ptot)
    areas = np.array(areas)
    ptot = ptot/areas
    summ = np.sum(ptot*areas)-pent
    # Total radiation
    prad = np.sum((bbb.erliz+bbb.erlrc)[ixpt+2:com.nx+1,1:com.ny+1])
    if bbb.isimpon != 0:
        irad = np.sum((bbb.prad*com.vol)[ixpt+2:com.nx+1,1:com.ny+1])
    else:
        irad = 0
    pout += prad + irad
    text = '\n'.join([r'$\int p_{H\ rad}\ dV=$%.2g MW' % (prad/1e6), 
                      r'$\int p_{imp\ rad}\ dV=$%.2g MW' % (irad/1e6),
                      r'$P_{\parallel\ in}=$%.2g MW' % (pent/1e6),
                      r'$P_{out}=$%.2g MW' % (pout/1e6)])
    plt.text(0.03, 0.03, text, fontsize=fontsize, color=c, ha='left', va='bottom', transform=ax.transAxes, bbox=bdict)
    # Plot line collection
    norm = matplotlib.colors.LogNorm()
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(ptot/1e6)
    lc.set_linewidth(4)
    line = ax.add_collection(lc)
    plt.colorbar(line, ax=ax, orientation='horizontal', pad=0, label=r'$q_{surf\ tot}$ [MW/m$^2$]')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('gray')
    ax.grid(False)
    plt.axis('equal')
    plt.title('Outer leg')
    plt.xlabel(r'$R$ [m]')
    plt.ylabel(r'$Z$ [m]')
    # Scale view to include labels that might otherwise be cut off
    plt.draw()
    for label in labels:
        bbox = label.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        bbox_data = bbox.transformed(ax.transData.inverted()).padded(0)
        ax.update_datalim(bbox_data.corners())
    ax.autoscale_view()
    
    
def plotPowerBalance():
    patches = getPatches()
    args = {'patches': patches, 'rzlabels': False, 'show': False}
    argsBal = {'patches': patches, 'rzlabels': False, 'show': False, 'sym': True}
    plt.subplot(331)
    summ = np.sum(bbb.erliz/1e6)
    plotvar(bbb.erliz/com.vol/1e6, title=r'$P_{rad\ ioniz}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    plt.subplot(332)
    summ = np.sum(bbb.erlrc/1e6)
    plotvar(bbb.erlrc/com.vol/1e6, title=r'$P_{rad\ recomb}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    plt.subplot(333)
    if bbb.isimpon != 0:
        summ = np.sum(bbb.prad/1e6*com.vol)
        plotvar(bbb.prad/1e6, title=r'$P_{rad\ imp}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    else:
        plt.axis('off')
    plt.subplot(334)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourcePoloidal(bbb.feex+bbb.feix, ix, iy))/com.vol/1e6, title=r'$P_{poloidal}$ [MW/m$^3$]', log=True, **argsBal)
    # sumAbs = analysis.gridPowerSumAbs()
    # plotvar(-bbb.erliz/sumAbs, title=r'$\frac{P_{rad\ ioniz}}{\Sigma_j|P_j|}$', **argsBal)
    plt.subplot(335)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourceRadial(bbb.feey+bbb.feiy, ix, iy))/com.vol/1e6, title=r'$P_{radial}$ [MW/m$^3$]', log=True, **argsBal)
    # plotvar(-bbb.erlrc/sumAbs, title=r'$\frac{P_{rad\ recomb}}{\Sigma_j|P_j|}$', **argsBal)
    # plt.subplot(336)
    # if bbb.isimpon != 0:
    #     plotvar(-bbb.prad/sumAbs, title=r'$\frac{P_{rad\ imp}}{\Sigma_j|P_j|}$', **argsBal)
    # else:
    #     plt.axis('off')
    # plt.subplot(337)
    # plt.subplot(338)
    plt.subplot(337)
    plotvar(analysis.PionParallelKE()/com.vol/1e6, title=r'$P_{ion\ KE}$ [MW/m$^3$]', log=True, **argsBal)
    plt.subplot(339)
    plotvar(analysis.gridPowerBalance()/com.vol/1e6, title=r'Power balance [MW/m$^3$]', log=True, **argsBal)


    
def plotPowerBalance_shahinul():
    patches = getPatches()
    args = {'patches': patches, 'rzlabels': False, 'show': False}
    argsBal = {'patches': patches, 'rzlabels': False, 'show': False, 'sym': True}
    #plt.subplot(331)
    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    summ = np.sum(bbb.erliz/1e6)
    plotvar(bbb.erliz/com.vol/1e6, title=r'$P_{rad\ ioniz}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    plt.subplot(221)
    summ = np.sum(bbb.erlrc/1e6)
    plotvar(bbb.erlrc/com.vol/1e6, title=r'$P_{rad\ recomb}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    plt.subplot(222)
    if bbb.isimpon != 0:
        summ = np.sum(bbb.prad/1e6*com.vol)
        plotvar(bbb.prad/1e6, title=r'$P_{rad\ imp}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    else:
        plt.axis('off')
    plt.subplot(223)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourcePoloidal(bbb.feex+bbb.feix, ix, iy))/com.vol/1e6, title=r'$P_{poloidal}$ [MW/m$^3$]', log=True, **argsBal)
    # sumAbs = analysis.gridPowerSumAbs()
    # plotvar(-bbb.erliz/sumAbs, title=r'$\frac{P_{rad\ ioniz}}{\Sigma_j|P_j|}$', **argsBal)
    plt.subplot(335)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourceRadial(bbb.feey+bbb.feiy, ix, iy))/com.vol/1e6, title=r'$P_{radial}$ [MW/m$^3$]', log=True, **argsBal)
    # plotvar(-bbb.erlrc/sumAbs, title=r'$\frac{P_{rad\ recomb}}{\Sigma_j|P_j|}$', **argsBal)
    # plt.subplot(336)
    # if bbb.isimpon != 0:
    #     plotvar(-bbb.prad/sumAbs, title=r'$\frac{P_{rad\ imp}}{\Sigma_j|P_j|}$', **argsBal)
    # else:
    #     plt.axis('off')
    # plt.subplot(337)
    # plt.subplot(338)
    plt.subplot(224)
    plotvar(analysis.PionParallelKE()/com.vol/1e6, title=r'$P_{ion\ KE}$ [MW/m$^3$]', log=True, **argsBal)
    plt.subplot(225)
    plotvar(analysis.gridPowerBalance()/com.vol/1e6, title=r'Power balance [MW/m$^3$]', log=True, **argsBal)



    
    
def plotDensityBalance(patches):
    args = {'patches': patches, 'rzlabels': False, 'show': False, 'sym': True}
    plt.subplot(321)
    #sumAbs = analysis.gridParticleSumAbs()
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourcePoloidal(bbb.fnix[:,:,0], ix, iy)), title=r'Poloidal source [s$^{-1}$]', log=True, **args)
    plt.subplot(322)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourceRadial(bbb.fniy[:,:,0], ix, iy)), title=r'Radial source [s$^{-1}$]', log=True, **args)
    plt.subplot(323)
    plotvar(bbb.psor[:,:,0], title=r'Ionization source [s$^{-1}$]', log=True, **args)
    plt.subplot(324)
    plotvar(-bbb.psorrg[:,:,0], title=r'Recombination source [s$^{-1}$]', log=True, **args)
    plt.subplot(325)
    plotvar(analysis.gridParticleBalance(), title=r'Particle balance [s$^{-1}$]', log=True, **args)
    
    
def plotRadialFluxes():
    fniy = np.zeros((com.nx+2,com.ny+2))
    fniydd = np.zeros((com.nx+2,com.ny+2))
    fniydif = np.zeros((com.nx+2,com.ny+2))
    fniyconv = np.zeros((com.nx+2,com.ny+2))
    fniyef = np.zeros((com.nx+2,com.ny+2))
    fniybf = np.zeros((com.nx+2,com.ny+2))
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0]-vyconv
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            # This is for upwind scheme (methn=33)
            if bbb.vy[ix,iy,0] > 0:
                t2 = bbb.niy0[ix,iy,0] # outside sep in case I developed this with
            else:
                t2 = bbb.niy1[ix,iy,0] # inside sep in case I developed this with
            fniy[ix,iy] = bbb.cnfy*bbb.vy[ix,iy,0]*com.sy[ix,iy]*t2
            fniydd[ix,iy] = bbb.vydd[ix,iy,0]*com.sy[ix,iy]*t2
            fniydif[ix,iy] = vydif[ix,iy]*com.sy[ix,iy]*t2
            fniyconv[ix,iy] = vyconv[ix,iy]*com.sy[ix,iy]*t2
            fniyef[ix,iy] = bbb.cfyef*bbb.vyce[ix,iy,0]*com.sy[ix,iy]*t2
            fniybf[ix,iy] = bbb.cfybf*bbb.vycb[ix,iy,0]*com.sy[ix,iy]*t2
            if bbb.vy[ix,iy,0]*(bbb.ni[ix,iy,0]-bbb.ni[ix,iy+1,0]) < 0:
                fniy[ix,iy] = fniy[ix,iy]/(1+(bbb.nlimiy[0]/bbb.ni[ix,iy+1,0])**2+(bbb.nlimiy[0]/bbb.ni[ix,iy,0])**2) # nlimiy is 0 rn... might help with converg. problems in SPARC

    def upwind(f, p1, p2): 
        return max(f,0)*p1+min(f,0)*p2

    def upwindProxy(f, g, p1, p2):
        return max(f,0)/f*g*p1+min(f,0)/f*g*p2

    feey = np.zeros((com.nx+2,com.ny+2))
    econv = np.zeros((com.nx+2,com.ny+2))
    econd = np.zeros((com.nx+2,com.ny+2))
    feiy = np.zeros((com.nx+2,com.ny+2))
    iconv = np.zeros((com.nx+2,com.ny+2))
    nconv = np.zeros((com.nx+2,com.ny+2))
    icond = np.zeros((com.nx+2,com.ny+2))
    ncond = np.zeros((com.nx+2,com.ny+2))
    conyn = com.sy*bbb.hcyn/com.dynog
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            econd[ix,iy]=-bbb.conye[ix,iy]*(bbb.te[ix,iy+1]-bbb.te[ix,iy])
            econv[ix,iy]=upwind(bbb.floye[ix,iy],bbb.te[ix,iy],bbb.te[ix,iy+1])
            ncond[ix,iy]=-conyn[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])
            icond[ix,iy]=-bbb.conyi[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])-ncond[ix,iy]
            floyn = bbb.cfneut*bbb.cfneutsor_ei*2.5*bbb.fngy[ix,iy,0]
            floyi = bbb.floyi[ix,iy]-floyn # ions only, unlike bbb.floyi
            iconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyi,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
            nconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyn,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
    feey = econd+econv # should match bbb.feey
    feiy = icond+iconv+ncond+nconv # should match bbb.feiy

    ix = com.isixcore == 1

    plt.subplot(311)
    plt.plot(com.yyc*1000, np.sum(bbb.fniy[ix,:,0],axis=0), c='k', label=r'Total ion flux');
    plt.plot(com.yyc*1000, np.sum(fniydif[ix,:],axis=0), label=r'Diffusion', c='C0')
    #plt.plot(com.yyc*1000, (-bbb.dif_use[:,:,0]*(bbb.niy1[:,:,0]-bbb.niy0[:,:,0])*com.gyf*com.sy)[ix,:], label=r'$-D(\nabla n) A_y$',c='C0',ls='--');
    plt.plot(com.yyc*1000, np.sum(fniyconv[ix,:],axis=0), label=r'Convection', c='C1')
    #plt.plot(com.yyc*1000, (vyconv*bbb.ni[:,:,0]*com.sy)[ix,:], label=r'$v_{conv}n_iA_y$',c='C1',ls='-');
    plt.plot(com.yyc*1000, np.sum(fniyef[ix,:],axis=0), label=r'$E\times B$ convection',c='C7',ls='--')
    plt.plot(com.yyc*1000, np.sum(fniybf[ix,:],axis=0), label=r'$\nabla B$ convection',c='C3',ls='--')
    plt.plot(com.yyc*1000, np.sum((bbb.fngy[:,:,0])[ix,:],axis=0), label='(Neutral flux)',c='C4',ls='-.')
    plt.plot(com.yyc*1000, np.sum((fniydif+fniyconv+fniyef+fniybf)[ix,:],axis=0), label='Sum of components',c='k',ls=':')
    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([-maxabs, maxabs])
    plt.ylabel('Flux [s$^{-1}$]')
    plt.xlabel(r'$R-R_{sep}$ [mm]')
    plt.title('Sum over core poloidal cells\n')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

    plt.subplot(312)
    mytot = icond+ncond+iconv+nconv
    plt.plot(com.yyc*1000, np.sum(bbb.feiy[ix,:]/1e6,axis=0), c='k', label='i+n conv.+cond.');
    plt.plot(com.yyc*1000, np.sum(icond[ix,:]/1e6,axis=0), ls='-', label='Ion conduction', c='C0');
    plt.plot(com.yyc*1000, np.sum(ncond[ix,:]/1e6,axis=0), ls='--', label='Neutral conduction', c='C0');
    plt.plot(com.yyc*1000, np.sum(iconv[ix,:]/1e6,axis=0), ls='-', label='Ion convection', c='C1');
    plt.plot(com.yyc*1000, np.sum(nconv[ix,:]/1e6,axis=0), ls='--', label='Neutral convection', c='C1');
    plt.plot(com.yyc*1000, np.sum(mytot[ix,:]/1e6,axis=0), c='k', ls=':', label='Sum of components')

    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([-maxabs, maxabs])
    plt.ylabel('Power [MW]')
    plt.xlabel(r'$R-R_{sep}$ [mm]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

    plt.subplot(313)

    plt.plot(com.yyc*1000, np.sum(bbb.feey[ix,:]/1e6,axis=0), c='k', label='Electron conv.+cond.');
    plt.plot(com.yyc*1000, np.sum(econd[ix,:]/1e6,axis=0), ls='-', label='Electron conduction', c='C0');
    plt.plot(com.yyc*1000, np.sum(econv[ix,:]/1e6,axis=0), ls='-', label='Electron convection', c='C1');
    plt.plot(com.yyc*1000, np.sum((econd+econv)[ix,:]/1e6,axis=0), c='k', ls=':', label='Sum of components')

    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([-maxabs, maxabs])
    plt.ylabel('Power [MW]')
    plt.xlabel(r'$R-R_{sep}$ [mm]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

def plotRadialFluxes_particle():
    fniy = np.zeros((com.nx+2,com.ny+2))
    fniydd = np.zeros((com.nx+2,com.ny+2))
    fniydif = np.zeros((com.nx+2,com.ny+2))
    fniyconv = np.zeros((com.nx+2,com.ny+2))
    fniyef = np.zeros((com.nx+2,com.ny+2))
    fniybf = np.zeros((com.nx+2,com.ny+2))
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0]-vyconv
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            # This is for upwind scheme (methn=33)
            if bbb.vy[ix,iy,0] > 0:
                t2 = bbb.niy0[ix,iy,0] # outside sep in case I developed this with
            else:
                t2 = bbb.niy1[ix,iy,0] # inside sep in case I developed this with
            fniy[ix,iy] = bbb.cnfy*bbb.vy[ix,iy,0]*com.sy[ix,iy]*t2
            fniydd[ix,iy] = bbb.vydd[ix,iy,0]*com.sy[ix,iy]*t2
            fniydif[ix,iy] = vydif[ix,iy]*com.sy[ix,iy]*t2
            fniyconv[ix,iy] = vyconv[ix,iy]*com.sy[ix,iy]*t2
            fniyef[ix,iy] = bbb.cfyef*bbb.vyce[ix,iy,0]*com.sy[ix,iy]*t2
            fniybf[ix,iy] = bbb.cfybf*bbb.vycb[ix,iy,0]*com.sy[ix,iy]*t2
            if bbb.vy[ix,iy,0]*(bbb.ni[ix,iy,0]-bbb.ni[ix,iy+1,0]) < 0:
                fniy[ix,iy] = fniy[ix,iy]/(1+(bbb.nlimiy[0]/bbb.ni[ix,iy+1,0])**2+(bbb.nlimiy[0]/bbb.ni[ix,iy,0])**2) # nlimiy is 0 rn... might help with converg. problems in SPARC

    def upwind(f, p1, p2): 
        return max(f,0)*p1+min(f,0)*p2

    def upwindProxy(f, g, p1, p2):
        return max(f,0)/f*g*p1+min(f,0)/f*g*p2

    feey = np.zeros((com.nx+2,com.ny+2))
    econv = np.zeros((com.nx+2,com.ny+2))
    econd = np.zeros((com.nx+2,com.ny+2))
    feiy = np.zeros((com.nx+2,com.ny+2))
    iconv = np.zeros((com.nx+2,com.ny+2))
    nconv = np.zeros((com.nx+2,com.ny+2))
    icond = np.zeros((com.nx+2,com.ny+2))
    ncond = np.zeros((com.nx+2,com.ny+2))
    conyn = com.sy*bbb.hcyn/com.dynog
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            econd[ix,iy]=-bbb.conye[ix,iy]*(bbb.te[ix,iy+1]-bbb.te[ix,iy])
            econv[ix,iy]=upwind(bbb.floye[ix,iy],bbb.te[ix,iy],bbb.te[ix,iy+1])
            ncond[ix,iy]=-conyn[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])
            icond[ix,iy]=-bbb.conyi[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])-ncond[ix,iy]
            floyn = bbb.cfneut*bbb.cfneutsor_ei*2.5*bbb.fngy[ix,iy,0]
            floyi = bbb.floyi[ix,iy]-floyn # ions only, unlike bbb.floyi
            iconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyi,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
            nconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyn,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
    feey = econd+econv # should match bbb.feey
    feiy = icond+iconv+ncond+nconv # should match bbb.feiy

    ix = com.isixcore == 1

    #plt.subplot(311)
    plt.figure(figsize=(5,3.))
    plt.plot(com.yyc*1000, np.sum(bbb.fniy[ix,:,0],axis=0)/1e22, c='k', label=r'Total', linewidth=2);
    plt.plot(com.yyc*1000, np.sum(fniydif[ix,:],axis=0)/1e22, c='b', label=r'Diffusion',linewidth=2 )
    plt.plot(com.yyc*1000, np.sum(fniyef[ix,:],axis=0)/1e22, c='r', ls='--', label=r'$E\times B$', linewidth=2)
    plt.plot(com.yyc*1000, np.sum(fniybf[ix,:],axis=0)/1e22,   c='g', ls=':', label='$\\nabla B$', linewidth=2)
    #plt.plot(com.yyc*1000, np.sum((fniydif+fniyconv+fniyef+fniybf)[ix,:],axis=0)/1e22,  c='r', ls=':', label='Sum of components',linewidth=2)
    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([0, maxabs])

    plt.ylabel('Radial Flux [10$^{22}$ s$^{-1}$]', fontsize=16)
    plt.xlabel(r'$R_{omp}-R_{sep}$ [mm]',  fontsize=16)
    #plt.title('Sum over core poloidal cells\n')
    #plt.legend(loc='best', bbox_to_anchor=(1, 0.5));
    plt.legend(loc='best', fontsize=10);
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig('flux.png', dpi = 300)



def plotRadialFluxes_heat():
    fniy = np.zeros((com.nx+2,com.ny+2))
    pe = np.zeros((com.nx+2,com.ny+2))
    pi = np.zeros((com.nx+2,com.ny+2))
    q_cond = np.zeros((com.nx+2,com.ny+2))
    fniydd = np.zeros((com.nx+2,com.ny+2))
    fniydif = np.zeros((com.nx+2,com.ny+2))
    fniyconv = np.zeros((com.nx+2,com.ny+2))
    fniyef = np.zeros((com.nx+2,com.ny+2))
    fniybf = np.zeros((com.nx+2,com.ny+2))
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0]-vyconv
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            # This is for upwind scheme (methn=33)
            if bbb.vy[ix,iy,0] > 0:
                t2 = bbb.niy0[ix,iy,0] # outside sep in case I developed this with
            else:
                t2 = bbb.niy1[ix,iy,0] # inside sep in case I developed this with
            fniy[ix,iy] = bbb.cnfy*bbb.vy[ix,iy,0]*com.sy[ix,iy]*t2
            fniydd[ix,iy] = bbb.vydd[ix,iy,0]*com.sy[ix,iy]*t2
            fniydif[ix,iy] = vydif[ix,iy]*com.sy[ix,iy]*t2
            fniyconv[ix,iy] = vyconv[ix,iy]*com.sy[ix,iy]*t2
            fniyef[ix,iy] = bbb.cfyef*bbb.vyce[ix,iy,0]*com.sy[ix,iy]*t2
            fniybf[ix,iy] = bbb.cfybf*bbb.vycb[ix,iy,0]*com.sy[ix,iy]*t2
            pe[ix,iy] = 2.5 * (bbb.ne[ix,iy] * bbb.te[ix,iy])
            pi[ix,iy] =  2.5*t2 * bbb.ti[ix,iy]
            q_cond[ix,iy]= (-(5/2) * bbb.kye * (bbb.ne[ix,iy] * bbb.gtey[ix,iy] + t2* bbb.gtiy[ix,iy]))*com.sy[ix,iy]
            if bbb.vy[ix,iy,0]*(bbb.ni[ix,iy,0]-bbb.ni[ix,iy+1,0]) < 0:
                fniy[ix,iy] = fniy[ix,iy]/(1+(bbb.nlimiy[0]/bbb.ni[ix,iy+1,0])**2+(bbb.nlimiy[0]/bbb.ni[ix,iy,0])**2) # nlimiy is 0 rn... might help with converg. problems in SPARC

    def upwind(f, p1, p2): 
        return max(f,0)*p1+min(f,0)*p2

    def upwindProxy(f, g, p1, p2):
        return max(f,0)/f*g*p1+min(f,0)/f*g*p2

    feey = np.zeros((com.nx+2,com.ny+2))
    econv = np.zeros((com.nx+2,com.ny+2))
    econd = np.zeros((com.nx+2,com.ny+2))
    feiy = np.zeros((com.nx+2,com.ny+2))
    iconv = np.zeros((com.nx+2,com.ny+2))
    nconv = np.zeros((com.nx+2,com.ny+2))
    icond = np.zeros((com.nx+2,com.ny+2))
    ncond = np.zeros((com.nx+2,com.ny+2))
    conyn = com.sy*bbb.hcyn/com.dynog
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            econd[ix,iy]=-bbb.conye[ix,iy]*(bbb.te[ix,iy+1]-bbb.te[ix,iy])
            econv[ix,iy]=upwind(bbb.floye[ix,iy],bbb.te[ix,iy],bbb.te[ix,iy+1])
            ncond[ix,iy]=-conyn[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])
            icond[ix,iy]=-bbb.conyi[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])-ncond[ix,iy]
            floyn = bbb.cfneut*bbb.cfneutsor_ei*2.5*bbb.fngy[ix,iy,0]
            floyi = bbb.floyi[ix,iy]-floyn # ions only, unlike bbb.floyi
            iconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyi,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
            nconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyn,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
       
    feey = econd+econv # should match bbb.feey
    feiy = icond+iconv+ncond+nconv # should match bbb.feiy

    ix = com.isixcore == 1

    q_radial = (bbb.fniy[:,:,0] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6 
    pwrx = bbb.feex+bbb.feix
    pwry = (bbb.feey+bbb.feiy)/1e6
    #p = 2.5 * (bbb.ne * bbb.te + bbb.ni[:, :, 0] * bbb.ti)
    nx, ny = com.nx + 2, com.ny + 2
    q_ExB = np.zeros((nx, ny))
    q_gradB = np.zeros((nx, ny))

    q_ExB[:, :] = (com.sy*bbb.vyce[:, :,0] * (pe[:,:]+pi[:,:]))/1e6
    q_gradB[:, :] = com.sy*(bbb.vycb[:, :,0] * pi[:,:] + bbb.veycb[:, :] * pe[:,:])/1e6
    q_radial = (bbb.fniy[:,:,0] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6 
     
    q_dif = (fniydif[:,:] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6 
    q_yef = (fniyef[:,:] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6
    q_ybf = (fniybf[:,:] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6

    dTe_dy = bbb.gtey   # [J/m]
    dTi_dy = bbb.gtiy 

    #q_cond = (-(5/2) * bbb.kye * (bbb.ne * dTe_dy + bbb.ni[:,:,0] * dTi_dy))*com.sy #*vydif
    q_conv = ((pe+pi) * vydif)*com.sy
    q_anom = q_conv #+q_cond [:,:] # Total radial heat flux density [W/m ]
    q_anom = q_anom/1e6
    Total_heat = q_anom+q_gradB+q_ExB
    lambda_exp_fit_out, lambda_eich_fit_out = analysis.eich_exp_shahinul_odiv_final()
    lambda_q = lambda_exp_fit_out  # In mm




    plt.figure(figsize=(5,3.))
    #plt.plot(com.yyc*1000, np.sum(q_radial[ix,:] ,axis=0), c='k', label=r'Total', linewidth=2)
    plt.plot(com.yyc[1:-1]*1000, np.sum((pwry)[ix,1:-1],axis=0),  c='r', ls=':', label='feiy+feey',linewidth=2)
    plt.plot(com.yyc[1:-1]*1000, np.sum(q_anom[ix,1:-1],axis=0), c='b', label=r'Diffusion',linewidth=2 )
    plt.plot(com.yyc[1:-1]*1000, np.sum(q_ExB[ix,1:-1],axis=0), c='r', ls='--', label=r'$E\times B$', linewidth=2)
    plt.plot(com.yyc[1:-1]*1000, np.sum(q_gradB[ix,1:-1],axis=0),   c='g', ls=':', label='$\\nabla B$', linewidth=2)
    plt.plot(com.yyc[1:-1]*1000, np.sum(Total_heat[ix,1:-1],axis=0),  c='k', ls='-', label='Total',linewidth=2)
    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([0, maxabs])

    plt.ylabel('Radial Flux [MW]', fontsize=16)
    plt.xlabel(r'$R_{omp}-R_{sep}$ [mm]',  fontsize=16)
    #plt.title('Sum over core poloidal cells\n')
    #plt.legend(loc='best', bbox_to_anchor=(1, 0.5));
    plt.legend(loc='best', fontsize=10);
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig('heat_radial_flux.png', dpi = 300)
    
        #plt.subplot(311)
    plt.figure(figsize=(5,3.))
    plt.plot(com.yyc*1000, np.sum(bbb.fniy[ix,:,0],axis=0)/1e22, c='k', label=r'fniy', linewidth=2);
    plt.plot(com.yyc*1000, np.sum(fniydif[ix,:],axis=0)/1e22, c='b', label=r'Diffusion',linewidth=2 )
    plt.plot(com.yyc*1000, np.sum(fniyef[ix,:],axis=0)/1e22, c='r', ls='--', label=r'$E\times B$', linewidth=2)
    plt.plot(com.yyc*1000, np.sum(fniybf[ix,:],axis=0)/1e22,   c='g', ls=':', label='$\\nabla B$', linewidth=2)
    plt.plot(com.yyc*1000, np.sum((fniydif+fniyconv+fniyef+fniybf)[ix,:],axis=0)/1e22,  c='r', ls=':', label='Sum of components',linewidth=2)
    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([0, maxabs])

    plt.ylabel('Radial Flux [10$^{22}$ s$^{-1}$]', fontsize=16)
    plt.xlabel(r'$R_{omp}-R_{sep}$ [mm]',  fontsize=16)
    #plt.title('Sum over core poloidal cells\n')
    #plt.legend(loc='best', bbox_to_anchor=(1, 0.5));
    plt.legend(loc='best', fontsize=10);
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig('flux.png', dpi = 300)

    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    ax = axs[0]
    ax.plot(com.yyc*1000, np.sum(q_anom[ix,:], axis=0), c='b', label='Diffusion', linewidth=2)
    ax.plot(com.yyc*1000, np.sum(q_ExB[ix,:], axis=0), c='r', ls='--', label='$E\\times B$', linewidth=2)
    ax.plot(com.yyc*1000, np.sum(q_gradB[ix,:], axis=0), c='g', ls=':', label='$\\nabla B$', linewidth=2)
    ax.plot(com.yyc*1000, np.sum(Total_heat[ix,:], axis=0), c='k', ls='-', label='Total', linewidth=2)
    ax.plot(com.yyc*1000, np.sum((pwry)[ix,:],axis=0),  c='r', ls=':', label='feiy+feey',linewidth=2)

    ylim = ax.get_ylim()
    maxabs = np.max(np.abs(ylim))
    ax.set_ylim([0, 6])
    ax.set_ylabel('q$_{radial}$ [MW]', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(r"$\lambda_q \approx {:.2f}\ \mathrm{{mm}}$".format(lambda_q), fontsize=14)
    ax.grid(True, which='both', axis='y', color='#ddd')
    ax.set_axisbelow(True)
    for i in com.yyc*1000:
        ax.axvline(i, c='#ddd', lw=0.5, zorder=0)

    ax = axs[1]
    ax.plot(com.yyc*1000, np.sum(fniydif[ix,:], axis=0)/1e22, c='b', label='Diffusion', linewidth=2)
    ax.plot(com.yyc*1000, np.sum(fniyef[ix,:], axis=0)/1e22, c='r', ls='--', label='$E\\times B$', linewidth=2)
    ax.plot(com.yyc*1000, np.sum(fniybf[ix,:], axis=0)/1e22, c='g', ls=':', label='$\\nabla B$', linewidth=2)
    ax.plot(com.yyc*1000, np.sum(bbb.fniy[ix,:,0], axis=0)/1e22, c='k', ls='-', label='Total', linewidth=2)
    #plt.plot(com.yyc*1000, np.sum(bbb.fniy[ix,:,0],axis=0)/1e22, c='k', label=r'fniy', linewidth=2);
    ylim = ax.get_ylim()
    maxabs = np.max(np.abs(ylim))
    ax.set_ylim([0, maxabs])
    ax.set_ylabel('$\Gamma_{radial}$ [10$^{22}$ s$^{-1}$]', fontsize=14)
    ax.set_xlabel(r'$R_{omp}-R_{sep}$ [mm]', fontsize=14)
    #ax.legend(loc='best', fontsize=9)
    ax.tick_params(axis='both', labelsize=12)
    for i in com.yyc*1000:
         ax.axvline(i, c='#ddd', lw=0.5, zorder=0)
    ax.grid(True, which='both', axis='y', color='#ddd')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('flux_subplots.png', dpi=300)
    plt.show()


    

def RadialFluxes_heat_return():
    fniy = np.zeros((com.nx+2,com.ny+2))
    p = np.zeros((com.nx+2,com.ny+2))
    q_cond = np.zeros((com.nx+2,com.ny+2))
    fniydd = np.zeros((com.nx+2,com.ny+2))
    fniydif = np.zeros((com.nx+2,com.ny+2))
    fniyconv = np.zeros((com.nx+2,com.ny+2))
    fniyef = np.zeros((com.nx+2,com.ny+2))
    fniybf = np.zeros((com.nx+2,com.ny+2))
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0]-vyconv
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            # This is for upwind scheme (methn=33)
            if bbb.vy[ix,iy,0] > 0:
                t2 = bbb.niy0[ix,iy,0] # outside sep in case I developed this with
            else:
                t2 = bbb.niy1[ix,iy,0] # inside sep in case I developed this with
            fniy[ix,iy] = bbb.cnfy*bbb.vy[ix,iy,0]*com.sy[ix,iy]*t2
            fniydd[ix,iy] = bbb.vydd[ix,iy,0]*com.sy[ix,iy]*t2
            fniydif[ix,iy] = vydif[ix,iy]*com.sy[ix,iy]*t2
            fniyconv[ix,iy] = vyconv[ix,iy]*com.sy[ix,iy]*t2
            fniyef[ix,iy] = bbb.cfyef*bbb.vyce[ix,iy,0]*com.sy[ix,iy]*t2
            fniybf[ix,iy] = bbb.cfybf*bbb.vycb[ix,iy,0]*com.sy[ix,iy]*t2
            p[ix,iy] = 2.5 * (bbb.ne[ix,iy] * bbb.te[ix,iy] + t2 * bbb.ti[ix,iy])
            q_cond[ix,iy] = (-(5/2) * bbb.kye * (bbb.ne[ix,iy] * bbb.gtey[ix,iy] + t2* bbb.gtiy[ix,iy]))*com.sy[ix,iy]
            if bbb.vy[ix,iy,0]*(bbb.ni[ix,iy,0]-bbb.ni[ix,iy+1,0]) < 0:
                fniy[ix,iy] = fniy[ix,iy]/(1+(bbb.nlimiy[0]/bbb.ni[ix,iy+1,0])**2+(bbb.nlimiy[0]/bbb.ni[ix,iy,0])**2) # nlimiy is 0 rn... might help with converg. problems in SPARC

    def upwind(f, p1, p2): 
        return max(f,0)*p1+min(f,0)*p2

    def upwindProxy(f, g, p1, p2):
        return max(f,0)/f*g*p1+min(f,0)/f*g*p2

    feey = np.zeros((com.nx+2,com.ny+2))
    econv = np.zeros((com.nx+2,com.ny+2))
    econd = np.zeros((com.nx+2,com.ny+2))
    feiy = np.zeros((com.nx+2,com.ny+2))
    iconv = np.zeros((com.nx+2,com.ny+2))
    nconv = np.zeros((com.nx+2,com.ny+2))
    icond = np.zeros((com.nx+2,com.ny+2))
    ncond = np.zeros((com.nx+2,com.ny+2))
    conyn = com.sy*bbb.hcyn/com.dynog
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            econd[ix,iy]=-bbb.conye[ix,iy]*(bbb.te[ix,iy+1]-bbb.te[ix,iy])
            econv[ix,iy]=upwind(bbb.floye[ix,iy],bbb.te[ix,iy],bbb.te[ix,iy+1])
            ncond[ix,iy]=-conyn[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])
            icond[ix,iy]=-bbb.conyi[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])-ncond[ix,iy]
            floyn = bbb.cfneut*bbb.cfneutsor_ei*2.5*bbb.fngy[ix,iy,0]
            floyi = bbb.floyi[ix,iy]-floyn # ions only, unlike bbb.floyi
            iconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyi,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
            nconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyn,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
       
    feey = econd+econv # should match bbb.feey
    feiy = icond+iconv+ncond+nconv # should match bbb.feiy

    ix = com.isixcore == 1

    q_radial = (bbb.fniy[:,:,0] * (3/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6 
    pwrx = bbb.feex+bbb.feix
    pwry = (bbb.feey+bbb.feiy)/1e6
    #p = 2.5 * (bbb.ne * bbb.te + bbb.ni[:, :, 0] * bbb.ti)
    nx, ny = com.nx + 2, com.ny + 2
    q_ExB = np.zeros((nx, ny))
    q_gradB = np.zeros((nx, ny))

    q_ExB[:, :] = (com.sy*bbb.vyce[:, :,0] * p[:,:])/1e6
    q_gradB[:, :] = (com.sy*bbb.vycb[:, :,0] * p[:,:])/1e6
    q_radial = (bbb.fniy[:,:,0] * (3/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6 
     
    q_dif = (fniydif[:,:] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6 
    q_yef = (fniyef[:,:] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6
    q_ybf = (fniybf[:,:] * (5/2) * (bbb.te[:,:] + bbb.ti[:,:])) / 1e6

    dTe_dy = bbb.gtey   # [J/m]
    dTi_dy = bbb.gtiy 

    #q_cond = (-(5/2) * bbb.kye * (bbb.ne * dTe_dy + bbb.ni[:,:,0] * dTi_dy))*com.sy #*vydif
    q_conv = (p[:,:] * vydif)*com.sy
    q_anom = q_conv + q_cond [:,:] # Total radial heat flux density [W/m ]
    q_anom = q_anom/1e6
    Total_heat = q_anom+q_gradB+q_ExB
    lambda_exp_fit_out, lambda_eich_fit_out = analysis.eich_exp_shahinul_odiv_final()
    lambda_q = lambda_exp_fit_out  # In mm

    return np.sum(q_anom[ix,:], axis=0),  np.sum(q_ExB[ix,:], axis=0), np.sum(q_gradB[ix,:], axis=0), np.sum(Total_heat[ix,:], axis=0), np.sum(fniydif[ix,:], axis=0)/1e22,  np.sum(fniyef[ix,:], axis=0)/1e22, np.sum(fniybf[ix,:], axis=0)/1e22, np.sum(bbb.fniy[ix,:,0], axis=0)/1e22



def plotRadialFluxes_new():
    # --- Initialize arrays ---
    shape = (com.nx+2, com.ny+2)
    fniy = np.zeros(shape)
    fniydif = np.zeros(shape)
    fniyconv = np.zeros(shape)
    fniyef = np.zeros(shape)
    fniybf = np.zeros(shape)

    vyconv = bbb.vcony[0] + bbb.vy_use[:, :, 0] + bbb.vy_cft[:, :, 0]
    vydif = bbb.vydd[:, :, 0] - vyconv

    for ix in range(com.nx+2):
        for iy in range(com.ny+1):
            t2 = bbb.niy0[ix, iy, 0] if bbb.vy[ix, iy, 0] > 0 else bbb.niy1[ix, iy, 0]
            fniy[ix, iy] = bbb.cnfy * bbb.vy[ix, iy, 0] * com.sy[ix, iy] * t2
            fniydif[ix, iy] = vydif[ix, iy] * com.sy[ix, iy] * t2
            fniyconv[ix, iy] = vyconv[ix, iy] * com.sy[ix, iy] * t2
            fniyef[ix, iy] = bbb.cfyef * bbb.vyce[ix, iy, 0] * com.sy[ix, iy] * t2
            fniybf[ix, iy] = bbb.cfybf * bbb.vycb[ix, iy, 0] * com.sy[ix, iy] * t2

            # Convergence damping if needed
            if bbb.vy[ix, iy, 0] * (bbb.ni[ix, iy, 0] - bbb.ni[ix, iy+1, 0]) < 0:
                fniy[ix, iy] /= (1 + (bbb.nlimiy[0]/bbb.ni[ix, iy+1, 0])**2 + (bbb.nlimiy[0]/bbb.ni[ix, iy, 0])**2)

    # --- Upwind utilities ---
    def upwind(f, p1, p2): return max(f, 0)*p1 + min(f, 0)*p2
    def upwindProxy(f, g, p1, p2): return max(f, 0)/f * g * p1 + min(f, 0)/f * g * p2 if f != 0 else 0

    # --- Heat flux decomposition ---
    feey = np.zeros(shape)
    feiy = np.zeros(shape)
    econd = np.zeros(shape)
    econv = np.zeros(shape)
    icond = np.zeros(shape)
    iconv = np.zeros(shape)
    ncond = np.zeros(shape)
    nconv = np.zeros(shape)
    conyn = com.sy * bbb.hcyn / com.dynog

    for ix in range(com.nx+2):
        for iy in range(com.ny+1):
            econd[ix, iy] = -bbb.conye[ix, iy] * (bbb.te[ix, iy+1] - bbb.te[ix, iy])
            econv[ix, iy] = upwind(bbb.floye[ix, iy], bbb.te[ix, iy], bbb.te[ix, iy+1])
            ncond[ix, iy] = -conyn[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy])
            icond[ix, iy] = -bbb.conyi[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy]) - ncond[ix, iy]
            floyn = bbb.cfneut * bbb.cfneutsor_ei * 2.5 * bbb.fngy[ix, iy, 0]
            floyi = bbb.floyi[ix, iy] - floyn
            iconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyi, bbb.ti[ix, iy], bbb.ti[ix, iy+1])
            nconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyn, bbb.ti[ix, iy], bbb.ti[ix, iy+1])

    feey = econd + econv
    feiy = icond + iconv + ncond + nconv

    ix = com.isixcore == 1
    x_mm = com.yyc * 1000

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    # --- Ion radial fluxes ---
    axes[0].plot(x_mm, np.sum(bbb.fniy[ix, :, 0], axis=0), 'k', label='Total ion flux')
    axes[0].plot(x_mm, np.sum(fniydif[ix, :], axis=0), label='Diffusion', c='C0')
    axes[0].plot(x_mm, np.sum(fniyconv[ix, :], axis=0), label='Convection', c='C1')
    axes[0].plot(x_mm, np.sum(fniyef[ix, :], axis=0), label=r'$E\times B$', c='C7', ls='--')
    axes[0].plot(x_mm, np.sum(fniybf[ix, :], axis=0), label=r'$\nabla B$', c='C3', ls='--')
    axes[0].plot(x_mm, np.sum(bbb.fngy[:, :, 0][ix, :], axis=0), label='Neutral flux', c='C4', ls='-.')
    axes[0].plot(x_mm, np.sum((fniydif + fniyconv + fniyef + fniybf)[ix, :], axis=0), label='Sum of components', c='k', ls=':')
    axes[0].set_ylabel('Ion Flux [s$^{-1}$]',fontsize= 16)
    axes[0].set_title('Ion Radial Flux Components')
    axes[0].grid(True)
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # --- Ion + Neutral parallel heat flux ---
    total_feiy = feiy[ix, :] / 1e6
    axes[1].plot(x_mm, np.sum(total_feiy, axis=0), 'k', label='Total (i+n)')
    axes[1].plot(x_mm, np.sum(icond[ix, :]/1e6, axis=0), label='Ion cond.', c='C0')
    axes[1].plot(x_mm, np.sum(ncond[ix, :]/1e6, axis=0), label='Neutral cond.', c='C0', ls='--')
    axes[1].plot(x_mm, np.sum(iconv[ix, :]/1e6, axis=0), label='Ion conv.', c='C1')
    axes[1].plot(x_mm, np.sum(nconv[ix, :]/1e6, axis=0), label='Neutral conv.', c='C1', ls='--')
    axes[1].set_ylabel('Power [MW]', fontsize= 16)
    axes[1].set_title('Ion + Neutral Parallel Heat Flux')
    axes[1].grid(True)
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # --- Electron parallel heat flux ---
    total_feey = feey[ix, :] / 1e6
    axes[2].plot(x_mm, np.sum(total_feey, axis=0), 'k', label='Total e')
    axes[2].plot(x_mm, np.sum(econd[ix, :]/1e6, axis=0), label='Electron cond.', c='C0')
    axes[2].plot(x_mm, np.sum(econv[ix, :]/1e6, axis=0), label='Electron conv.', c='C1')
    axes[2].set_xlabel(r'$R - R_{sep}$ [mm]', fontsize= 16)
    axes[2].set_ylabel('Power [MW]', fontsize= 16)
    axes[2].set_title('Electron Parallel Heat Flux')
    axes[2].grid(True)
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for ax in axes:
        ylim = ax.get_ylim()
        maxabs = np.max(np.abs(ylim))
        ax.set_ylim([-maxabs, maxabs])
        for xline in x_mm:
            ax.axvline(xline, color='#ddd', lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plotConductionConvection(export_csv=False, csv_filename="heat_flux_profiles.csv"):
    shape = (com.nx+2, com.ny+2)
    econd = np.zeros(shape)
    econv = np.zeros(shape)
    icond = np.zeros(shape)
    iconv = np.zeros(shape)
    ncond = np.zeros(shape)
    nconv = np.zeros(shape)
    conyn = com.sy * bbb.hcyn / com.dynog

    def upwind(f, p1, p2): return max(f, 0)*p1 + min(f, 0)*p2
    def upwindProxy(f, g, p1, p2): return max(f, 0)/f * g * p1 + min(f, 0)/f * g * p2 if f != 0 else 0

    for ix in range(com.nx+2):
        for iy in range(com.ny+1):
            econd[ix, iy] = -bbb.conye[ix, iy] * (bbb.te[ix, iy+1] - bbb.te[ix, iy])
            econv[ix, iy] = upwind(bbb.floye[ix, iy], bbb.te[ix, iy], bbb.te[ix, iy+1])
            ncond[ix, iy] = -conyn[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy])
            icond[ix, iy] = -bbb.conyi[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy]) - ncond[ix, iy]
            floyn = bbb.cfneut * bbb.cfneutsor_ei * 2.5 * bbb.fngy[ix, iy, 0]
            floyi = bbb.floyi[ix, iy] - floyn
            iconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyi, bbb.ti[ix, iy], bbb.ti[ix, iy+1])
            nconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyn, bbb.ti[ix, iy], bbb.ti[ix, iy+1])

    ix = com.isixcore == 1
    x_mm = com.yyc * 1000

    ion_cond = np.sum(icond[ix, :]/1e6, axis=0)
    ion_conv = np.sum(iconv[ix, :]/1e6, axis=0)
    neut_cond = np.sum(ncond[ix, :]/1e6, axis=0)
    neut_conv = np.sum(nconv[ix, :]/1e6, axis=0)
    elec_cond = np.sum(econd[ix, :]/1e6, axis=0)
    elec_conv = np.sum(econv[ix, :]/1e6, axis=0)

    if export_csv:
        df = pd.DataFrame({
            "R-R_sep_mm": x_mm,
            "Ion_conduction_MW": ion_cond,
            "Neutral_conduction_MW": neut_cond,
            "Ion_convection_MW": ion_conv,
            "Neutral_convection_MW": neut_conv,
            "Electron_conduction_MW": elec_cond,
            "Electron_convection_MW": elec_conv
        })
        df.to_csv(csv_filename, index=False)
        print(f"Profiles exported to {csv_filename}")

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(x_mm, ion_cond, label='Ion conduction', c='C0')
    axes[0].plot(x_mm, neut_cond, label='Neutral conduction', c='C0', ls='--')
    axes[0].plot(x_mm, ion_conv, label='Ion convection', c='C1')
    axes[0].plot(x_mm, neut_conv, label='Neutral convection', c='C1', ls='--')
    axes[0].set_ylabel('Power [MW]')
    axes[0].set_title('Ion + Neutral Heat Flux Components')
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0].grid(True)

    axes[1].plot(x_mm, elec_cond, label='Electron conduction', c='C0')
    axes[1].plot(x_mm, elec_conv, label='Electron convection', c='C1')
    axes[1].set_ylabel('Power [MW]')
    axes[1].set_xlabel(r'$R - R_{sep}$ [mm]')
    axes[1].set_title('Electron Heat Flux Components')
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].grid(True)

    for ax in axes:
        ylim = ax.get_ylim()
        maxabs = np.max(np.abs(ylim))
        ax.set_ylim([-maxabs, maxabs])
        for xline in x_mm:
            ax.axvline(xline, color='#ddd', lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    

def getThetaHat(ix, iy):
    p1R = com.rm[ix,iy,1]
    p1Z = com.zm[ix,iy,1]
    p2R = com.rm[ix,iy,2]
    p2Z = com.zm[ix,iy,2]
    dR = p2R-p1R
    dZ = p2Z-p1Z
    mag = (dR**2+dZ**2)**.5
    return dR/mag, dZ/mag
    
    
def getrHat(ix, iy):
    dR = com.rm[ix,iy,2]-com.rm[ix,iy,1]
    dZ = com.zm[ix,iy,2]-com.zm[ix,iy,1]
    mag = (dR**2+dZ**2)**.5
    return -dZ/mag, dR/mag    

    
def getDriftVector(v2, vy, ix, iy):
    v2 = -np.sign(bbb.b0)*v2
    return v2[ix,iy]*(1-com.rr[ix,iy]**2)**.5*np.array(getThetaHat(ix, iy))+vy[ix,iy]*np.array(getrHat(ix, iy))


def getDriftVectorLog(v2, vy, ix, iy):
    v = getDriftVector(v2, vy, ix, iy)
    vmag = (v[0]**2+v[1]**2)**.5
    v = np.log(vmag)*v/vmag
    if len(v[~np.isfinite(v)]) > 0:
        return np.array([0, 0])
    return v


def getDriftR(v2, vy, ix, iy):
    return getDriftVectorLog(v2, vy, ix, iy)[0]
    
    
def getDriftZ(v2, vy, ix, iy):
    return getDriftVectorLog(v2, vy, ix, iy)[1]
    
    

def plotImps(show=True, set_limits=True, cmap='turbo', Impurity = 'Li', min = 1e10, max=1e19):

    plt.figure(figsize=(6, 6))

    imps = bbb.ni.shape[2] - 2
    ncols = nrows = int((imps + 1) ** 0.5 + 0.5)

    nimp = np.zeros((com.nx + 2, com.ny + 2, imps + 1))
    nimp[:, :, 0] = bbb.ng[:, :, 1]
    nimp[:, :, 1:] = bbb.ni[:, :, 2:]

    vmin = min #np.min(analysis.nonGuard(nimp))
    vmax = max  #np.max(analysis.nonGuard(nimp))

    # Convert colormap name to a colormap object
    cmap_obj = plt.get_cmap(cmap)

    for i in range(imps + 1):
        ax = plt.subplot(nrows, ncols, i + 1)
        plotvar(
            nimp[:, :, i],
            log=True,
            rzlabels=False,
            show=False,
            cmap=cmap_obj,  # Pass the colormap object
            title=f'{Impurity} +{i} [m$^{{-3}}$]',
            vmin=vmin,
            vmax=vmax
        )

        if set_limits:
            ax.set_xlim([0.2, 0.9])
            ax.set_ylim([-1.59, -1.0])

        ax.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], '--m', linewidth=2)

    if show:
        plt.tight_layout()
        plt.show()

def plotImps_Shahinul(show=True, set_limits=True, cmap='turbo', Impurity='Li', min=1e10, max=1e21):
    imps = 3#bbb.ni.shape[2] - 2
    nplots = imps + 1
    ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))

    nimp = np.zeros((com.nx + 2, com.ny + 2, nplots))
    nimp[:, :, 0] = bbb.ng[:, :, 1]
    nimp[:, :, 1:] = bbb.ni[:, :, com.nhsp:]

    vmin = min
    vmax = max
    cmap_obj = plt.get_cmap(cmap)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.0 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    im = None  # For colorbar

    for i in range(nplots):
        ax = axes[i]
        plt.sca(ax)
        im = plotvar(
            nimp[:, :, i],
            log=True,
            rzlabels=True,
            show=False,
            cmap=cmap_obj,
            #title=f'{Impurity}$^{+i}$',
            vmin=vmin,
            vmax=vmax, stats=False
        )
        ax.text(
            0.98, 0.98,
            f'({chr(97 + i)}) {Impurity}$^{{+{i}}}$',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )
        if set_limits:
            ax.set_xlim([0.4, 0.9])
            ax.set_ylim([-1.62, -1.0])
        ax.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], '--m', linewidth=2)
        ax.set_aspect('auto')
        ax.set_facecolor('white')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        if i % ncols == 0:
            ax.set_ylabel('z [m]')
        else:
            ax.tick_params(axis='y', labelleft=False)

        if i < nplots - ncols:
            ax.tick_params(axis='x', labelbottom=False)

    for ax in axes:
        ax.set_xlabel('')
    for ax in axes:
    	ax.set_ylabel('')

    for i in range(nplots - ncols, nplots):
        axes[i].set_xlabel('R [m]', fontsize=16)
    for i in range(0, nplots, ncols):
    	axes[i].set_ylabel('z [m]', fontsize=16)

    for j in range(nplots, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()

    if show:
        plt.show()
    plt.savefig('Li_density.png', dpi=300)


def plotImps_Shahinul_new(show=True, set_limits=True, cmap='turbo', Impurity='Li', min=1e10, max=1e21):
    imps = bbb.ni.shape[2] - 2
    nplots = imps + 1
    ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))

    nimp = np.zeros((com.nx + 2, com.ny + 2, nplots))
    nimp[:, :, 0] = bbb.ng[:, :, 1]
    nimp[:, :, 1:] = bbb.ni[:, :, 2:]

    vmin = min
    vmax = max
    cmap_obj = plt.get_cmap(cmap)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.75 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    im = None  # For colorbar

    for i in range(nplots):
        ax = axes[i]
        plt.sca(ax)
        im = plotvar(
            nimp[:, :, i],
            log=True,
            rzlabels=True,
            show=False,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            stats=False
        )

        # Ion label like Li^{+0}, Li^{+1}, ...
        ax.text(
            0.98, 0.98,
            f'({chr(97 + i)}) {Impurity}$^{{+{i}}}$',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=14,fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
        )

        if set_limits:
            ax.set_xlim([0.2, 0.9])
            ax.set_ylim([-1.59, -1.0])

        ax.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], '--m', linewidth=2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        if i % ncols == 0:
            ax.set_ylabel('z [m]')
        else:
            ax.tick_params(axis='y', labelleft=False)

        if i < nplots - ncols:
            ax.tick_params(axis='x', labelbottom=False)

    # Clean up axes labels
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Add x-labels only on bottom row
    for i in range(nplots - ncols, nplots):
        axes[i].set_xlabel('R [m]', fontsize=16)

    # Add y-labels only on left column
    for i in range(0, nplots, ncols):
        axes[i].set_ylabel('z [m]', fontsize=16)

    # Hide any unused subplots
    for j in range(nplots, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()

    if show:
        plt.show()
    plt.savefig('Li_density.png', dpi=300)


def plotDrift(v2, vy):
    args = {'width': .0015, 'alpha': 1}
    sepwidth = .5
    bdry = []
    bdry.extend([list(zip(com.rm[0,iy,[1,2,4,3,1]],com.zm[0,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[com.nx+1,iy,[1,2,4,3,1]],com.zm[com.nx+1,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[ix,0,[1,2,4,3,1]],com.zm[ix,0,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    bdry.extend([list(zip(com.rm[ix,com.ny+1,[1,2,4,3,1]],com.zm[ix,com.ny+1,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    dR = analysis.toGrid(lambda ix, iy: getDriftR(v2, vy, ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(v2, vy, ix, iy))
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim([np.min(com.zm), com.zm[com.ixpt1[0],com.iysptrx+1,0]/2])
    
    
def plotDrifts(patches):
    bdry = []
    bdry.extend([list(zip(com.rm[0,iy,[1,2,4,3,1]],com.zm[0,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[com.nx+1,iy,[1,2,4,3,1]],com.zm[com.nx+1,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[ix,0,[1,2,4,3,1]],com.zm[ix,0,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    bdry.extend([list(zip(com.rm[ix,com.ny+1,[1,2,4,3,1]],com.zm[ix,com.ny+1,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    args = {'width': .001, 'alpha': 1}
    ylim = [np.min(com.zm), com.zm[com.ixpt1[0],com.iysptrx+1,0]/2]
    sepwidth = .5
    vtot = (bbb.v2ce[:,:,0]**2+bbb.vyce[:,:,0]**2)**.5+(bbb.v2cb[:,:,0]**2+bbb.vycb[:,:,0]**2)**.5+(bbb.v2dd[:,:,0]**2+bbb.vydd[:,:,0]**2)**.5
    # Total drifts
    plt.subplot(221)
    plt.title('Sum of all drifts')
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2[:,:,0], bbb.vy[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2[:,:,0], bbb.vy[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # ExB
    plt.subplot(222)
    percent = np.mean(analysis.nonGuard((bbb.v2ce[:,:,0]**2+bbb.vyce[:,:,0]**2)**.5/vtot))*100
    plt.title(r'$\mathbf{E}\times \mathbf{B}$ drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2ce[:,:,0], bbb.vyce[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2ce[:,:,0], bbb.vyce[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # Grad B
    plt.subplot(223)
    percent = np.mean(analysis.nonGuard((bbb.v2cb[:,:,0]**2+bbb.vycb[:,:,0]**2)**.5/vtot))*100
    plt.title(r'$\mathbf{B}\times\nabla B$ drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.ve2cb[:,:], bbb.veycb[:,:], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.ve2cb[:,:], bbb.veycb[:,:], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], label='electrons', color='C1', **args)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2cb[:,:,0], bbb.vycb[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2cb[:,:,0], bbb.vycb[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    # plt.legend()
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # # Diamagnetic/grad P x B
    # plt.subplot(324)
    # percent = np.mean(analysis.nonGuard((bbb.v2cd[:,:,0]**2+bbb.vycp[:,:,0]**2)**.5/vtot))*100
    # plt.title(r'Diamagnetic drift ($\nabla P\times \mathbf{B}$) (%.2g%%)' % percent)
    # plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    # plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.ve2cd[:,:], bbb.veycp[:,:], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.ve2cd[:,:], bbb.veycp[:,:], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], label='electrons', color='C1', **args)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2cd[:,:,0], bbb.vycp[:,:,0], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2cd[:,:,0], bbb.vycp[:,:,0], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], label='ions', **args)
    # plt.legend()
    # plt.axis('equal')
    # plt.xlim([.45, .75])
    # plt.ylim([-.6, -.2])
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # # Resistive drift
    # plt.subplot(325)
    # percent = np.mean(analysis.nonGuard((bbb.v2rd[:,:,0]**2+bbb.vyrd[:,:,0]**2)**.5/vtot))*100
    # plt.title('Resistive drift (%.2g%%)' % percent)
    # plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    # plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2rd[:,:,0], bbb.vyrd[:,:,0], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2rd[:,:,0], bbb.vyrd[:,:,0], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    # plt.axis('equal')
    # plt.xlim([.45, .75])
    # plt.ylim([-.6, -.2])
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # Anomalous drift
    plt.subplot(224)
    percent = np.mean(analysis.nonGuard((bbb.v2dd[:,:,0]**2+bbb.vydd[:,:,0]**2)**.5/vtot))*100
    plt.title('Anomalous drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2dd[:,:,0], bbb.vydd[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2dd[:,:,0], bbb.vydd[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

def plotDrifts_new(patches): 
    # Boundary setup (unchanged)
    bdry = []
    bdry.extend([list(zip(com.rm[0, iy, [1, 2, 4, 3, 1]], com.zm[0, iy, [1, 2, 4, 3, 1]])) 
                 for iy in np.arange(0, com.ny + 2)])
    bdry.extend([list(zip(com.rm[com.nx + 1, iy, [1, 2, 4, 3, 1]], com.zm[com.nx + 1, iy, [1, 2, 4, 3, 1]])) 
                 for iy in np.arange(0, com.ny + 2)])
    bdry.extend([list(zip(com.rm[ix, 0, [1, 2, 4, 3, 1]], com.zm[ix, 0, [1, 2, 4, 3, 1]])) 
                 for ix in np.arange(0, com.nx + 2)])
    bdry.extend([list(zip(com.rm[ix, com.ny + 1, [1, 2, 4, 3, 1]], com.zm[ix, com.ny + 1, [1, 2, 4, 3, 1]])) 
                 for ix in np.arange(0, com.nx + 2)])
    
    args = {'width': 0.001, 'alpha': 1}
    ylim = [-1.62, -1.3]  # Custom y-limits
    xlim = [0.35, 0.9]  # Custom x-limits
    sepwidth = 0.5
    vtot = (bbb.v2ce[:, :, 0]**2 + bbb.vyce[:, :, 0]**2)**0.5 + \
           (bbb.v2cb[:, :, 0]**2 + bbb.vycb[:, :, 0]**2)**0.5 + \
           (bbb.v2dd[:, :, 0]**2 + bbb.vydd[:, :, 0]**2)**0.5

    plt.figure(figsize=(6, 12))  # Optional: adjust figure size

    # Total Drift
    plt.subplot(311)
    plt.title('Sum of all drifts')
    plt.gca().add_collection(LineCollection(bdry, linewidths=0.2, color='black', zorder=0))
    plt.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], c='red', lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2[:, :, 0], bbb.vy[:, :, 0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2[:, :, 0], bbb.vy[:, :, 0], ix, iy))
    plt.quiver(com.rm[1:-1, 1:-1, 0], com.zm[1:-1, 1:-1, 0], dR[1:-1, 1:-1], dZ[1:-1, 1:-1], **args)
    plt.axis('equal')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel('Z [m]')
    plt.xticks([])  # No x-ticks

    # E   B Drift
    plt.subplot(312)
    percent = np.mean(analysis.nonGuard((bbb.v2ce[:, :, 0]**2 + bbb.vyce[:, :, 0]**2)**0.5 / vtot)) * 100
    plt.title(r'$\mathbf{E}\times \mathbf{B}$ drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=0.2, color='black', zorder=0))
    plt.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], c='red', lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2ce[:, :, 0], bbb.vyce[:, :, 0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2ce[:, :, 0], bbb.vyce[:, :, 0], ix, iy))
    plt.quiver(com.rm[1:-1, 1:-1, 0], com.zm[1:-1, 1:-1, 0], dR[1:-1, 1:-1], dZ[1:-1, 1:-1], **args)
    plt.axis('equal')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel('Z [m]')
    plt.xticks([])  # No x-ticks

    # B   ?B Drift
    plt.subplot(313)
    percent = np.mean(analysis.nonGuard((bbb.v2cb[:, :, 0]**2 + bbb.vycb[:, :, 0]**2)**0.5 / vtot)) * 100
    plt.title(r'$\mathbf{B}\times\nabla B$ drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=0.2, color='black', zorder=0))
    plt.plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], c='red', lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2cb[:, :, 0], bbb.vycb[:, :, 0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2cb[:, :, 0], bbb.vycb[:, :, 0], ix, iy))
    plt.quiver(com.rm[1:-1, 1:-1, 0], com.zm[1:-1, 1:-1, 0], dR[1:-1, 1:-1], dZ[1:-1, 1:-1], **args)
    plt.axis('equal')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')

    plt.tight_layout()  # Adjust spacing between subplots for cleaner look
    plt.savefig('Drfts_all.png', dpi=300)


def plotDrifts_EBgradB(patches): 
    # Boundary setup (unchanged)
    bdry = []
    bdry.extend([list(zip(com.rm[0, iy, [1, 2, 4, 3, 1]], com.zm[0, iy, [1, 2, 4, 3, 1]])) 
                 for iy in np.arange(0, com.ny + 2)])
    bdry.extend([list(zip(com.rm[com.nx + 1, iy, [1, 2, 4, 3, 1]], com.zm[com.nx + 1, iy, [1, 2, 4, 3, 1]])) 
                 for iy in np.arange(0, com.ny + 2)])
    bdry.extend([list(zip(com.rm[ix, 0, [1, 2, 4, 3, 1]], com.zm[ix, 0, [1, 2, 4, 3, 1]])) 
                 for ix in np.arange(0, com.nx + 2)])
    bdry.extend([list(zip(com.rm[ix, com.ny + 1, [1, 2, 4, 3, 1]], com.zm[ix, com.ny + 1, [1, 2, 4, 3, 1]])) 
                 for ix in np.arange(0, com.nx + 2)])
    
    args = {'width': 0.001, 'alpha': 1}
    ylim = [-1.62, -1.2]  # Custom y-limits
    xlim = [0.3, 0.9]  # Custom x-limits
    sepwidth = 0.5
    vtot = (bbb.v2ce[:, :, 0]**2 + bbb.vyce[:, :, 0]**2)**0.5 + \
           (bbb.v2cb[:, :, 0]**2 + bbb.vycb[:, :, 0]**2)**0.5 + \
           (bbb.v2dd[:, :, 0]**2 + bbb.vydd[:, :, 0]**2)**0.5

    # Create subplots with shared y-axis
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), sharey=True)  # Adjusted for two subplots

    # E   B Drift
    percent = np.mean(analysis.nonGuard((bbb.v2ce[:, :, 0]**2 + bbb.vyce[:, :, 0]**2)**0.5 / vtot)) * 100
    axs[0].set_title(r'$\mathbf{E}\times \mathbf{B}$ drift (%.2g%%)' % percent)
    axs[0].add_collection(LineCollection(bdry, linewidths=0.5, color='black', zorder=0))
    axs[0].plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], c='red', lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2ce[:, :, 0], bbb.vyce[:, :, 0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2ce[:, :, 0], bbb.vyce[:, :, 0], ix, iy))
    axs[0].quiver(com.rm[1:-1, 1:-1, 0], com.zm[1:-1, 1:-1, 0], dR[1:-1, 1:-1], dZ[1:-1, 1:-1], **args)
    axs[0].axis('equal')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].set_ylabel('Z [m]')
    axs[0].set_xticks([]) 

    # B   ?B Drift
    percent = np.mean(analysis.nonGuard((bbb.v2cb[:, :, 0]**2 + bbb.vycb[:, :, 0]**2)**0.5 / vtot)) * 100
    axs[1].set_title(r'$\mathbf{B}\times\nabla B$ drift (%.2g%%)' % percent)
    axs[1].add_collection(LineCollection(bdry, linewidths=0.5, color='black', zorder=0))
    axs[1].plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], c='red', lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2cb[:, :, 0], bbb.vycb[:, :, 0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2cb[:, :, 0], bbb.vycb[:, :, 0], ix, iy))
    axs[1].quiver(com.rm[1:-1, 1:-1, 0], com.zm[1:-1, 1:-1, 0], dR[1:-1, 1:-1], dZ[1:-1, 1:-1], **args)
    axs[1].axis('equal')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)
    axs[1].set_xlabel('R [m]')
    axs[1].set_ylabel('Z [m]')

    plt.tight_layout()  # Adjust spacing between subplots for cleaner look
    plt.savefig('Drfts_Eb_gradB.png', dpi=300)


def plotDrifts_EBgradB_2(patches): 
    # --- Boundary setup ---
    bdry = []
    bdry.extend([list(zip(com.rm[0, iy, [1, 2, 4, 3, 1]], com.zm[0, iy, [1, 2, 4, 3, 1]])) 
                 for iy in np.arange(0, com.ny + 2)])
    bdry.extend([list(zip(com.rm[com.nx + 1, iy, [1, 2, 4, 3, 1]], com.zm[com.nx + 1, iy, [1, 2, 4, 3, 1]])) 
                 for iy in np.arange(0, com.ny + 2)])
    bdry.extend([list(zip(com.rm[ix, 0, [1, 2, 4, 3, 1]], com.zm[ix, 0, [1, 2, 4, 3, 1]])) 
                 for ix in np.arange(0, com.nx + 2)])
    bdry.extend([list(zip(com.rm[ix, com.ny + 1, [1, 2, 4, 3, 1]], com.zm[ix, com.ny + 1, [1, 2, 4, 3, 1]])) 
                 for ix in np.arange(0, com.nx + 2)])

    # --- Axis limits ---
    ylim = [-1.62, -1.2]
    xlim = [0.3, 0.9]
    sepwidth = 0.7

    # --- Normalization for vectors ---
    def normalize(u, v):
        mag = np.sqrt(u**2 + v**2) + 1e-12
        return u/mag, v/mag, mag

    # --- Figure setup ---
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharey=True)

    # === 1. E × B drift ===
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2ce[:, :, 0], bbb.vyce[:, :, 0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2ce[:, :, 0], bbb.vyce[:, :, 0], ix, iy))
    dR_norm, dZ_norm, mag = normalize(dR, dZ)

    axs[0].set_title(r'$\mathbf{E}\times\mathbf{B}$ drift', fontsize=14)
    axs[0].add_collection(LineCollection(bdry, linewidths=0.5, color='black', zorder=0))
    axs[0].plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], 
                c='red', lw=sepwidth)

    q = axs[0].quiver(com.rm[1:-1, 1:-1, 0], com.zm[1:-1, 1:-1, 0], 
                      dR_norm[1:-1, 1:-1], dZ_norm[1:-1, 1:-1], 
                      mag[1:-1, 1:-1], cmap="viridis",
                      scale=25, width=0.004, alpha=0.9)

    axs[0].axis('equal')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].set_ylabel(r'$Z$ [m]', fontsize=12)

    # === 2. B × ∇B drift ===
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2cb[:, :, 0], bbb.vycb[:, :, 0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2cb[:, :, 0], bbb.vycb[:, :, 0], ix, iy))
    dR_norm, dZ_norm, mag = normalize(dR, dZ)

    axs[1].set_title(r'$\mathbf{B}\times\nabla B$ drift', fontsize=14)
    axs[1].add_collection(LineCollection(bdry, linewidths=0.5, color='black', zorder=0))
    axs[1].plot(com.rm[:, com.iysptrx + 1, 2], com.zm[:, com.iysptrx + 1, 2], 
                c='red', lw=sepwidth)

    axs[1].quiver(com.rm[1:-1, 1:-1, 0], com.zm[1:-1, 1:-1, 0], 
                  dR_norm[1:-1, 1:-1], dZ_norm[1:-1, 1:-1], 
                  mag[1:-1, 1:-1], cmap="viridis",
                  scale=25, width=0.004, alpha=0.9)

    axs[1].axis('equal')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)
    axs[1].set_xlabel(r'$R$ [m]', fontsize=12)
    axs[1].set_ylabel(r'$Z$ [m]', fontsize=12)

    # --- Colorbar ---
    #cbar = fig.colorbar(q, ax=axs, orientation='vertical', fraction=0.025, pad=0.02)
    #cbar.set_label('Normalized drift magnitude', fontsize=12)

    # --- Layout and save ---
    plt.tight_layout()
    plt.savefig('Drifts_EB_gradB.png', dpi=600, bbox_inches='tight')
    plt.close()


def finishPage(pdf):
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    
def toPlate(arr):
    """
    Poloidal flux in direction of closest divertor plate, through the face
    nearest that plate.
    
    Arguments
        arr: numpy array with dimension (nx, ny)
    """
    arr[:bbb.ixmp, :] *= -1
    if ix <= bbb.ixmp - 1:
        sign = -1
        ix = ix - 1
    else:
        sign = 1
    return sign*arr[ix, ]


def plotall(savefile='plots', plotV0=True):    
    savefile = savefile + '.pdf'
    plt.close()
    plt.rcParams['font.size'] = 12
    
    targetDataFiles = glob.glob("targetData_*")
    if targetDataFiles:
        h5 = h5py.File(targetDataFiles[0], 'r')
        plotV0 = False
    else:
        h5 = None
    
    # Calculate photon power fluxes (required for some plots)
    bbb.pradpltwl()
    
    patches = getPatches()
    
    with PdfPages(savefile) as pdf:
        # Page 1: input text and chi, D graphs
        fig = plt.figure(figsize=(8.5, 11))
        txt = getConfigText()
        plt.subplot(311)
        plt.axes(frameon=False)
        fig.text(0.1, 0.95, txt, transform=fig.transFigure, size=9, horizontalalignment='left', verticalalignment='top')
        plotTransportCoeffs(patches)
        finishPage(pdf)
        
        # Page 2: n and T 2D plots
        plt.figure(figsize=(8.5, 11))
        plot2Dvars(patches)
        finishPage(pdf)
        
        # Page 3: n and T line profiles
        plt.figure(figsize=(8.5, 11))
        plotnTprofiles(plotV0, h5)
        pdf.savefig()
        plt.close()
        # finishPage(pdf)
        
        # Page 4: n, T, Prad going down legs
        plt.figure(figsize=(8.5, 11))
        plotAlongLegs()
        finishPage(pdf)
        
        # Page 5: pressure, lamda_q
        plt.figure(figsize=(8.5, 11))
        plotPressures()
        plotqFits(h5)
        finishPage(pdf)

        # Page 6: power breakdown
        fig = plt.figure(figsize=(8.5, 11))
        plotPowerBreakdown()
        plotPowerSurface()
        finishPage(pdf)
        
        # Page 7: radiation balance 2D plots
        plt.figure(figsize=(8.5, 11))
        plotPowerBalance(patches)
        finishPage(pdf)
        
        # Page 8: density balance 2D plots
        plt.figure(figsize=(8.5, 11))
        plotDensityBalance(patches)
        finishPage(pdf)
        
        # Page 9: Density and power flux at midplane
        plt.figure(figsize=(8.5, 11))
        plotRadialFluxes()
        finishPage(pdf)
        
        # Page 10: drifts
        if bbb.cfjve+bbb.jhswitch+bbb.isfdiax+bbb.cfyef+bbb.cf2ef+bbb.cfybf+bbb.cf2bf+bbb.cfqybf+bbb.cfq2bf > 0:
            plt.figure(figsize=(8.5, 11))
            plotDrifts(patches)
            finishPage(pdf)

        d = pdf.infodict()
        d['Title'] = savefile
        d['CreationDate'] = datetime.datetime.today()
    
    plt.close('all')




def plot_eich_exp_shahinul_final(omp=False, save_prefix='lambdaq_result'):
  
    fx = analysis.calculate_flux_expansion()
    fx = np.round(fx)
    print("fx", fx)

    # === Select which q_parallel to fit (OMP vs ODIV)
    bbb.plateflux()
    ppar = analysis.Pparallel()
    rrf = analysis.getrrf()
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

    # === Plot: Exponential ===
    plt.figure(figsize=(4.5, 2.75))
    plt.plot(xq, qparo / 1e9, '*', markersize=8, c='blue', label=r'UEDGE')
    if qofit is not None:
        plt.plot(xq[omax:], expfun(xq[omax:], *qofit) / 1e9, c='red', ls='--',
                 label=r'Exp Fit: $\lambda_q$ = %.2f mm' % lqo)
    plt.xlabel(r'$r_{omp} - r_{sep}$ (m)', fontsize=16)
    plt.ylabel(r'$q_\parallel^{OMP}$ (GW/m$^2$)' if omp else r'$q_\parallel^{Odiv}$ (GW/m$^2$)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, np.max(qparo / 1e9) * 1.2])
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_expfit.png", dpi=300)
    plt.show()
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

        # R 
        residuals = q_fit - q_fit_full
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((q_fit - np.mean(q_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    except Exception as e:
        print("Eich fit failed:", e)
        popt = [0, 0, 0, 0]
        lambda_q_fit = 0.0
        r_squared = 0.0

    # === Plot: Eich Fit ===
 

    plt.figure(figsize=(4.5, 2.75))
    plt.plot(s_omp, q_perp_odiv / 1e6, 'ko', label='UEDGE')
    if popt[2] > 0:
        plt.plot(s_fit, q_fit_full / 1e6, 'r--', label=f'Eich Fit: $\lambda_q$  = {lambda_q_fit * 1000:.2f} mm')
    plt.xlabel(r'$r_{div} - r_{sep}$ (m)', fontsize=16)
    plt.ylabel(r'$q_\perp^{Odiv}$ (MW/m$^2$)', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_eichfit.png", dpi=300)
    plt.show()

    # === Combined Subplot ===
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 5.5), sharex=False)

    axs[0].plot(xq_omp, qparo / 1e9, '*', c='blue', label='UEDGE')
    if qofit is not None:
        axs[0].plot(xq_omp[omax:], expfun(xq[omax:], *qofit) / 1e9, 'r--', label=f' $\lambda_q$  = {lqo:.2f} mm')
    axs[0].set_ylabel(r'$q_\parallel$ (GW/m$^2$)', fontsize=12)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(s_omp, q_perp_odiv / 1e6, 'ko', label='UEDGE')
    if popt[2] > 0:
        axs[1].plot(s_fit, q_fit_full / 1e6, 'r--', label=f' $\lambda_q$  = {lambda_q_fit * 1e3:.2f} mm')
    axs[1].set_xlabel(r'$r - r_{sep}$ (m)', fontsize=12)
    axs[1].set_ylabel(r'$q_\perp$ (MW/m$^2$)', fontsize=12)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_subplots.png", dpi=300)
    plt.show()

    # === Print Summary ===
    print("\n--- Fit Summary ---")
    print(f"Exponential ?q (parallel): {lqo:.2f} mm")
    print(f"Eich ?q (perpendicular):  {lambda_q_fit * 1000:.2f} mm")
    print(f"R^2 (Eich Fit): {r_squared:.4f}")  



    return {
        'lambdaq_exp_mm': lqo,
        'lambdaq_eich_mm': lambda_q_fit * 1000,
        'r_squared': r_squared,
        'S_fit_mm': S_fit * 1000 if lambda_q_fit > 0 else None,
        'q0_fit': q0_fit,
        's0_fit': s0_fit
    }


def plt_eich_exp_shahinul_in_out_final(omp=False, save_prefix='lambdaq_result'):
    bbb.plateflux()
    ppar = Pparallel()
    rrf = getrrf()

    fx_out = (com.bpol[bbb.ixmp, :, 0] * com.rm[bbb.ixmp, :, 0]) / \
             (com.bpol[com.nx, :, 0] * com.rm[com.nx, :, 0])
    fx_out = np.round(fx_out[com.iysptrx + 1])

    fx_in = (com.bpol[bbb.ixmp, :, 0] * com.rm[bbb.ixmp, :, 0]) / \
            (com.bpol[0, :, 0] * com.rm[0, :, 0])
    fx_in = np.round(fx_in[com.iysptrx + 1])

    q_para_omp = ppar[bbb.ixmp, :-1] / com.sx[bbb.ixmp, :-1] / rrf[bbb.ixmp, :-1]
    q_para_odiv = ppar[com.nx, :-1] / com.sxnp[com.nx, :-1] / rrf[com.nx, :-1]
    q_para_idiv = abs(ppar[0, :-1]) / com.sxnp[0, :-1] / rrf[0, :-1]

    q_data = bbb.sdrrb + bbb.sdtrb
    q_data_idiv = bbb.sdrlb + bbb.sdtlb
    q_perp = q_data.reshape(-1)[:-1]
    q_perp_idiv = q_data_idiv.reshape(-1)[:-1]

    yyc = com.yyc.reshape(-1)[:-1]
    yyrb = com.yyrb.reshape(-1)[:-1]
    yylb = com.yylb.reshape(-1)[:-1]
    iy_sep = com.iysptrx + 1

    s_odiv = yyrb[iy_sep:]
    q_perp_odiv = q_perp[iy_sep:]
    s_idiv = yylb[iy_sep:]
    q_perp_idiv = q_perp_idiv[iy_sep:]

    def expfit(xq, qpara):
        try:
            imax = np.argmax(qpara)
            expfun = lambda x, A, lamda_q_inv: A * np.exp(-x * lamda_q_inv)
            qofit, _ = curve_fit(expfun, xq[imax:], qpara[imax:], p0=[np.max(qpara), 1000], bounds=(0, np.inf))
            return 1000 / qofit[1], expfun, qofit
        except Exception as e:
            print('Exponential fit failed:', e)
            return 1.0, None, None

    def eichFunction(x, S, lq, q0, s0, fx_val):
        sBar = x - s0
        t0 = (S / (2 * lq * fx_val)) ** 2
        t1 = sBar / (lq * fx_val)
        t2 = S / (2 * lq * fx_val)
        t3 = sBar / S
        return (q0 / 2) * np.exp(t0 - t1) * erfc(t2 - t3)

    def eich_fit_wrapper(s, q, fx_val):
        try:
            interp_fun = interp1d(s, q, kind='cubic', fill_value="extrapolate")
            s_interp = np.linspace(s.min(), s.max(), 300)
            q_interp = interp_fun(s_interp)
            s0_guess = np.median(s_interp)
            q0_guess = np.max(q_interp)
            p0 = [0.003, 0.002, q0_guess, s0_guess]
            bounds = ([0.0005, 0.0005, 1e6, s_interp.min() - 0.01],
                      [0.02, 0.02, 1e9, s_interp.max() + 0.01])
            popt, _ = curve_fit(lambda x, S, lq, q0, s0: eichFunction(x, S, lq, q0, s0, fx_val),
                                s_interp, q_interp, p0=p0, bounds=bounds, maxfev=10000)
            return popt[1] * 1000, eichFunction, s_interp, q_interp, popt
        except Exception as e:
            print("Eich fit failed:", e)
            return 0.0, None, None, None, None

    # === Fits ===
    lqo, expfun_o, expfit_o = expfit(yyc[iy_sep:-1], q_para_odiv[iy_sep:-1])
    lqi, expfun_i, expfit_i = expfit(yyc[:iy_sep], q_para_idiv[:iy_sep])

    lambdaq_eich_out_mm, eichfun_o, s_o, q_o, popt_o = eich_fit_wrapper(s_odiv, q_perp_odiv, fx_out)
    lambdaq_eich_in_mm, eichfun_i, s_i, q_i, popt_i = eich_fit_wrapper(s_idiv, q_perp_idiv, fx_in)

    # === Plot ===
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axs = axs.flatten()

    # Top-left: q_parallel Outer
    axs[0].set_title("q? Outer Divertor (vs yyc)")
    axs[0].plot(yyc[iy_sep:-1], q_para_odiv[iy_sep:-1], 'k-', label='q?')
    if expfun_o:
        axs[0].plot(yyc[iy_sep:-1], expfun_o(yyc[iy_sep:-1], *expfit_o), 'b--',
                    label=f'Exp Fit\n?q={lqo:.2f} mm')
    axs[0].set_ylabel(r'$q_{\parallel}$ [W/m$^2$]')
    axs[0].legend()
    axs[0].grid(True)

    # Top-right: q_parallel Inner
    axs[1].set_title("q? Inner Divertor (vs yyc)")
    axs[1].plot(yyc[:iy_sep], q_para_idiv[:iy_sep], 'k-', label='q?')
    if expfun_i:
        axs[1].plot(yyc[:iy_sep], expfun_i(yyc[:iy_sep], *expfit_i), 'b--',
                    label=f'Exp Fit\n?q={lqi:.2f} mm')
    axs[1].legend()
    axs[1].grid(True)

    # Bottom-left: q_perp Outer
    axs[2].set_title("q? Outer Divertor (vs yyrb)")
    axs[2].plot(s_o, q_o, 'k--', label='q?')
    if eichfun_o:
        axs[2].plot(s_o, eichfun_o(s_o, *popt_o, fx_out), 'r-', label=f'Eich Fit\n?q={lambdaq_eich_out_mm:.2f} mm')
    axs[2].set_xlabel('s (m)')
    axs[2].set_ylabel(r'$q_{\perp}$ [W/m$^2$]')
    axs[2].legend()
    axs[2].grid(True)

    # Bottom-right: q_perp Inner
    axs[3].set_title("q? Inner Divertor (vs yylb)")
    axs[3].plot(s_i, q_i, 'k--', label='q?')
    if eichfun_i:
        axs[3].plot(s_i, eichfun_i(s_i, *popt_i, fx_in), 'r-', label=f'Eich Fit\n?q={lambdaq_eich_in_mm:.2f} mm')
    axs[3].set_xlabel('s (m)')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300)
    plt.close()

    return lqo, lqi, lambdaq_eich_out_mm, lambdaq_eich_in_mm


def plot_surface_recombination(x_target=None):
    """
    Plot surface recombination heat flux and total power for outer or inner divertor.
    Uses com.yyrb for outer target and com.yylb for inner target radial coordinates.
    Multiplies coordinates and fluxes by xsign for consistent orientation.
    """
    eV_to_J = 1.602e-19

    I_D  = bbb.ebind
    I0, I1, I2 = 5.39, 75.6, 122.4
    E_Li1, E_Li2, E_Li3 = I0, I0+I1, I0+I1+I2

    if x_target is None:
        x_target = com.nx  # outer target default

    # Determine xsign based on target
    xsign = +1 if x_target == com.nx else -1

    # Ion indices
    idx_D   = 0
    idx_Li1 = com.nhsp + 0
    idx_Li2 = com.nhsp + 1
    idx_Li3 = com.nhsp + 2

    # Select correct coordinate array
    if xsign > 0:
        y_pos = com.yyrb[1:-1]
    else:
        y_pos = com.yylb[1:-1]

    y_pos = xsign * y_pos  # apply sign to position

    area = com.sxnp[x_target, 1:-1]

    # Fluxes (already signed properly in fnix usually, but apply xsign to be consistent)
    Gamma_D   = xsign * bbb.fnix[x_target, 1:-1, idx_D]
    Gamma_Li1 = xsign * bbb.fnix[x_target, 1:-1, idx_Li1]
    Gamma_Li2 = xsign * bbb.fnix[x_target, 1:-1, idx_Li2]
    Gamma_Li3 = xsign * bbb.fnix[x_target, 1:-1, idx_Li3]

    P_D   = Gamma_D   * I_D   * eV_to_J
    P_Li1 = Gamma_Li1 * E_Li1 * eV_to_J
    P_Li2 = Gamma_Li2 * E_Li2 * eV_to_J
    P_Li3 = Gamma_Li3 * E_Li3 * eV_to_J

    P_total = {
    	 'D+':  np.sum(P_D),
   	  'Li+': np.sum(P_Li1),
    	 'Li2+': np.sum(P_Li2),
    	 'Li3+': np.sum(P_Li3)
      			  }

    q_D_MW   = P_D   / area * 1e-6
    q_Li1_MW = P_Li1 / area * 1e-6
    q_Li2_MW = P_Li2 / area * 1e-6
    q_Li3_MW = P_Li3 / area * 1e-6

    plt.figure(figsize=(5,3))
    plt.plot(y_pos, q_D_MW, label=r'D$^+$ $\rightarrow$  D$^0$', color='green', linestyle='-', linewidth=2)
    plt.plot(y_pos, q_Li1_MW, label=r'Li$^+$ $\rightarrow$ Li$^0$', color='blue', linestyle='--',linewidth=2)
    plt.plot(y_pos, q_Li2_MW, label=r'Li$^{2+}$ $\rightarrow$  Li$^0$', color='orange', linestyle=':', linewidth=2)
    plt.plot(y_pos, q_Li3_MW, label=r'Li$^{3+}$ $\rightarrow$  Li$^0$', color='red', linestyle='-.',linewidth=2)
    plt.xlabel("Target vertical position [m]", fontsize=12)
    plt.ylabel("q$_{recombincation}$ [MW/$m^2$]", fontsize=12)
    plt.title(f"Surface Recombination - {'Outer' if xsign > 0 else 'Inner'} Target", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    ymax = max(np.max(q_D_MW), np.max(q_Li1_MW), np.max(q_Li2_MW), np.max(q_Li3_MW))
    plt.ylim(0, ymax * 1.05)
    plt.tight_layout()
    plt.savefig('P_rec.png', dpi=300)
    plt.show()

    species = list(P_total.keys())
    powers = list(P_total.values())
    colors = ['green', 'blue', 'orange', 'red']

    plt.figure(figsize=(5,4))
    plt.bar(species, powers, color=colors)
    plt.ylabel("Integrated recombination power [W]", fontsize=12)
    plt.title(f"Total Surface Recombination Power - {'Outer' if xsign > 0 else 'Inner'} Target", fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('P_rec_pie.png', dpi=300)
    plt.show()

    print(f"=== Total Surface Recombination Power on {'Outer' if xsign > 0 else 'Inner'} Target ===")
    for sp, pw in P_total.items():
        print(f"{sp:5s}: {pw:.2f} W")
    print(f"Total : {sum(powers):.2f} W")







def plot_surface_heatflux_components(
    target='outer',
    log='true',
    show_titles=False,
    legend_fontsize=8,
    axis_label_fontsize=14,
    labelsize_font=12
):
    """
    Plot surface heat flux components and surface recombination breakdown
    for inner or outer divertor. Default is outer.

    Parameters:
    -----------
    target : str
        'outer' (default) or 'inner'
    log : str or bool
        'true' (default) or 'false', or boolean True/False
    show_titles : bool
        If True, show subplot titles. If False, omit titles (default)
    legend_fontsize : int
        Font size for the legend (default 8)
    axis_label_fontsize : int
        Font size for x and y axis labels (default 14)
    labelsize_font : int
        Font size for tick labels (default 12)
    """
    import matplotlib.pyplot as plt
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
        side_label = 'Outer target'
    elif target == 'inner':
        xtarget = 0
        plateIndex = 0
        xsign = -1
        y_pos = yylb
        q_data_target = q_data_idiv
        side_label = 'Inner target'
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

    ion_ke = xsign * analysis.PionParallelKE()[xtarget, 1:-1] / area / 1e6
    h_photons = bbb.pwr_plth[1:-1, plateIndex] / 1e6
    imp_photons = bbb.pwr_pltz[1:-1, plateIndex] / 1e6
    total_flux = conv_cond + surf_recomb + ion_ke + h_photons + imp_photons

    # Log logic
    if isinstance(log, str):
        log = log.lower() == 'true'

    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # --- First subplot ---
    axs[0].plot(y_pos, conv_cond, label='Conv. + Cond.', linestyle='-', color='blue', linewidth=1.5)
    axs[0].plot(y_pos, q_D_MW, label='Surface recomb.', linestyle='--', color='green', linewidth=1.5)
    axs[0].plot(y_pos, ion_ke, label='Ion KE', linestyle='-.', color='black', linewidth=1.5)
    axs[0].plot(y_pos, h_photons, label='H photons', linestyle=':', color='purple', linewidth=1.5)
    axs[0].plot(y_pos, imp_photons, label='Imp. photons', linestyle='-', color='red', linewidth=1.5)
    axs[0].plot(y_pos, total_flux, 'm-', lw=2, label='Total')

    if log:
        axs[0].set_yscale('log')
        axs[0].set_ylim([1e-3, 10])
    else:
        axs[0].set_ylim([0, np.max(total_flux)*1.05])
        axs[0].set_ylim([0, 10])

    axs[0].set_ylabel(r'$q_{\perp}$ [MW/m$^2$]', fontsize=axis_label_fontsize)
    if show_titles:
        axs[0].set_title(f'Surface Heat Flux Components: {side_label}')
    axs[0].legend(fontsize=legend_fontsize, ncol=2)
    axs[0].grid(True, which='both', axis='both', linestyle=':', color='gray')
    axs[0].set_axisbelow(True)
    axs[0].text(0.02, 0.95, '(a)', transform=axs[0].transAxes, fontsize=14, va='top', ha='left', fontweight='bold')
    axs[0].tick_params(axis='both', labelsize=labelsize_font)

    # --- Second subplot ---
    axs[1].plot(y_pos, q_D_MW, label=r'D$^+$ $\rightarrow$ D', color='blue', linestyle='-')
    axs[1].plot(y_pos, q_Li1_MW, label=r'Li$^+$ $\rightarrow$ Li', color='green', linestyle='--')
    axs[1].plot(y_pos, q_Li2_MW, label=r'Li$^{2+}$ $\rightarrow$ Li', color='orange', linestyle='-.')
    axs[1].plot(y_pos, q_Li3_MW, label=r'Li$^{3+}$ $\rightarrow$ Li', color='red', linestyle=':')

    axs[1].set_xlabel(r'r$_{div}$ - r$_{sep}$ (m)', fontsize=axis_label_fontsize)
    axs[1].set_ylabel(r'q$_{recomb.}$ [MW/m$^2$]', fontsize=axis_label_fontsize)
    if log:
        axs[1].set_yscale('log')
        axs[1].set_ylim([1e-5, 2])
    else:
        axs[1].set_ylim([0, 2])

    if show_titles:
        axs[1].set_title('Surface Recombination Breakdown')
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].grid(True, which='both', axis='both', linestyle=':', color='gray')
    axs[1].set_axisbelow(True)
    axs[1].text(0.02, 0.95, '(b)', transform=axs[1].transAxes, fontsize=14, va='top', ha='left', fontweight='bold')
    axs[1].tick_params(axis='both', labelsize=labelsize_font)

    plt.tight_layout()
    plt.savefig(f'q_surface_{target}.png', dpi=300)
    plt.show()

def plot_Li_flux_omp(fontsize=14, abs_val=True):
    Gamma_Li1 = bbb.fnix[bbb.ixmp,1:-1,2] / com.sx[bbb.ixmp,1:-1]
    Gamma_Li2 = bbb.fnix[bbb.ixmp,1:-1,3] / com.sx[bbb.ixmp,1:-1]
    Gamma_Li3 = bbb.fnix[bbb.ixmp,1:-1,4] / com.sx[bbb.ixmp,1:-1]
    Gamma_Li = Gamma_Li1 + Gamma_Li2 + Gamma_Li3

    if abs_val:
        Gamma_Li = np.abs(Gamma_Li)
        ylim = [0, np.max(Gamma_Li)*1.05]
    else:
        ylim = [np.min(Gamma_Li)*1.05, np.max(Gamma_Li)*1.05]

    plt.figure(figsize=(5,3))
    plt.plot(com.yyc[1:-1], Gamma_Li)
    plt.ylabel('$\Gamma_{Li}^{OMP}$ (/m$^2$s)', fontsize=fontsize)
    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=fontsize)
    plt.ylim(ylim)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('Li_flux.png', dpi=300)

def plot_uedge_grid_resolution(rm, zm, separatrix_index=None, titlesuffix=""):
    """
    Plot 2D heatmaps of ?R and ?Z spatial resolution from UEDGE RZ grid.

    Parameters:
    -----------
    rm : ndarray
        2D array of radial cell centers (com.rm[:,:,0]), shape (nx+1, ny+1)
    zm : ndarray
        2D array of vertical cell centers (com.zm[:,:,0]), shape (nx+1, ny+1)
    separatrix_index : int, optional
        Radial index of separatrix to mark on the plot
    titlesuffix : str
        Optional string to append to plot title
    """
    # Compute ?R (difference in R between adjacent radial points)
    dR = rm[1:, :] - rm[:-1, :]  # shape (nx, ny+1)
    dR = dR[:, 1:-1]  # remove poloidal ghost cells ? shape (nx, ny)

    # Compute ?Z (difference in Z between adjacent poloidal points)
    dZ = zm[:, 1:] - zm[:, :-1]  # shape (nx+1, ny)
    dZ = dZ[1:-1, :]  # remove radial ghost cells ? shape (nx, ny)

    # --- Plotting ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    im0 = axs[0].imshow(dR.T * 1e3, origin='lower', aspect='auto', cmap='viridis')
    axs[0].set_title(f'?R Resolution (mm) {titlesuffix}')
    axs[0].set_xlabel('Radial Cell Index')
    axs[0].set_ylabel('Poloidal Cell Index')
    if separatrix_index:
        axs[0].axvline(separatrix_index - 1, color='r', linestyle='--')

    im1 = axs[1].imshow(dZ.T * 1e3, origin='lower', aspect='auto', cmap='plasma')
    axs[1].set_title(f'?Z Resolution (mm) {titlesuffix}')
    axs[1].set_xlabel('Radial Cell Index')
    if separatrix_index:
        axs[1].axvline(separatrix_index - 1, color='r', linestyle='--')

    cbar0 = plt.colorbar(im0, ax=axs[0])
    cbar0.set_label('?R (mm)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axs[1])
    cbar1.set_label('?Z (mm)', fontsize=12)

    plt.suptitle("UEDGE Grid Spatial Resolution", fontsize=14)
    plt.tight_layout()
    plt.show()

def get_q_drifts():
    """Get the ExB and grad B convective heat fluxes. Outputs have dimensions [com.nx+2,com.ny+2,2], where the third dimension contains the x and y components of the vector (in UEDGE coordinates)

    :return: q_ExB, q_gradB
    """
    # Compute the heat fluxes
    q_ExB = np.zeros((com.nx + 2, com.ny + 2, 2))
    q_gradB = np.zeros((com.nx + 2, com.ny + 2, 2))
    p = 2.5*(bbb.ne * bbb.te) + 2.5*(bbb.ni[:, :, 0] * bbb.ti)
    q_ExB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2ce[:, :, 0] * p
    q_ExB[:, :, 1] = bbb.vyce[:, :, 0] * p
    q_gradB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2cb[:, :, 0] * p
    q_gradB[:, :, 1] = bbb.vycb[:, :, 0] * p

    return q_ExB, q_gradB


def get_gamma_drifts():
    """Get the ExB and grad B convective particle fluxes. Outputs have dimensions [com.nx+2,com.ny+2,2], where the third dimension contains the x and y components of the vector (in UEDGE coordinates)

    :return: gamma_ExB, gamma_gradB
    """
    gamma_ExB = np.zeros((com.nx + 2, com.ny + 2, 2))
    gamma_gradB = np.zeros((com.nx + 2, com.ny + 2, 2))
    p = bbb.ni[:, :, 0]
    gamma_ExB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2ce[:, :, 0] * p
    gamma_ExB[:, :, 1] = bbb.vyce[:, :, 0] * p
    gamma_gradB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2cb[:, :, 0] * p
    gamma_gradB[:, :, 1] = bbb.vycb[:, :, 0] * p

    return gamma_ExB, gamma_gradB


def plot_gamma_streamlines(plot_which='both', cmap='viridis', density=1.2, linewidth=1.5):
 
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    q_ExB, q_gradB = get_gamma_drifts()
    omp_cell = bbb.ixmp  # OMP poloidal cell index (guard cells excluded assumed)
    separatrix_cell = com.iysptrx   # separatrix cell number
    outer_wall_cell = com.ny        # outer wall cell number
    oxpt = com.ixpt2[0]
    ixpt = com.ixpt1[0]

    def plot_single(ax, q_vec, title):
        U = q_vec[1:-1, 1:-1, 0].T
        V = q_vec[1:-1, 1:-1, 1].T
        ny, nx = U.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        magnitude = (np.sqrt(U**2 + V**2))/1e22  # MW/m^2
        vmax = np.nanmax(magnitude)
        vmin = np.nanmin(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1e-2
        norm = LogNorm(vmin=vmin, vmax=vmax)
        strm = ax.streamplot(X, Y, U, V, color=magnitude, cmap=cmap,
                             linewidth=linewidth, density=density, norm=norm)

        cbar = plt.colorbar(strm.lines, ax=ax, orientation='vertical', pad=0.01, aspect=30)
        cbar.set_label(r'$\Gamma$ [10$^{22}$ m$^{-2}s^{-1}$]', fontsize=13)
        strm.lines.set_clim(vmin, vmax)

        ax.set_title(title, pad=10)
        ax.set_ylabel('Radial cell', fontsize=13)

        # OMP, X-points
        if 0 <= omp_cell-1 < nx:
            ax.axvline(omp_cell - 1, color='red', linestyle='--', linewidth=2, label='OMP')
            ax.axvspan(omp_cell - 1 - 0.5, omp_cell - 1 + 0.5, color='red', alpha=0.08)
            ax.text(omp_cell - 1, ny, "OMP", color='red', fontsize=13, ha='center', va='bottom', fontweight='bold')

        if 0 <= oxpt-1 < nx:
            ax.axvline(oxpt - 1, color='orange', linestyle='--', linewidth=2, label='Outer X-point')
            ax.axvspan(oxpt - 1 - 0.5, oxpt - 1 + 0.5, color='orange', alpha=0.08)
            ax.text(oxpt - 1, -0.2, "Oxpt", color='orange', fontsize=13, ha='center', va='bottom', fontweight='bold')
        if 0 <= ixpt-1 < nx:
            ax.axvline(ixpt - 1, color='green', linestyle='--', linewidth=2, label='Inner X-point')
            ax.axvspan(ixpt - 1 - 0.5, ixpt - 1 + 0.5, color='green', alpha=0.08)
            ax.text(ixpt - 1, -0.2, "Ixpt", color='green', fontsize=13, ha='center', va='bottom', fontweight='bold')

        # Separatrix
        if 0 <= separatrix_cell-1 < ny:
            ax.axhline(separatrix_cell - 1, color='magenta', linestyle='-.', linewidth=2, label='Separatrix')
            ax.text(0, separatrix_cell - 1, "Sep", color='magenta', fontsize=13, ha='left', va='bottom', fontweight='bold')

        # Outer wall
        if 0 <= outer_wall_cell-1 < ny:
            ax.axhline(outer_wall_cell - 1, color='blue', linestyle=':', linewidth=2, label='Outer Wall')
            ax.text(nx+1 - com.nx, outer_wall_cell - 1, "Owall", color='blue', fontsize=13, ha='right', va='bottom', fontweight='bold')

        # Inner and outer divertor
        #ax.text(0, -0.08*ny, "Idiv", color='black', fontsize=13, ha='left', va='top', fontweight='bold')
        #ax.text(nx-1, -0.08*ny, "Odiv", color='black', fontsize=13, ha='right', va='top', fontweight='bold')

       
        ax.set_xlim(0, nx+1)
        ax.set_ylim(0, ny+1)
        ax.set_aspect('auto')

        # Remove duplicate legend entries
        #handles, labels = ax.get_legend_handles_labels()
        #by_label = dict(zip(labels, handles))
        #if by_label:
        #    ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, fontsize=12)

    plot_which = plot_which.lower()

    if plot_which == 'exb':
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_single(ax, q_ExB, r'$G_{ExB} = \frac{5}{2} P v_{ExB}$')
        ax.set_xlabel('Poloidal cell', fontsize=13)
        plt.tight_layout()
        plt.savefig('q_drifts_exb.png', dpi=600, bbox_inches='tight')
        plt.savefig('q_drifts_exb.pdf', bbox_inches='tight')
        plt.show()

    elif plot_which == 'gradb':
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_single(ax, q_gradB, r'$q_{\nabla B} = \frac{5}{2} P v_{\nabla B}$')
        ax.set_xlabel('Poloidal cell', fontsize=13)
        plt.tight_layout()
        plt.savefig('q_drifts_gradb.png', dpi=600, bbox_inches='tight')
        plt.savefig('q_drifts_gradb.pdf', bbox_inches='tight')
        plt.show()

    elif plot_which == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        plot_single(ax1, q_ExB, r'$\Gamma_{ExB} = n v_{ExB}$')
        plot_single(ax2, q_gradB, r'$\Gamma_{\nabla B} = n v_{\nabla B}$')
        ax2.set_xlabel('Poloidal cell', fontsize=13)
        plt.tight_layout()
        plt.savefig('Gamma_drifts_both.png', dpi=600, bbox_inches='tight')
        plt.savefig('Gamma_drifts_both.pdf', bbox_inches='tight')
        plt.show()

    else:
        raise ValueError("plot_which must be one of 'ExB', 'gradB', or 'both'.")



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_q_streamlines(plot_which='both', cmap='viridis', density=1.2, linewidth=1.5):
    """
    Plot heat flux streamlines from get_q_drifts_func(), excluding guard cells.
    Supports plotting ExB, gradB, or both, with OMP poloidal cell highlighted.

    Parameters:
        get_q_drifts_func (callable): function returning (q_ExB, q_gradB)
        bbb (object): must contain attribute 'ixmp' for OMP poloidal cell index
        plot_which (str): 'ExB', 'gradB', or 'both'
        cmap (str): colormap for streamlines magnitude
        density (float): streamline density
        linewidth (float): streamline width
    """
    # Publication-quality settings
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    q_ExB, q_gradB = get_q_drifts()
    omp_cell = bbb.ixmp  # OMP poloidal cell index (guard cells excluded assumed)
    separatrix_cell = com.iysptrx   # separatrix cell number
    outer_wall_cell = com.ny        # outer wall cell number
    oxpt = com.ixpt2[0]
    ixpt = com.ixpt1[0]

    def plot_single(ax, q_vec, title):
        U = q_vec[1:-1, 1:-1, 0].T
        V = q_vec[1:-1, 1:-1, 1].T
        ny, nx = U.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        magnitude = (np.sqrt(U**2 + V**2))/1e6  # MW/m^2
        vmax = np.nanmax(magnitude)
        vmin = np.nanmin(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1e-2
        norm = LogNorm(vmin=vmin, vmax=vmax)
        strm = ax.streamplot(X, Y, U, V, color=magnitude, cmap=cmap,
                             linewidth=linewidth, density=density, norm=norm)

        cbar = plt.colorbar(strm.lines, ax=ax, orientation='vertical', pad=0.01, aspect=30)
        cbar.set_label(r'q [MW m$^{-2}$]', fontsize=13)
        strm.lines.set_clim(vmin, vmax)

        ax.set_title(title, pad=10)
        ax.set_ylabel('Radial cell', fontsize=13)

        # OMP, X-points
        if 0 <= omp_cell-1 < nx:
            ax.axvline(omp_cell - 1, color='red', linestyle='--', linewidth=2, label='OMP')
            ax.axvspan(omp_cell - 1 - 0.5, omp_cell - 1 + 0.5, color='red', alpha=0.08)
            ax.text(omp_cell - 1, ny, "OMP", color='red', fontsize=13, ha='center', va='bottom', fontweight='bold')

        if 0 <= oxpt-1 < nx:
            ax.axvline(oxpt - 1, color='orange', linestyle='--', linewidth=2, label='Outer X-point')
            ax.axvspan(oxpt - 1 - 0.5, oxpt - 1 + 0.5, color='orange', alpha=0.08)
            ax.text(oxpt - 1, -0.2, "Oxpt", color='orange', fontsize=13, ha='center', va='bottom', fontweight='bold')
        if 0 <= ixpt-1 < nx:
            ax.axvline(ixpt - 1, color='green', linestyle='--', linewidth=2, label='Inner X-point')
            ax.axvspan(ixpt - 1 - 0.5, ixpt - 1 + 0.5, color='green', alpha=0.08)
            ax.text(ixpt - 1, -0.2, "Ixpt", color='green', fontsize=13, ha='center', va='bottom', fontweight='bold')

        # Separatrix
        if 0 <= separatrix_cell-1 < ny:
            ax.axhline(separatrix_cell - 1, color='magenta', linestyle='-.', linewidth=2, label='Separatrix')
            ax.text(0, separatrix_cell - 1, "Sep", color='magenta', fontsize=13, ha='left', va='bottom', fontweight='bold')

        # Outer wall
        if 0 <= outer_wall_cell-1 < ny:
            ax.axhline(outer_wall_cell - 1, color='blue', linestyle=':', linewidth=2, label='Outer Wall')
            ax.text(nx+1 - com.nx, outer_wall_cell - 1, "Owall", color='blue', fontsize=13, ha='right', va='bottom', fontweight='bold')

        # Inner and outer divertor
        #ax.text(0, -0.08*ny, "Idiv", color='black', fontsize=13, ha='left', va='top', fontweight='bold')
        #ax.text(nx-1, -0.08*ny, "Odiv", color='black', fontsize=13, ha='right', va='top', fontweight='bold')

       
        ax.set_xlim(0, nx+1)
        ax.set_ylim(0, ny+1)
        ax.set_aspect('auto')

        # Remove duplicate legend entries
        #handles, labels = ax.get_legend_handles_labels()
        #by_label = dict(zip(labels, handles))
        #if by_label:
        #    ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, fontsize=12)

    plot_which = plot_which.lower()

    if plot_which == 'exb':
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_single(ax, q_ExB, r'$q_{ExB} = \frac{5}{2} P v_{ExB}$')
        ax.set_xlabel('Poloidal cell', fontsize=13)
        plt.tight_layout()
        plt.savefig('q_drifts_exb.png', dpi=600, bbox_inches='tight')
        plt.savefig('q_drifts_exb.pdf', bbox_inches='tight')
        plt.show()

    elif plot_which == 'gradb':
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_single(ax, q_gradB, r'$q_{\nabla B} = \frac{5}{2} P v_{\nabla B}$')
        ax.set_xlabel('Poloidal cell', fontsize=13)
        plt.tight_layout()
        plt.savefig('q_drifts_gradb.png', dpi=600, bbox_inches='tight')
        plt.savefig('q_drifts_gradb.pdf', bbox_inches='tight')
        plt.show()

    elif plot_which == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        plot_single(ax1, q_ExB, r'$q_{ExB} = \frac{5}{2} P v_{ExB}$')
        plot_single(ax2, q_gradB, r'$q_{\nabla B} = \frac{5}{2} P v_{\nabla B}$')
        ax2.set_xlabel('Poloidal cell', fontsize=13)
        plt.tight_layout()
        plt.savefig('q_drifts_both.png', dpi=600, bbox_inches='tight')
        plt.savefig('q_drifts_both.pdf', bbox_inches='tight')
        plt.show()

    else:
        raise ValueError("plot_which must be one of 'ExB', 'gradB', or 'both'.")


def parallel_ion_flux(sep=9, savefig=True, filename=None, dpi=600):
    plt.figure(figsize=(5,3))
    # Perpendicular ion flux
    L = analysis.para_conn_length(bbb,com)
    plt.plot(L[1:-1,sep], bbb.fniy[1:-1,sep,0]/com.sy[1:-1,sep], '--k', label='$\\Gamma_{\\perp}$')
    # Parallel ion flux
    plt.plot(L[1:-1,sep], bbb.fnix[1:-1,sep,0]/com.sx[1:-1,sep], '-r', label='$\\Gamma_{||}$')
    omp_x = L[bbb.ixmp,sep]
    plt.axvline(omp_x, color='g', linestyle=':', linewidth=2)
    plt.legend()
    plt.grid()
    plt.ylabel('$\\Gamma_{D+}$ (m/s)', fontsize=16)
    plt.xlabel('L$_{||}$ from Idiv (m)', fontsize=16)
    plt.title(f'Ion Fluxes for Flux Tube {sep}', fontsize=16)
    plt.tight_layout()
    if savefig:
        if filename is None:
            filename = f"ion_fluxes_flux_tube_{sep}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved as {filename} with dpi={dpi}")
    plt.show()


def parallel_heat_flux(sep=10, savefig=True, filename=None, dpi=600):
    plt.figure(figsize=(5,3))
    L = analysis.para_conn_length(bbb,com)
    q_para = bbb.feex + bbb.feix
    q_perp = bbb.feey + bbb.feiy
    plt.plot(L[1:-1,sep], q_perp[1:-1,sep], '--k', label='$q_{\\perp}$')
    plt.plot(L[1:-1,sep], q_para[1:-1,sep], '-r', label='$q_{||}$')
    plt.legend()
    omp_x = L[bbb.ixmp,sep]
    plt.axvline(omp_x, color='g', linestyle=':', linewidth=2)
    plt.grid()
    plt.ylabel('Heat flux (W/m$^2$)', fontsize=16)
    plt.xlabel('L$_{||}$ from Idiv (m)', fontsize=16)
    plt.title(f'Heat Fluxes for Flux Tube {sep}', fontsize=16)
    plt.tight_layout()
    if savefig:
        if filename is None:
            filename = f"heat_fluxes_flux_tube_{sep}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved as {filename} with dpi={dpi}")
    plt.show()




def plot_ne_te_upara(sep=10, savefig=True, filename=None, dpi=600):
    L = analysis.para_conn_length(bbb, com)
    x = L[1:-1, sep]
    ne = bbb.ne[1:-1, sep]
    te = bbb.te[1:-1, sep] / bbb.ev
    upara = bbb.up[1:-1, sep, 0]
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
    
    axs[0].plot(x, ne/1e20, '-b')
    axs[0].set_ylabel('$n_e$ (10$^{20}$ m$^{-3}$)', fontsize=14)
    axs[0].set_title(f'Flux Tube {sep}', fontsize=16)
    axs[0].grid(True)
    
    axs[1].plot(x, te, '-r')
    axs[1].set_ylabel('$T_e$ (eV)', fontsize=14)
    axs[1].grid(True)
    
    axs[2].plot(x, upara/1e3, '-g')
    axs[2].set_ylabel('$u_{||}$ (km/s)', fontsize=14)
    axs[2].set_xlabel('L$_{||}$ from Idiv (m)', fontsize=16)
    axs[2].grid(True)
    omp_x = L[bbb.ixmp,sep]
    axs[2].axvline(omp_x, color='g', linestyle=':', linewidth=2)

    
    plt.tight_layout()
    
    if savefig:
        if filename is None:
            filename = f"plasma_params_flux_tube_{sep}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved as {filename} with dpi={dpi}")
    plt.show()

def plot_radial_drifts_vel_profiles(ion_index=0, iy_flux =10,  y_mid = 74):
    nx, ny = com.nx, com.ny

    v2cb   = bbb.v2cb[:, :, 0]     # ion poloidal grad-B drift
    ve2cb  = bbb.ve2cb[:, :]               # electron poloidal grad-B drift
    v2ce   = bbb.v2ce[:, :, 0]     # ion poloidal ExB drift
    vex    = bbb.vex[:, :]                 # electron ExB poloidal drift

    vycb   = bbb.vycb[:, :, 0]     # ion radial grad-B drift
    veycb  = bbb.veycb[:, :]               # electron radial grad-B drift
    vyce   = bbb.vyce[:, :, 0]     # ion radial ExB drift

    grad_b = vycb 
    Diff = bbb.vydd[bbb.ixmp, :, 0]

    # --- Radial Drift at OMP ---
    plt.figure(figsize=(5, 3))
    plt.plot(com.yyc, grad_b[y_mid, :], label="($\\nabla B$)", linewidth=2)
    plt.plot(com.yyc, vyce[y_mid, :], '--', label="(ExB)", alpha=0.7)
    plt.xlabel('r$_{omp}$ - r$_{sep}$ [m]', fontsize=16)
    plt.ylabel("Radial velocity [m/s]", fontsize=16)
    plt.title("Radial Drift at OMP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim([0, np.max(vyce[y_mid, :])*1.05])
    plt.savefig(f'drifts_vel_ion{ion_index}.png', dpi=300)
    plt.show()

    # --- Plasma Potential and Er ---
    phi = bbb.phi[y_mid, :]
    Er = bbb.ey[bbb.ixmp, :]  # radial electric field [V/m]

    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    axs[0].plot(com.yyc, phi, label='Plasma Potential $\phi$')
    axs[0].set_ylabel('$\phi$ [V]', fontsize=16)
    axs[0].set_title('Midplane profile')
    axs[0].grid(True)
    axs[0].set_ylim([0, np.max(phi)*1.05])
    axs[0].legend()

    axs[1].plot(com.yyc, Er)
    axs[1].set_xlabel('r$_{omp}$ - r$_{sep}$ [m]', fontsize=16)
    axs[1].set_ylabel('E$_r$ [V/m]', fontsize=16)
    axs[1].grid(True)
    axs[1].set_ylim([0, np.max(Er)*1.05])
    plt.tight_layout()
    plt.savefig(f'Er_phi_omp_ion{ion_index}.png', dpi=300)
    plt.show()

    # --- Parallel Connection Length and Potential ---
    L = analysis.para_conn_length(bbb, com)  # shape (nx, ny)
    l_parallel = L[:, iy_flux]

    plt.figure(figsize=(5, 3))
    plt.plot(l_parallel, bbb.phi[:, iy_flux], label="Plasma Potential")
    plt.xlabel("Parallel Connection Length [m]")

    omp_x = l_parallel[bbb.ixmp]
    plt.axvline(omp_x, color='k', linestyle=':', label='OMP')
    y_annotate = np.max(bbb.phi[:, iy_flux]) * 0.95
    plt.text(omp_x, y_annotate, "OMP", rotation=90, va='top', ha='right', color='k', fontsize=10)

    plt.ylabel("Potential [V]")
    plt.title(f"Flux tube ({iy_flux})")
    plt.ylim([0, np.max(bbb.phi[:, iy_flux])*1.05])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Phi_pol_ion{ion_index}.png', dpi=300)
    plt.show()

    # --- Combined 3-panel Plot ---
    fig, axs = plt.subplots(3, 1, figsize=(4, 7), sharex=True)

    axs[0].plot(com.yyc, phi, label='Plasma Potential $\phi$')
    axs[0].set_ylabel('$\phi$ [V]', fontsize=14)
    axs[0].set_title('Midplane Profile')
    axs[0].grid(True)
    axs[0].set_ylim([0, np.max(phi)*1.05])
    axs[0].legend()

    axs[1].plot(com.yyc, Er, color='tab:orange', label='E$_r$')
    axs[1].set_ylabel('E$_r$ [V/m]', fontsize=14)
    axs[1].grid(True)
    axs[1].set_ylim([0, np.max(Er)*1.05])
    axs[1].legend()

    axs[2].plot(com.yyc, grad_b[y_mid, :], label="($\\nabla B$)", linewidth=2)
    axs[2].plot(com.yyc, vyce[y_mid, :], '-', label="(ExB)", linewidth=2)
    axs[2].set_xlabel('r$_{omp}$ - r$_{sep}$ [m]', fontsize=14)
    axs[2].set_ylabel("Radial velocity [m/s]", fontsize=14)
    axs[2].grid(True)
    #axs[2].set_ylim([0, np.max(vyce[y_mid, :])*1.05])
    axs[2].legend()
    plt.tight_layout()
    plt.savefig(f'combined_3panel_ion{ion_index}.png', dpi=300)
    plt.show()


def divertor_variable(index=com.nx):
    te_div = bbb.te[index, 1:-1] / bbb.ev
    ne_div = bbb.ne[index, 1:-1]
    q_perp = (bbb.sdrrb + bbb.sdtrb).reshape(-1)
    q_perp = q_perp[1:-1]
    return te_div, ne_div, q_perp



def plot_Pparallel_boundary(com, bbb, boundary_idx=None, y_axis=None):
    # Set defaults if not provided
    if boundary_idx is None:
        boundary_idx = com.nx
    if y_axis is None:
        y_axis = com.yyrb

    def getrrf():
        bpol_local = 0.5 * (com.bpol[:, :, 2] + com.bpol[:, :, 4])
        bphi_local = 0.5 * (com.bphi[:, :, 2] + com.bphi[:, :, 4])
        btot_local = np.sqrt(bpol_local ** 2 + bphi_local ** 2)
        return bpol_local / btot_local

    def PionParallelKE():
        return 0.5 * bbb.mi[0] * bbb.up[:, :, 0] ** 2 * bbb.fnix[:, :, 0]

    def Pparallel():
        return bbb.feex + bbb.feix + PionParallelKE()

    if y_axis is com.yyrb:
        boundary_idx_index = "Odiv"
    elif y_axis is com.yylb:
        boundary_idx_index = "Idiv"
    else:
        raise ValueError("y_axis must be yyrb or yylb")

    ppar_boundary = np.abs(Pparallel()[boundary_idx, :])
    rrf_boundary = getrrf()[boundary_idx, :]
    sxnp_boundary = com.sxnp[boundary_idx, :]
    ppar = ppar_boundary / sxnp_boundary / rrf_boundary

    plt.figure(figsize=(5, 3))
    plt.plot(y_axis[1:-1], ppar[1:-1] / 1e6, label=f"{boundary_idx_index}")
    plt.xlabel("r$_{div}$ - r$_{sep}$ (m)", fontsize=16)
    plt.ylabel('q$_{||}$ (MW/m$^2$)', fontsize=16)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylim([0, np.max(ppar / 1e6) * 1.05])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('qpara.png', dpi=300)
    plt.show()


def parallel_ion_flux(sep=10):
    L = analysis.para_conn_length(bbb, com)
    x = L[1:-1, sep]
    gamma_perp = bbb.fniy[1:-1, sep, 0] / com.sy[1:-1, sep]
    gamma_para = bbb.fnix[1:-1, sep, 0] / com.sx[1:-1, sep]
    omp_x = L[bbb.ixmp, sep]
    return omp_x, gamma_perp, gamma_para

def parallel_heat_flux(sep=10):
    L = analysis.para_conn_length(bbb, com)
    x = L[1:-1, sep]
    q_para = (bbb.feex + bbb.feix)/com.sx
    q_perp = (bbb.feey + bbb.feiy)/com.sy
    omp_x = L[bbb.ixmp, sep]
    return   q_perp[1:-1, sep], q_para[1:-1, sep], omp_x
    

def ne_te_upara(sep=10):
    L = analysis.para_conn_length(bbb, com)
    x = L[1:-1, sep]
    ne = bbb.ne[1:-1, sep]
    te = bbb.te[1:-1, sep] / bbb.ev
    upara = bbb.up[1:-1, sep, 0]
    omp_x = L[bbb.ixmp, sep]
    return  omp_x ,ne, te, upara
    

def plot_heat_flux_streamlines_with_markers(
    omp_cell=bbb.ixmp, oxpt=com.ixpt2[0], ixpt= com.ixpt1[0] , separatrix_cell=com.iysptrx,
    title=None, cmap='viridis', linewidth=1.2, density=1.5):
   
    nx = bbb.feex.shape[0] - 2  
    ny = bbb.feex.shape[1] - 2

 
    r_idx = np.arange(1, nx+1)
    z_idx = np.arange(1, ny+1)
    R_idx, Z_idx = np.meshgrid(r_idx, z_idx, indexing='xy')

    q_x = (bbb.feex + bbb.feix)[1:-1, 1:-1] / com.sx[1:-1, 1:-1] / 1e6
    q_y = (bbb.feey + bbb.feiy)[1:-1, 1:-1] / com.sy[1:-1, 1:-1] / 1e6

    # Compute magnitude for colorbar
    q_mag = np.sqrt(q_x**2 + q_y**2)
    vmax = np.nanmax(q_mag)
    vmin = np.nanmin(q_mag[q_mag > 0]) if np.any(q_mag > 0) else 1e-2
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7,5))
    strm = ax.streamplot(
        R_idx, Z_idx,
        q_x.T, q_y.T,
        color=q_mag.T,
        linewidth=linewidth,
        cmap=cmap,
        density=density,
        norm=norm
    )

    cbar = plt.colorbar(strm.lines, ax=ax, orientation='vertical', pad=0.01, aspect=30)
    cbar.set_label(r'q [MW m$^{-2}$]', fontsize=13)
    strm.lines.set_clim(vmin, vmax)

    ax.set_title(title, pad=10)
    ax.set_ylabel('Radial cell', fontsize=13)
    ax.set_xlabel('Poloidal cell', fontsize=13)

    if omp_cell is not None and 0 <= omp_cell-1 < nx:
        ax.axvline(omp_cell - 1, color='red', linestyle='--', linewidth=2, label='OMP')
        ax.axvspan(omp_cell - 1 - 0.5, omp_cell - 1 + 0.5, color='red', alpha=0.08)
        ax.text(omp_cell - 1, ny, "OMP", color='red', fontsize=13, ha='center', va='bottom', fontweight='bold')

    if oxpt is not None and 0 <= oxpt-1 < nx:
        ax.axvline(oxpt - 1, color='orange', linestyle='--', linewidth=2, label='Outer X-point')
        ax.axvspan(oxpt - 1 - 0.5, oxpt - 1 + 0.5, color='orange', alpha=0.08)
        ax.text(oxpt - 1, -0.2, "Oxpt", color='orange', fontsize=13, ha='center', va='bottom', fontweight='bold')
    if ixpt is not None and 0 <= ixpt-1 < nx:
        ax.axvline(ixpt - 1, color='green', linestyle='--', linewidth=2, label='Inner X-point')
        ax.axvspan(ixpt - 1 - 0.5, ixpt - 1 + 0.5, color='green', alpha=0.08)
        ax.text(ixpt - 1, -0.2, "Ixpt", color='green', fontsize=13, ha='center', va='bottom', fontweight='bold')

    if separatrix_cell is not None and 0 <= separatrix_cell-1 < ny:
        ax.axhline(separatrix_cell - 1, color='magenta', linestyle='-.', linewidth=2, label='Separatrix')
        ax.text(0, separatrix_cell - 1, "Sep", color='magenta', fontsize=13, ha='left', va='bottom', fontweight='bold')

    ax.set_xlim([0, nx+1])
    ax.set_ylim([0, ny+1])
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig('heat.png', dpi=300)
    plt.show()


def computeConductionFluxes2D():
    shape = (com.nx + 2, com.ny + 2)
    econd = np.zeros(shape)
    icond = np.zeros(shape)
    ncond = np.zeros(shape)
    conyn = com.sy * bbb.hcyn / com.dynog

    for ix in range(com.nx + 2):
        for iy in range(com.ny + 1):
            econd[ix, iy] = -bbb.conye[ix, iy] * (bbb.te[ix, iy + 1] - bbb.te[ix, iy])
            ncond[ix, iy] = -conyn[ix, iy] * (bbb.ti[ix, iy + 1] - bbb.ti[ix, iy])
            icond[ix, iy] = -bbb.conyi[ix, iy] * (bbb.ti[ix, iy + 1] - bbb.ti[ix, iy]) - ncond[ix, iy]

    return econd, icond, ncond




def compute_radial_heat_fluxes(j_index=bbb.ixmp, chi=bbb.kye, plot=True, y = com.yyc):
    """
    Compute and optionally plot radial heat fluxes due to:
    1. grad-B drift convection
    2. anomalous thermal diffusion

    Parameters:
    -----------
    bbb : object
        Object containing ne, ni, te, ti, vxcb, xgrid, ev, etc.
    j_index : int, optional
        Poloidal index to analyze (default = midplane)
    chi : float, optional
        Anomalous radial thermal diffusivity [m^2/s]
    plot : bool, optional
        If True, plot the heat fluxes

    Returns:
    --------
    x : ndarray
        Radial coordinate [m]
    q_gradB_1D : ndarray
        grad-B convective radial heat flux [W/m^2]
    q_anom_1D : ndarray
        Anomalous radial heat flux [W/m^2]
    """
    # Physical constant
    eV_to_J = bbb.ev

    # Default poloidal index: midplane
    if j_index is None:
        j_index = com.iysptrx+1

    # Convert temperature from eV to J
    Te = (bbb.te / bbb.ev) * eV_to_J     # [J]
    Ti = (bbb.ti / bbb.ev) * eV_to_J     # [J]

 
    ne = bbb.ne                          # [m^-3]
    ni = bbb.ni[:, :, 0]                 # [m^-3]
    vx_gradB = bbb.vycb[:, :, 0]         # [m/s]
    x = analysis.para_conn_length(bbb,com)
       

    P = 2.5 * (ne * Te) + 2.5 * (ni * Ti)    # [Pa]

    q_gradB = vx_gradB * P                 # [W/m^2]

    dTe_dy = bbb.gtey   # [J/m]
    dTi_dy = bbb.gtiy 

    # Anomalous radial heat flux: -5/2 * chi * ne * ?_r Te
    q_anom = - (5/2) * chi * (ne * dTe_dy + ni * dTi_dy)
    q_pwtol = (bbb.feey + bbb.feiy)

    q_gradB_1D = q_gradB[j_index,:]
    q_anom_1D = q_anom[j_index, :]
    print("size of q_gradB_1D is", len(q_gradB_1D))
    print("size of q_anom_1D is", len(q_anom_1D))
    print("size of y is", len(y))
    omp_cell=bbb.ixmp
    
    q_gradB_power_per_length = (q_gradB * com.sy[:, :]).sum(axis=0)  
    q_anom_power_per_length = (q_anom * com.sy[:, :]).sum(axis=0)    

    plt.figure(figsize=(5,3))
    plt.plot(com.yyc, q_gradB_power_per_length / 1e6, label='grad-B', linewidth=2)
    plt.plot(com.yyc, q_anom_power_per_length / 1e6, label='Anomalous', linewidth=2)
    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
    plt.ylabel('q$_{radial}$ [MW/m]', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('power_per_length_vs_yyc.png', dpi=300)
    plt.show()


    if plot:
        plt.figure(figsize=(5, 3))
        plt.plot(y, q_gradB_1D/1e6, label='grad-B', linewidth=2)
        plt.plot(y, q_anom_1D/1e6, label='Anomalous', linewidth=2)
        plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
        plt.ylabel('q$_{radial}$ [MW/m$^2$]', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('radial_drift.png', dpi=300)
        plt.show()
     

      

    return q_gradB_1D, q_anom_1D


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

def plot_poloidal_heat_flux(mid_idx=com.iysptrx+1):
    nx, ny = bbb.te.shape
    kappa_elec = np.zeros((nx, ny))
    kappa_ion = np.zeros((nx, ny))
    ni_species0 = bbb.ni[:, :, 0]

    q_poloidal_elec = - bbb.hcxe * bbb.gtex   # W/m^2
    q_poloidal_ion  =  - bbb.hcxi * bbb.gtix   # W/m^2
    q_poloidal_total = q_poloidal_elec + q_poloidal_ion
    q_spitzer_pol = q_poloidal_total

    p = 2.5 * (bbb.ne * bbb.te + bbb.ni[:, :, 0] * bbb.ti)
    nx, ny = com.nx + 2, com.ny + 2
    q_ExB = np.zeros((nx, ny, 2))
    q_gradB = np.zeros((nx, ny, 2))
    poloidal_indices = np.arange(nx)

    q_ExB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2ce[:, :, 0] * p
    q_ExB[:, :, 1] = bbb.vyce[:, :, 0] * p
    q_gradB[:, :, 0] = -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.v2cb[:, :, 0] * 2.5*bbb.ni[:, :, 0] * bbb.ti - -np.sign(bbb.b0) * np.sqrt(1 - com.rr**2) * bbb.ve2cb * 2.5   *bbb.ne * bbb.te
    #q_gradB[:, :, 0] = -np.sign(bbb.b0) * com.rr* bbb.v2cb[:, :, 0] * p
    q_gradB[:, :, 1] = bbb.vycb[:, :, 0] * p

    q_ExB_pol = q_ExB[:, :, 0]
    q_gradB_pol = q_gradB[:, :, 0]
    total_pol = q_spitzer_pol + q_gradB_pol + q_ExB_pol
    qwtot = (bbb.feix+bbb.feex)/com.sx

    np.save('q_spitzer_gradBdrfits.npy', q_spitzer_pol)
    np.save('q_gradB_pol_gradBdrfits.npy', q_gradB_pol)
    np.save('q_ExB_pol_nodrfits.npy', q_ExB_pol)

    plt.figure(figsize=(5.5, 3.25))
    plt.plot(poloidal_indices[1:-1], q_spitzer_pol[1:-1, mid_idx]/1e6, label = r'$q_{\parallel}$', linewidth=2)
    plt.plot(poloidal_indices[1:-1], q_gradB_pol[1:-1, mid_idx]/1e6, label=r'$q_{\nabla B}$', linewidth=2)
    plt.plot(poloidal_indices[1:-1], q_ExB_pol[1:-1, mid_idx]/1e6, label=r'$q_{E \times B}$', linewidth=2)
    plt.plot(poloidal_indices[1:-1], total_pol[1:-1, mid_idx]/1e6, label=r'Total', linewidth=2)
    plt.plot(poloidal_indices[1:-1], qwtot[1:-1, mid_idx]/1e6, label='feex+feix', linewidth=2)

    plt.xlabel('Poloidal grid index', fontsize=16)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(poloidal_indices[bbb.ixmp], color='red', linestyle=':', linewidth=2)
    plt.ylabel('q$_{Poloidal}$ [MW/m$^2$]', fontsize=16)
    plt.legend(fontsize=10)
    plt.xlim([0, 106])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("poloidal_heat_flux.png", dpi=300)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def plot_ion_radial_heat_flux(bbb, com, scale=1e6):
    """
    Plot ion radial heat flux components:
      - Anomalous diffusion
      - Grad-B drift
      - E×B drift
      - Total
    
    Parameters
    ----------
    bbb : object
        UEDGE bbb object with velocity and flux arrays
    com : object
        UEDGE com object with grid data
    Pi : ndarray
        Ion pressure array, shape (nx, ny)
    scale : float
        Unit scaling (default = 1e6 for MW/m^2)
    """
    # individual components
    Pi = 2.5*bbb.ni[:,:,0]*bbb.ti[:,:]
    q_ion = Pi*bbb.vy[:,:,0]
     
    q_an_dif  =  Pi * bbb.vydd[:,:,0] * com.sy
    q_grad_ion = Pi * bbb.vycb[:,:,0] * com.sy
    qEB_ion    = Pi * bbb.vyce[:,:,0] * com.sy

  
    q_an_dif  = np.sum(q_an_dif, axis=0) / scale
    q_grad_ion = np.sum(q_grad_ion, axis=0) / scale
    qEB_ion    = np.sum(qEB_ion, axis=0) / scale
    q_uedge = np.sum(bbb.feiy, axis =0) /scale
    q_cond= (-(5/2) * bbb.kyi * (bbb.ni[:,:,0]* bbb.gtiy))*com.sy
    q_cond= np.sum(q_cond, axis=0)/scale

    
    q_total = q_an_dif + q_grad_ion + qEB_ion + q_cond

    # plotting
    plt.figure(figsize=(7,5))
    plt.plot(com.yyc, q_cond,  label='Conduction')
    plt.plot(com.yyc, q_an_dif,  label='Anom. Diff')
    plt.plot(com.yyc, q_grad_ion, label='∇B drift')
    plt.plot(com.yyc, qEB_ion,    label='E×B drift')
    plt.plot(com.yyc, q_total,    label='Total', lw=2, color='k')
    plt.plot(com.yyc, q_uedge,    label='UEDGE', lw=2, color='red', linestyle = '--')

    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
    plt.ylabel('q$_{radial}^{ion}$ [MW]', fontsize=16)
    plt.legend()
    plt.grid()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_elec_radial_heat_flux(bbb, com, scale=1e6):
    """
    Plot electron radial heat flux components:
      - Conduction
      - Anomalous diffusion
      - Grad-B drift
      - E×B drift
      - Total
    
    Parameters
    ----------
    bbb : object
        UEDGE bbb object with velocity and flux arrays
    com : object
        UEDGE com object with grid data
    scale : float
        Unit scaling (default = 1e6 for MW)
    """
    # electron pressure
    Pe = 2.5 * bbb.ne[:,:] * bbb.te[:,:]

    # individual components
    q_an_dif  = Pe * bbb.vydd[:,:,0] * com.sy
    q_grad_e  = Pe * bbb.veycb[:,:] * com.sy
    qEB_e     = Pe * bbb.vyce[:,:,0] * com.sy

    # conduction term: -(5/2) * kye * (ne * dTe/dy) * Sy
    q_cond = (-(5/2) * bbb.kye * (bbb.ne[:,:] * bbb.gtey)) * com.sy

    q_uedge = np.sum(bbb.feey, axis=0) / scale


    q_an_dif = np.sum(q_an_dif, axis=0) / scale
    q_grad_e = np.sum(q_grad_e, axis=0) / scale
    qEB_e    = np.sum(qEB_e, axis=0) / scale
    q_cond   = np.sum(q_cond, axis=0) / scale

    # total
    q_total = q_an_dif + abs(q_grad_e) + qEB_e + q_cond

    # plotting
    plt.figure(figsize=(7,5))
    plt.plot(com.yyc, q_cond,    label='Conduction')
    plt.plot(com.yyc, q_an_dif,  label='Anom. Diff')
    plt.plot(com.yyc, q_grad_e,  label='∇B drift')
    plt.plot(com.yyc, qEB_e,     label='E×B drift')
    plt.plot(com.yyc, q_total,   label='Total', lw=2, color='k')
    plt.plot(com.yyc, q_uedge,   label='UEDGE', lw=2, color='red', linestyle='--')

    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
    plt.ylabel('q$_{radial}^{e}$ [MW]', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_total_radial_heat_flux_ele_ion(bbb, com, scale=1e6):
    """
    Plot combined ion + electron radial heat flux components:
      - Conduction
      - Anomalous diffusion
      - Grad-B drift
      - E×B drift
      - Total
    
    Parameters
    ----------
    bbb : object
        UEDGE bbb object with velocity and flux arrays
    com : object
        UEDGE com object with grid data
    scale : float
        Unit scaling (default = 1e6 for MW/m^2)
    """

    # --- Pressures ---
    Pi = 2.5 * bbb.ni[:,:,0] * bbb.ti[:,:]     # ion pressure
    Pe = 2.5 * bbb.ne[:,:]   * bbb.te[:,:]     # electron pressure

    # --- ION contributions ---
    q_an_dif_i  = Pi * bbb.vydd[:,:,0] * com.sy
    q_grad_i    = Pi * bbb.vycb[:,:,0] * com.sy
    qEB_i       = Pi * bbb.vyce[:,:,0] * com.sy
    q_cond_i    = (-(5/2) * bbb.kyi * (bbb.ni[:,:,0]*bbb.gtiy)) * com.sy

    # --- ELEC contributions ---
    q_an_dif_e  = Pe * bbb.vydd[:,:,0] * com.sy
    q_grad_e    = Pe * bbb.veycb[:,:] * com.sy
    qEB_e       = Pe * bbb.vyce[:,:,0] * com.sy
    q_cond_e    = (-(5/2) * bbb.kye * (bbb.ne[:,:]*bbb.gtey)) * com.sy

    # --- Sum over x (radial integration) ---
    q_an_dif = (np.sum(q_an_dif_i, axis=0) + np.sum(q_an_dif_e, axis=0)) / scale
    q_grad   = (np.sum(q_grad_i,   axis=0) + abs(np.sum(q_grad_e,   axis=0)) )/ scale
    qEB      = (np.sum(qEB_i,      axis=0) + (np.sum(qEB_e,      axis=0))) / scale
    q_cond   = (np.sum(q_cond_i,   axis=0) + np.sum(q_cond_e,   axis=0)) / scale

    # --- UEDGE flux sum (ion+electron) ---
    q_uedge  = (np.sum(bbb.feiy, axis=0) + np.sum(bbb.feey, axis=0)) / scale

    # --- Total ---
    q_total = q_an_dif + q_grad + qEB + q_cond

    # --- Plotting ---
    plt.figure(figsize=(7,5))
    plt.plot(com.yyc, q_cond,   label='Conduction')
    plt.plot(com.yyc, q_an_dif, label='Anom. Diff')
    plt.plot(com.yyc, q_grad,   label='∇B drift')
    plt.plot(com.yyc, qEB,      label='E×B drift')
    plt.plot(com.yyc, q_total,  label='Total', lw=2, color='k')
    plt.plot(com.yyc, q_uedge,  label='UEDGE', lw=2, color='red', linestyle='--')

    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=16)
    plt.ylabel('q$_{radial}^{i+e}$ [MW]', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plotRadialFluxes_ion():
    # --- Initialize arrays ---
    shape = (com.nx+2, com.ny+2)
    fniy = np.zeros(shape)
    fniydif = np.zeros(shape)
    fniyconv = np.zeros(shape)
    fniyef = np.zeros(shape)
    fniybf = np.zeros(shape)
    P = np.zeros(shape)

    vyconv = bbb.vcony[0] + bbb.vy_use[:, :, 0] + bbb.vy_cft[:, :, 0]
    vydif = bbb.vydd[:, :, 0] - vyconv

    for ix in range(com.nx+2):
        for iy in range(com.ny+1):
            t2 = bbb.niy0[ix, iy, 0] if bbb.vy[ix, iy, 0] > 0 else bbb.niy1[ix, iy, 0]
            fniy[ix, iy] = bbb.cnfy * bbb.vy[ix, iy, 0] * com.sy[ix, iy] * t2
            fniydif[ix, iy] = vydif[ix, iy] * com.sy[ix, iy] * t2
            fniyconv[ix, iy] = vyconv[ix, iy] * com.sy[ix, iy] * t2
            fniyef[ix, iy] = bbb.cfyef * bbb.vyce[ix, iy, 0] * com.sy[ix, iy] * t2
            fniybf[ix, iy] = bbb.cfybf * bbb.vycb[ix, iy, 0] * com.sy[ix, iy] * t2
            P[ix,iy] = 2.5*t2
     
            if bbb.vy[ix, iy, 0] * (bbb.ni[ix, iy, 0] - bbb.ni[ix, iy+1, 0]) < 0:
                fniy[ix, iy] /= (1 + (bbb.nlimiy[0]/bbb.ni[ix, iy+1, 0])**2 + (bbb.nlimiy[0]/bbb.ni[ix, iy, 0])**2)

    # --- Upwind utilities ---
    def upwind(f, p1, p2): return max(f, 0)*p1 + min(f, 0)*p2
    def upwindProxy(f, g, p1, p2): return max(f, 0)/f * g * p1 + min(f, 0)/f * g * p2 if f != 0 else 0

    # --- Heat flux decomposition ---
    feey = np.zeros(shape)
    feiy = np.zeros(shape)
    econd = np.zeros(shape)
    econv = np.zeros(shape)
    icond = np.zeros(shape)
    iconv = np.zeros(shape)
    ncond = np.zeros(shape)
    nconv = np.zeros(shape)
    conyn = com.sy * bbb.hcyn / com.dynog
    Qi_gradB = np.zeros(shape)
    Qi_EB = np.zeros(shape)
    Qi_An_diff = np.zeros(shape)

    for ix in range(com.nx+2):
        for iy in range(com.ny+1):
            econd[ix, iy] = -bbb.conye[ix, iy] * (bbb.te[ix, iy+1] - bbb.te[ix, iy])
            econv[ix, iy] = upwind(bbb.floye[ix, iy], bbb.te[ix, iy], bbb.te[ix, iy+1])
            ncond[ix, iy] = -conyn[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy])
            icond[ix, iy] = -bbb.conyi[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy]) - ncond[ix, iy]
            floyn = bbb.cfneut * bbb.cfneutsor_ei * 2.5 * bbb.fngy[ix, iy, 0]
            floyi = bbb.floyi[ix, iy] - floyn
            iconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyi, bbb.ti[ix, iy], bbb.ti[ix, iy+1])
            nconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyn, bbb.ti[ix, iy], bbb.ti[ix, iy+1])
            Qi_gradB[ix, iy] = P[ix,iy] *(bbb.ti[ix, iy])*bbb.vycb[ix,iy,0]*com.sy[ix,iy]
            Qi_EB[ix, iy] = P[ix,iy] *( bbb.ti[ix, iy])*bbb.vyce[ix,iy,0]*com.sy[ix,iy]
            Qi_An_diff[ix, iy] = P[ix,iy] *(bbb.ti[ix, iy]) *vydif[ix,iy]*com.sy[ix,iy]

    feey = econd + econv
    feiy = icond + iconv + ncond + nconv

    ix = com.isixcore == 1
    x_mm = com.yyc * 1000
   
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    # --- Ion radial fluxes ---
    axes[0].plot(x_mm, np.sum(bbb.fniy[ix, :, 0]/1e22, axis=0), 'k', label='UEDGE')
    axes[0].plot(x_mm, np.sum(fniydif[ix, :]/1e22, axis=0), label='An. Diff', c='blue')
    #axes[0].plot(x_mm, np.sum(fniyconv[ix, :]/1e22, axis=0), label='Convection', c='C1')
    axes[0].plot(x_mm, np.sum(fniyef[ix, :]/1e22, axis=0), label=r'$E\times B$', c='green', ls='--')
    axes[0].plot(x_mm, np.sum(fniybf[ix, :]/1e22, axis=0), label=r'$\nabla B$', c='magenta', ls='--')
    #axes[0].plot(x_mm, np.sum(bbb.fngy[:, :, 0][ix, :]/1e22, axis=0), label='Neutral flux', c='C4', ls='-.')
    axes[0].plot(x_mm, np.sum((fniydif + fniyef + fniybf)[ix, :]/1e22, axis=0), label='Sum', c='r', ls=':')
    axes[0].set_ylabel('Ion Flux [10$^{22}$ s$^{-1}$]',fontsize= 16)
    #axes[0].set_title('Ion Radial Flux Components')
    axes[0].grid(True)
    axes[0].legend(loc='best', fontsize = 12, ncol = 2)

    # --- Ion + Neutral parallel heat flux ---
    total_feiy = feiy[ix, :] / 1e6
    axes[1].plot(x_mm, np.sum(total_feiy, axis=0), 'k', label='UEDGE')
    axes[1].plot(x_mm, np.sum(Qi_An_diff[ix, :]/1e6, axis=0), label='An. Diff', c='blue')
    axes[1].plot(x_mm, np.sum(Qi_EB[ix, :]/1e6, axis=0),  label=r'$E\times B$', c='green', ls='--')
    #axes[1].plot(x_mm, np.sum(icond[ix, :]/1e6, axis=0), label='Ion cond.', c='C0')
    axes[1].plot(x_mm, np.sum(Qi_gradB[ix, :]/1e6, axis=0), label=r'$\nabla B$', c='magenta', ls='--')
    
    axes[1].plot(x_mm, np.sum((Qi_gradB + Qi_EB + Qi_An_diff)[ix, :]/1e6, axis=0), label='Sum', c='r', ls=':')
    axes[1].set_ylabel('Power [MW]', fontsize= 16)
    #axes[1].set_title('Ion + Neutral Parallel Heat Flux')
    axes[1].grid(True)
    axes[1].legend(loc='best', fontsize = 12, ncol=2)

   
    for ax in axes:
        ylim = ax.get_ylim()
        maxabs = np.max(np.abs(ylim))
        ax.set_ylim([-maxabs, maxabs])
        for xline in x_mm:
            ax.axvline(xline, color='#ddd', lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=12)  
    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize =16)
    plt.tight_layout()
    plt.savefig('radial_flux_ion.png', dpi=300)
    plt.show()

def plotRadialFluxes_ion_electron():
    # --- Initialize arrays ---
    shape = (com.nx+2, com.ny+2)
    fniy = np.zeros(shape)
    fniydif = np.zeros(shape)
    fniyconv = np.zeros(shape)
    fniyef = np.zeros(shape)
    fniybf = np.zeros(shape)
    P = np.zeros(shape)

    vyconv = bbb.vcony[0] + bbb.vy_use[:, :, 0] + bbb.vy_cft[:, :, 0]
    vydif = bbb.vydd[:, :, 0] - vyconv

    for ix in range(com.nx+2):
        for iy in range(com.ny+1):
            t2 = bbb.niy0[ix, iy, 0] if bbb.vy[ix, iy, 0] > 0 else bbb.niy1[ix, iy, 0]
            fniy[ix, iy] = bbb.cnfy * bbb.vy[ix, iy, 0] * com.sy[ix, iy] * t2
            fniydif[ix, iy] = vydif[ix, iy] * com.sy[ix, iy] * t2
            fniyconv[ix, iy] = vyconv[ix, iy] * com.sy[ix, iy] * t2
            fniyef[ix, iy] = bbb.cfyef * bbb.vyce[ix, iy, 0] * com.sy[ix, iy] * t2
            fniybf[ix, iy] = bbb.cfybf * bbb.vycb[ix, iy, 0] * com.sy[ix, iy] * t2
            P[ix,iy] = 2.5*t2
     
            if bbb.vy[ix, iy, 0] * (bbb.ni[ix, iy, 0] - bbb.ni[ix, iy+1, 0]) < 0:
                fniy[ix, iy] /= (1 + (bbb.nlimiy[0]/bbb.ni[ix, iy+1, 0])**2 + (bbb.nlimiy[0]/bbb.ni[ix, iy, 0])**2)

    # --- Upwind utilities ---
    def upwind(f, p1, p2): return max(f, 0)*p1 + min(f, 0)*p2
    def upwindProxy(f, g, p1, p2): return max(f, 0)/f * g * p1 + min(f, 0)/f * g * p2 if f != 0 else 0

    # --- Heat flux decomposition ---
    feey = np.zeros(shape)
    feiy = np.zeros(shape)
    econd = np.zeros(shape)
    econv = np.zeros(shape)
    icond = np.zeros(shape)
    iconv = np.zeros(shape)
    ncond = np.zeros(shape)
    nconv = np.zeros(shape)
    conyn = com.sy * bbb.hcyn / com.dynog
    Qi_gradB = np.zeros(shape)
    Qi_EB = np.zeros(shape)
    Qi_An_diff = np.zeros(shape)
    Qe_gradB = np.zeros(shape)
    Qe_EB = np.zeros(shape)
    Qe_An_diff = np.zeros(shape)
    Qe_veycp = np.zeros(shape)

    for ix in range(com.nx+2):
        for iy in range(com.ny+1):
            econd[ix, iy] = -bbb.conye[ix, iy] * (bbb.te[ix, iy+1] - bbb.te[ix, iy])
            econv[ix, iy] = upwind(bbb.floye[ix, iy], bbb.te[ix, iy], bbb.te[ix, iy+1])
            ncond[ix, iy] = -conyn[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy])
            icond[ix, iy] = -bbb.conyi[ix, iy] * (bbb.ti[ix, iy+1] - bbb.ti[ix, iy]) - ncond[ix, iy]
            floyn = bbb.cfneut * bbb.cfneutsor_ei * 2.5 * bbb.fngy[ix, iy, 0]
            floyi = bbb.floyi[ix, iy] - floyn
            iconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyi, bbb.ti[ix, iy], bbb.ti[ix, iy+1])
            nconv[ix, iy] = upwindProxy(bbb.floyi[ix, iy], floyn, bbb.ti[ix, iy], bbb.ti[ix, iy+1])
            Qi_gradB[ix, iy] = P[ix,iy] *(bbb.ti[ix, iy])*bbb.vycb[ix,iy,0]*com.sy[ix,iy]
            Qi_EB[ix, iy] = P[ix,iy] *( bbb.ti[ix, iy])*bbb.vyce[ix,iy,0]*com.sy[ix,iy]
            Qi_An_diff[ix, iy] = P[ix,iy] *(bbb.ti[ix, iy]) *vydif[ix,iy]*com.sy[ix,iy]
            Qe_gradB[ix, iy] = 2.5*(0.5*(bbb.niy0[ix,iy,0]+bbb.niy1[ix,iy,0])) *(bbb.te[ix, iy])*bbb.vycb[ix,iy,0]*com.sy[ix,iy]
            Qe_EB[ix, iy] =  2.5*(0.5*(bbb.niy0[ix,iy,0]+bbb.niy1[ix,iy,0])) *( bbb.te[ix, iy])*bbb.vyce[ix,iy,0]*com.sy[ix,iy]
            Qe_An_diff[ix, iy] = 2.5*(0.5*(bbb.niy0[ix,iy,0]+bbb.niy1[ix,iy,0])) *(bbb.te[ix, iy]) *vydif[ix,iy]*com.sy[ix,iy]
            Qe_veycp[ix, iy] =0# 2.5*(0.5*(bbb.ney0[ix,iy]+bbb.ney1[ix,iy])) *(bbb.te[ix, iy]) *bbb.veycp[ix,iy]*com.sy[ix,iy]
            

    feey = econd + econv
    feiy = icond + iconv + ncond + nconv

    ix = com.isixcore == 1
    x_mm = com.yyc * 1000
   
    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)

    # --- Ion radial fluxes ---
    axes[0].plot(x_mm, np.sum(bbb.fniy[ix, :, 0]/1e22, axis=0), 'k', label='UEDGE')
    axes[0].plot(x_mm, np.sum(fniydif[ix, :]/1e22, axis=0), label='An. Diff', c='blue')
    #axes[0].plot(x_mm, np.sum(fniyconv[ix, :]/1e22, axis=0), label='Convection', c='C1')
    axes[0].plot(x_mm, np.sum(fniyef[ix, :]/1e22, axis=0), label=r'$E\times B$', c='green', ls='--')
    axes[0].plot(x_mm, np.sum(fniybf[ix, :]/1e22, axis=0), label=r'$\nabla B$', c='magenta', ls='--')
    #axes[0].plot(x_mm, np.sum(bbb.fngy[:, :, 0][ix, :]/1e22, axis=0), label='Neutral flux', c='C4', ls='-.')
    axes[0].plot(x_mm, np.sum((fniydif + fniyef + fniybf)[ix, :]/1e22, axis=0), label='Sum', c='r', ls=':')
    axes[0].set_ylabel('$\Gamma_{ion}$ [10$^{22}$ s$^{-1}$]',fontsize= 16)
    #axes[0].set_title('Ion Radial Flux Components')
    axes[0].grid(True)
    axes[0].legend(loc='best', fontsize = 12, ncol = 2)

    # --- Ion + Neutral parallel heat flux ---
    total_feiy = (bbb.feiy[ix, :]) / 1e6
    axes[1].plot(x_mm, np.sum(total_feiy, axis=0), 'k', label='UEDGE')
    axes[1].plot(x_mm, np.sum((Qi_An_diff[ix, :])/1e6, axis=0), label='An. Diff', c='blue')
    axes[1].plot(x_mm, np.sum((Qi_EB[ix, :] ) /1e6, axis=0),  label=r'$E\times B$', c='green', ls='--')
    #axes[1].plot(x_mm, np.sum(icond[ix, :]/1e6, axis=0), label='Ion cond.', c='C0')
    axes[1].plot(x_mm, np.sum((Qi_gradB[ix, :])/1e6, axis=0), label=r'$\nabla B$', c='magenta', ls='--')
    
    axes[1].plot(x_mm, np.sum((Qi_gradB +Qi_EB + Qi_An_diff)[ix, :]/1e6, axis=0), label='Sum', c='r', ls=':')
    axes[1].set_ylabel('q$_{ion}$ [MW]', fontsize= 16)
    #axes[1].set_title('Ion + Neutral Parallel Heat Flux')
    axes[1].grid(True)
    axes[1].set_ylim([0,3])
    axes[1].legend(loc='best', fontsize = 12, ncol=2)

    total_feiy = (bbb.feey[ix, :]) / 1e6
    axes[2].plot(x_mm, np.sum(total_feiy, axis=0), 'k', label='UEDGE')
    axes[2].plot(x_mm, np.sum((Qe_An_diff[ix, :])/1e6, axis=0), label='An. Diff', c='blue')
    axes[2].plot(x_mm, np.sum((Qe_EB[ix, :]) /1e6, axis=0),  label=r'$E\times B$', c='green', ls='--')
    axes[2].plot(x_mm, np.sum((Qe_gradB[ix, :])/1e6, axis=0), label=r'$\nabla B$', c='magenta', ls='--')
    axes[2].plot(x_mm, np.sum(Qe_veycp[ix, :]/1e6, axis=0), label='Diam.', c='C0')
    gradB = np.sum(Qe_gradB[ix, :], axis =0)/1e6 
    axes[2].plot(x_mm, np.sum((Qe_EB+Qe_An_diff+econd+Qe_veycp)[ix, :]/1e6, axis=0) + abs(gradB), label='Sum', c='r', ls=':')
    axes[2].set_ylabel('q$_{ele}$ [MW]', fontsize= 16)
    #axes[1].set_title('Ion + Neutral Parallel Heat Flux')
    axes[2].grid(True)
    axes[2].set_ylim([-2,3])
    #axes[2].legend(loc='best', fontsize = 12, ncol=2)

   
    for ax in axes:
        ylim = ax.get_ylim()
        maxabs = np.max(np.abs(ylim))
        ax.set_ylim([-maxabs, maxabs])
        for xline in x_mm:
            ax.axvline(xline, color='#ddd', lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=12)  
    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize =16)
    plt.tight_layout()
    plt.savefig('radial_flux_ion.png', dpi=300)
    plt.show()





def plot_radial_heat_flux_check(bbb, com, scale=1e6, species=0):

    

    veb = bbb.cfyef * bbb.vyce[:, :, species]
    vgradb = bbb.cfybf * bbb.veycb[:, :]
    vdif = bbb.vydd[:, :, species]
    vy = bbb.vy[:, :, species]
    term = 0#bbb.cfjve * (bbb.fqy / (com.sy * bbb.qe)) / (0.5*(bbb.ney0 + bbb.ney1))
    #term[np.isnan(term)] = 0.0


    qeb = 2.5 * (0.5*(bbb.ney0 + bbb.ney1)) * bbb.te * veb
    qgradb = 2.5 * (0.5*(bbb.ney0 + bbb.ney1)) * bbb.te * vgradb
    qdiff = 2.5 * (0.5*(bbb.ney0 + bbb.ney1)) * bbb.te * vdif
    qy = 2.5 * (0.5*(bbb.ney0 + bbb.ney1)) * bbb.te * vy
    term =2.5 * (0.5*(bbb.ney0 + bbb.ney1)) * bbb.te * term
 
    plt.figure(figsize=(5,3))
    plt.plot(com.yyc, qeb[bbb.ixmp,:]/1e6, label='E×B drift')
    plt.plot(com.yyc, qgradb[bbb.ixmp,:]/1e6, label='Grad-B drift')
    plt.plot(com.yyc, qdiff[bbb.ixmp,:]/1e6, label='Ano-diffusion')
    plt.plot(com.yyc, bbb.feey[bbb.ixmp,:]/1e6, label='UEDGE', linestyle='--', color='k')
    plt.plot(com.yyc, (qeb+qgradb+qdiff+term)[bbb.ixmp,:]/1e6, label='Sum', linestyle='--', color='r')
    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)')
    plt.ylabel(r'q_${radial}$[MW /m$^2$]')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    qeb_int = np.sum(qeb * com.sy, axis=0) / scale
    qgradb_int = np.sum(qgradb * com.sy, axis=0) / scale
    qdiff_int = np.sum(qdiff * com.sy, axis=0) / scale
    qy_int = np.sum(qy * com.sy, axis=0) / scale
    term_int = np.sum(term* com.sy, axis=0) / scale

    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.plot(com.yyc, qeb_int, label='E×B drift')
    plt.plot(com.yyc, qgradb_int, label='Grad-B drift')
    plt.plot(com.yyc, qdiff_int, label='Anomalous diffusion')
    plt.plot(com.yyc, qy_int, label='Total vy flux', linestyle='--', color='k')
    plt.plot(com.yyc, qeb_int+ qgradb_int+qdiff_int-term_int, label='Sum', linestyle='--', color='r')
    #plt.plot(com.yyc, qeb_int+ abs(qgradb_int)+qdiff_int-term_int, label='Sum', linestyle='--', color='r')
    plt.xlabel('y (m)')
    plt.ylabel(f'Radial heat flux [W / {scale:.0e}]')
    plt.title('Radial Heat Flux Components')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Return arrays in case further processing is needed ---
    #return qeb_int, qgradb_int, qdiff_int, qy_int




def plotRadialFluxes_ion_electron_currents(bbb, com):
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Initialize shapes ---
    shape = (com.nx+2, com.ny+2)
    ix_mask = com.isixcore == 1
    x_mm = com.yyc * 1000  # convert to mm

    # --- Ion radial fluxes ---
    t2 = np.where(bbb.vy[:,:,0] > 0, bbb.niy0[:,:,0], bbb.niy1[:,:,0])
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0] - vyconv

    fniyconv = vyconv * com.sy * t2
    fniydif  = vydif * com.sy * t2
    fniyef   = bbb.cfyef * bbb.vyce[:,:,0] * com.sy * t2
    fniybf   = bbb.cfybf * bbb.vycb[:,:,0] * com.sy * t2
    fniy     = bbb.cnfy * bbb.vy[:,:,0] * com.sy * t2

    # --- Upwind correction ---
    factor = 1 + (bbb.nlimiy[0]/bbb.ni[:,:,1])**2 + (bbb.nlimiy[0]/bbb.ni[:,:,0])**2
    fniy *= np.where(bbb.vy[:,:,0]*(bbb.ni[:,:,0]-bbb.ni[:,:,1]) < 0, 1/factor, 1)

    # --- Heat flux decomposition ---
    P = 2.5 * t2
    ney_avg = 0.5*(bbb.ney0[:,:] + bbb.ney1[:,:])

    # Ion heat flux
    Qi_gradB   = P * bbb.ti[:,:] * bbb.vycb[:,:,0] * com.sy
    Qi_EB      = P * bbb.ti[:,:] * bbb.vyce[:,:,0] * com.sy
    Qi_An_diff = P * bbb.ti[:,:] * vydif * com.sy

    # Electron heat flux
    Qe_gradB   = 2.5 * t2 * bbb.te[:,:] * bbb.vycb[:,:,0] * com.sy
    Qe_EB      = 2.5 * t2 * bbb.te[:,:] * bbb.vyce[:,:,0] * com.sy
    Qe_An_diff = 2.5 * t2 * bbb.te[:,:] * vydif * com.sy

    # --- Radial current effect (veycp) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        Qe_fqy = -bbb.cfjve * (bbb.fqy / (com.sy * bbb.qe)) / ney_avg
        Qe_fqy[np.isnan(Qe_fqy)] = 0.0

    Qe_veycp = 2.5 * ney_avg * bbb.te[:,:] * Qe_fqy * com.sy

    # --- Total UEDGE fluxes ---
    total_feiy = bbb.feiy[ix_mask,:] / 1e6
    total_feey = bbb.feey[ix_mask,:] / 1e6

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Ion particle flux
    axes[0].plot(x_mm, np.sum(bbb.fniy[ix_mask,:,0]/1e22, axis=0), 'k', label='UEDGE')
    axes[0].plot(x_mm, np.sum(fniydif[ix_mask,:]/1e22, axis=0), c='blue', label='An. Diff')
    axes[0].plot(x_mm, np.sum(fniyef[ix_mask,:]/1e22, axis=0), c='green', ls='--', label=r'$E\times B$')
    axes[0].plot(x_mm, np.sum(fniybf[ix_mask,:]/1e22, axis=0), c='magenta', ls='--', label=r'$\nabla B$')
    axes[0].plot(x_mm, np.sum((fniydif+fniyef+fniybf)[ix_mask,:]/1e22, axis=0), 'r:', label='Sum')
    axes[0].set_ylabel(r'$\Gamma_{ion}$ [10$^{22}$ s$^{-1}$]', fontsize=14)
    axes[0].grid(True)
    axes[0].set_ylim([0, 5])
    axes[0].legend(loc='best', fontsize=10)

    # Ion heat flux
    axes[1].plot(x_mm, np.sum(total_feiy, axis=0), 'k', label='UEDGE')
    axes[1].plot(x_mm, np.sum(Qi_An_diff[ix_mask,:]/1e6, axis=0), c='blue', label='An. Diff')
    axes[1].plot(x_mm, np.sum(Qi_EB[ix_mask,:]/1e6, axis=0), c='green', ls='--', label=r'$E\times B$')
    axes[1].plot(x_mm, np.sum(Qi_gradB[ix_mask,:]/1e6, axis=0), c='magenta', ls='--', label=r'$\nabla B$')
    axes[1].plot(x_mm, np.sum((Qi_An_diff+Qi_EB+Qi_gradB)[ix_mask,:]/1e6, axis=0), 'r:', label='Sum')
    axes[1].set_ylabel('q$_{ion}$ [MW]', fontsize=14)
    axes[1].grid(True)
    axes[1].set_ylim([0,3])
    axes[1].legend(loc='best', fontsize=10)

    # Electron heat flux
    axes[2].plot(x_mm, np.sum(total_feey, axis=0), 'k', label='UEDGE')
    axes[2].plot(x_mm, np.sum(Qe_An_diff[ix_mask,:]/1e6, axis=0), c='blue', label='An. Diff')
    axes[2].plot(x_mm, np.sum(Qe_EB[ix_mask,:]/1e6, axis=0), c='green', ls='--', label=r'$E\times B$')
    axes[2].plot(x_mm, np.sum(Qe_gradB[ix_mask,:]/1e6, axis=0), c='magenta', ls='--', label=r'$\nabla B$')
    axes[2].plot(x_mm, np.sum(Qe_veycp[ix_mask,:]/1e6, axis=0), c='C0', label='Radial Current')
    # Sum including radial current
    axes[2].plot(x_mm, np.sum((Qe_An_diff+Qe_EB+Qe_veycp)[ix_mask,:]/1e6, axis=0) + np.abs(np.sum(Qe_gradB[ix_mask,:]/1e6, axis=0)),
                 'r:', label='Sum')
    axes[2].set_ylabel('q$_{ele}$ [MW]', fontsize=14)
    axes[2].grid(True)
    axes[2].set_ylim([-1,3])
    axes[2].legend(loc='best', fontsize=10,ncol=1)

    # --- Common formatting ---
    for ax in axes:
        ylim = ax.get_ylim()
        maxabs = np.max(np.abs(ylim))
       # ax.set_ylim([-maxabs, maxabs])
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for xline in x_mm:
            ax.axvline(xline, color='#ddd', lw=0.5, zorder=0)
    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
    plt.tight_layout()
    plt.savefig('radial_flux_ion_electron.png', dpi=300)
    plt.show()

def plotRadialFluxes_ion_electron_currents_all(bbb, com):
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Initialize shapes ---
    shape = (com.nx+2, com.ny+2)
    ix_mask = com.isixcore == 1
    x_mm = com.yyc * 1000  # convert to mm

    # --- Ion radial fluxes ---
    t2 = np.where(bbb.vy[:,:,0] > 0, bbb.niy0[:,:,0], bbb.niy1[:,:,0])
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0] - vyconv

    fniyconv = vyconv * com.sy * t2
    fniydif  = vydif * com.sy * t2
    fniyef   = bbb.cfyef * bbb.vyce[:,:,0] * com.sy * t2
    fniybf   = bbb.cfybf * bbb.vycb[:,:,0] * com.sy * t2
    fniy     = bbb.cnfy * bbb.vy[:,:,0] * com.sy * t2

    # --- Upwind correction ---
    factor = 1 + (bbb.nlimiy[0]/bbb.ni[:,:,1])**2 + (bbb.nlimiy[0]/bbb.ni[:,:,0])**2
    fniy *= np.where(bbb.vy[:,:,0]*(bbb.ni[:,:,0]-bbb.ni[:,:,1]) < 0, 1/factor, 1)

    # --- Heat flux decomposition ---
    P = 2.5 * t2
    ney_avg = 0.5*(bbb.ney0[:,:] + bbb.ney1[:,:])

    # Ion heat flux
    Qi_gradB   = P * bbb.ti[:,:] * bbb.vycb[:,:,0] * com.sy
    Qi_EB      = P * bbb.ti[:,:] * bbb.vyce[:,:,0] * com.sy
    Qi_An_diff = P * bbb.ti[:,:] * vydif * com.sy

    # Electron heat flux
    Qe_gradB   = 2.5 * t2 * bbb.te[:,:] * bbb.vycb[:,:,0] * com.sy
    Qe_EB      = 2.5 * t2 * bbb.te[:,:] * bbb.vyce[:,:,0] * com.sy
    Qe_An_diff = 2.5 * t2 * bbb.te[:,:] * vydif * com.sy

    # --- Radial current effect (veycp) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        Qe_fqy = -bbb.cfjve * (bbb.fqy / (com.sy * bbb.qe)) / ney_avg
        Qe_fqy[np.isnan(Qe_fqy)] = 0.0

    Qe_veycp = 2.5 * ney_avg * bbb.te[:,:] * Qe_fqy * com.sy

    # --- Total UEDGE fluxes ---
    total_feiy = bbb.feiy[ix_mask,:] / 1e6
    total_feey = bbb.feey[ix_mask,:] / 1e6

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    # Ion particle flux
    axes[0].plot(x_mm, np.sum(bbb.fniy[ix_mask,:,0]/1e22, axis=0), 'k', label='UEDGE')
    axes[0].plot(x_mm, np.sum(fniydif[ix_mask,:]/1e22, axis=0), c='blue', label='An. Diff')
    axes[0].plot(x_mm, np.sum(fniyef[ix_mask,:]/1e22, axis=0), c='green', ls='--', label=r'$E\times B$')
    axes[0].plot(x_mm, np.sum(fniybf[ix_mask,:]/1e22, axis=0), c='magenta', ls='--', label=r'$\nabla B$')
    axes[0].plot(x_mm, np.sum((fniydif+fniyef+fniybf)[ix_mask,:]/1e22, axis=0), 'r:', label='Sum')
    axes[0].set_ylabel(r'$\Gamma_{ion}$ [10$^{22}$ s$^{-1}$]', fontsize=14)
    axes[0].grid(True)
    axes[0].set_ylim([0, 5])
    axes[0].legend(loc='best', fontsize=10)
    ele = Qe_An_diff+Qe_EB+Qe_veycp+Qe_gradB
    # Ion heat flux
    axes[1].plot(x_mm, np.sum(total_feiy+total_feey, axis=0), 'k', label='UEDGE')
    axes[1].plot(x_mm, np.sum((Qi_An_diff + Qe_An_diff)[ix_mask,:]/1e6, axis=0), c='blue', label='An. Diff')
    axes[1].plot(x_mm, np.sum((Qi_EB+Qe_EB)[ix_mask,:]/1e6, axis=0), c='green', ls='--', label=r'$E\times B$')
    axes[1].plot(x_mm, np.sum((Qi_gradB+Qe_gradB)[ix_mask,:]/1e6, axis=0), c='magenta', ls='--', label=r'$\nabla B$')
    axes[1].plot(x_mm, np.sum((Qi_An_diff+Qi_EB+Qi_gradB+ele)[ix_mask,:]/1e6, axis=0), 'r:', label='Sum')
    axes[1].set_ylabel('q$_{radial}^{conv}$ [MW]', fontsize=14)
    axes[1].grid(True)
    axes[1].set_ylim([0,6])
    axes[1].legend(loc='best', fontsize=10)

  
    # --- Common formatting ---
    for ax in axes:
        ylim = ax.get_ylim()
        maxabs = np.max(np.abs(ylim))
       # ax.set_ylim([-maxabs, maxabs])
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for xline in x_mm:
            ax.axvline(xline, color='#ddd', lw=0.5, zorder=0)
    plt.xlabel('r$_{omp}$ - r$_{sep}$ (m)', fontsize=14)
    plt.tight_layout()
    plt.savefig('radial_flux_ion_electron.png', dpi=300)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt

def plot_poloidal_heat_flux_ion(bbb, com, mid_idx=com.iysptrx+1, savefig=True):
    """
    Compute and plot ion poloidal heat flux components.
    """

    # --- Shapes ---
    shape = (com.nx+2, com.ny+2)

    icond = np.zeros(shape)
    iconv = np.zeros(shape)
    poloidal_indices = analysis.para_conn_length(bbb,com)[:, mid_idx]

    # --- Upwind functions ---
    def upwind(f, p1, p2):   return max(f, 0)*p1 + min(f, 0)*p2

    def upwindProxy(f, g, p1, p2): return max(f, 0)/f * g * p1 + min(f, 0)/f * g * p2 if f != 0 else 0

    for ix in range(com.nx+1):
        for iy in range(com.ny):
 
            icond[ix, iy] = -bbb.conxi[ix, iy] * (bbb.ti[ix+1, iy] - bbb.ti[ix, iy])
            iconv[ix, iy] = upwindProxy(bbb.floxi[ix, iy], bbb.floxi[ix, iy], bbb.ti[ix, iy], bbb.ti[ix+1, iy])

    # --- Ion heat flux components ---
    qion_gradb = 2.5 * bbb.ni[:, :, 0] * bbb.v2cb[:, :, 0] * bbb.rbfbt * com.sx * bbb.ti
    qion_EB    = 2.5 * bbb.ni[:, :, 0] * bbb.v2ce[:, :, 0] * bbb.rbfbt * com.sx * bbb.ti
    qion_conv  = 2.5 * bbb.ni[:, :, 0] * bbb.upi[:, :, 0]  * com.rrv   * com.sx * bbb.ti
    q_ion      = bbb.cfcvti * 2.5 * bbb.fnix[:, :, 0] * bbb.ti
    qion_sum   = qion_gradb + qion_EB + qion_conv #+ icond
 

    # --- Plot ---
    plt.figure(figsize=(5, 3.5))
    plt.plot(poloidal_indices[1:-1], qion_gradb[1:-1, mid_idx]/1e6, label='gradB', color='green')
    plt.plot(poloidal_indices[1:-1], qion_EB[1:-1, mid_idx]/1e6,    label='E×B',   color='magenta')
    plt.plot(poloidal_indices[1:-1], qion_conv[1:-1, mid_idx]/1e6,  label='2.5nTu', color='blue')
    #plt.plot(poloidal_indices[1:-1], icond[1:-1, mid_idx]/1e6,      label='Cond',  color='orange')
    plt.plot(poloidal_indices[1:-1], bbb.feix[1:-1, mid_idx]/1e6,   label='feix',  color='black')
    plt.plot(poloidal_indices[1:-1], qion_sum[1:-1, mid_idx]/1e6,   label='sum',   linestyle='--', color='red')

    plt.xlabel('Poloidal grid index', fontsize=16)
    plt.ylabel(r'$q_\mathrm{poloidal}^{ion}$ [MW]', fontsize=16)
    plt.axvline(poloidal_indices[bbb.ixmp], color='black', linestyle=':', linewidth=2)
    plt.legend(fontsize=12, ncol=2)
    plt.xlim([0, 65])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.tight_layout()

    if savefig:
        plt.savefig("poloidal_heat_flux_ion.png", dpi=300)

    plt.show()

    #return qion_gradb, qion_EB, qion_conv, icond, qion_sum




def plot_li_flux(bbb, com, savefile="liflux.png", show=True):
    """
    Plot Li particle flux streamlines (vz, vr) from a UEDGE solution.

    Parameters
    ----------
    bbb : object
        UEDGE bbb object containing fnix and fniy.
    com : object
        UEDGE com object containing grid info (sx, sy, ixpt, ixmp, etc.).
    savefile : str or None
        If given, save the figure to this filename.
    show : bool
        If True, display the plot with plt.show().
    """

    # --- Extract grid markers ---
    omp_cell = bbb.ixmp
    oxpt = com.ixpt2[0]
    ixpt = com.ixpt1[0]
    separatrix_cell = com.iysptrx

    # --- Li particle flux (poloidal & radial) ---
    vz_flux = (bbb.fnix[:,:,com.nhsp] + bbb.fnix[:,:,com.nhsp+1] + bbb.fnix[:,:,com.nhsp+2]) / com.sx
    vr_flux = (bbb.fniy[:,:,com.nhsp] + bbb.fniy[:,:,com.nhsp+1] + bbb.fniy[:,:,com.nhsp+2]) / com.sy

    # --- Grid ---
    nx, ny = vr_flux.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # --- Plot flux flow ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # Streamlines only
    ax.streamplot(X, Y, vz_flux.T, vr_flux.T,
                  color='k', density=1.2, arrowsize=1.2)

    # --- Mark important locations ---
    if omp_cell is not None and 0 <= omp_cell-1 < nx:
        ax.axvline(omp_cell - 1, color='red', linestyle='--', linewidth=2)
        ax.text(omp_cell - 1, ny, "OMP", color='red', fontsize=12,
                ha='center', va='bottom', fontweight='bold')

    if oxpt is not None and 0 <= oxpt-1 < nx:
        ax.axvline(oxpt - 1, color='orange', linestyle='--', linewidth=2)
        ax.text(oxpt - 1, ny, "Oxpt", color='orange', fontsize=12,
                ha='center', va='bottom', fontweight='bold')

    if ixpt is not None:
        ax.axvline(ixpt - 1, color='green', linestyle='--', linewidth=2)
        ax.text(ixpt - 1, ny, "Ixpt", color='green', fontsize=12,
                ha='center', va='bottom', fontweight='bold')

    if separatrix_cell is not None and 0 <= separatrix_cell-1 < ny:
        ax.axhline(separatrix_cell - 1, color='magenta', linestyle='-.', linewidth=2)
        ax.text(0, separatrix_cell - 1, "Sep", color='magenta', fontsize=12,
                ha='left', va='bottom', fontweight='bold')

    # --- Labels ---
    ax.set_xlabel('Poloidal cell', fontsize=14)
    ax.set_ylabel('Radial cell', fontsize=14)
    ax.set_title(r'Li particle flux direction ($\Gamma_{Li}$)', fontsize=14)

    plt.xlim([0, nx])
    plt.ylim([0, ny])
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)



