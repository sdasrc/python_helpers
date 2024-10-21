# --------------------------------------------------------------------------- #
#               G E T  S D S S   S P E C   F R O M   F I T S                  #
# ----------------------------------------------------------------------------#
def get_sdss_spec(fname=''):
    '''
    Given a valid sdss LITE spectra fit file, returns
    flux : Flambda [x 10^-17 ergs cm^-2 s^-1 A^-1]
    error in flux (same units)
    wavelengths in angstrom
    redshift,
    linenames,
    line wavelengths in Angstroms.
    '''

    import os
    import numpy as np
    from astropy.io import fits
    
    if not os.path.isfile(fname): return None, None, None, -1, None, None 
    
    sdsshdu = fits.open(fname)
    sdssdata = sdsshdu[1].data

    flux_wave = np.array([sdssdata[xx][0] for xx in range(len(sdssdata))])
    flux_var_wave = 1/np.sqrt(np.array([sdssdata[xx][2] for xx in range(len(sdssdata))]))

    waves = 10**np.array([sdssdata[xx][1] for xx in range(len(sdssdata))])  # in ang
    # freqs = 3.e18/waves   # in hz

    # get redshift info
    sdssinfo = sdsshdu[2].data
    sdssinfocols = np.array([xx.name for xx in sdsshdu[2].columns])
    zind = np.where(sdssinfocols == 'Z')[0][0]
    z = sdssinfo[0][zind]

    # get line names and freq
    sdsslines = sdsshdu[3].data
    linenames = np.array([sdsslines[xx][3] for xx in range(len(sdsslines))])
    linewave = np.array([sdsslines[xx][4] for xx in range(len(sdsslines))])

    return flux_wave, flux_var_wave, waves, z, linenames, linewave



# --------------------------------------------------------------------------- #
#            P L O T   I N D I V I D U A L  S D S S   S P E C                 #
# ----------------------------------------------------------------------------#
def plot_sdss_spec(fname='',objname='',outfile='',restframe=True,outpdf=True):
    '''
    Given an sdss spectra (lite), plots the spectra in rest frame,
    with a few lines. Returns redshift.
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    flux_wave, flux_var_wave, waves, z, linenames, linewave = get_sdss_spec(fname=fname)

    if z == -1: return -1
    
    if restframe:
        twave = waves/(1+z)
        wavetag = 'Rest-frame'
        pltag = '_rest'
    else:
        twave = waves
        wavetag = 'Observed-frame'
        pltag = '_obs'


    # plot
    if outpdf:
        plt.rcParams['figure.figsize'] = (12,7)
        plt.rcParams['font.size'] = 12
        outfile = outfile+pltag+'_z_{0:.3f}'.format(z)+'.pdf'
    else:
        plt.rcParams['figure.figsize'] = (25,14)
        plt.rcParams['font.size'] = 18
        outfile = outfile+pltag+'_z_{0:.3f}'.format(z)+'.jpg'
        
    plt.figure()

    cols = ['crimson','darkorange','dodgerblue','darkcyan','fuchsia','goldenrod','green']
    lslist = [':','--','-.']

    tlines = [5,6,12,15,16,17,18,21,24]
    uplim = np.nanmax(1.2*flux_wave)

    for ii in range(len(tlines)):
        tl = tlines[ii]
        labelpos = 0.7 if ii%2 else 0.5
        plt.axvline(linewave[tl], color=cols[len(cols) - ii%len(cols) - 1],
                    lw=2, ls=lslist[ii//len(cols)],alpha=0.9)
        t = plt.text(linewave[tl]+30,labelpos*uplim, linenames[tl], rotation=90, 
                 color=cols[len(cols) - ii%len(cols) - 1])
        t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='none'))

    plt.errorbar(x=twave,y=flux_wave,yerr=flux_var_wave,lw=1,color='gray',alpha=1)

    plt.plot(twave,flux_wave,lw=1,color='mediumblue',alpha=1,zorder=1000)
    # plt.legend(loc='best',alpha=0.2)

    # plt.yscale('log')
    
    plt.grid(color='gainsboro', linestyle='-', linewidth=1, alpha=0.3)

    plt.xlabel(wavetag + r' Wavelength [$\AA$]')
    plt.ylabel(r'$F_\lambda~[10^{-17}~ergs~cm^{-2}~s^{-1}~\AA^{-1}]$')
    plt.title('Obj Name : {0}, z : {1:.3f}'.format(objname,z))
    plt.tight_layout()

    plt.savefig(outfile)
    plt.close()
    
    return z