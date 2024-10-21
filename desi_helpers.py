import os,sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, "/home/overlord/git/desi/desispec/py")
sys.path.insert(1, "/home/overlord/git/desi/desiutil/py")
sys.path.insert(1, "/home/overlord/git/desi/desimodel/py")
sys.path.insert(1, "/home/overlord/git/desi/desitarget/py")

from astropy.convolution import convolve, Gaussian1DKernel
import desispec
from desispec import io, coaddition
from astropy.io import fits

import tqdm

class CatalogueHandler:
    def __init__(self, cat_path='/beegfs/car/sdas/desi/catalogues/EN1_DESI_CROSSMATCH_MS.FITS',
                match_cache='/beegfs/car/sdas/desi/catalogues/EN1_DESI_CROSSMATCH_MS.txt'):
        self.cat_path = cat_path
        self.match_cache = match_cache
        self.catdict, self.catkeys = self.load_catalogue()

    def load_catalogue(self):
        cathdu = fits.open(self.cat_path)
        catarr = cathdu[1].data
        catcols = cathdu[1].columns
        catkeys = {catcols[ii].name: ii for ii in range(len(catcols))}
        catdict = {ths[catkeys['TARGETID']]: ths for ths in catarr}
        if not os.path.isfile(self.match_cache):
            desi_ids = np.array([catdict[xx][catkeys['TARGETID']] for xx in catdict.keys()])
            lotss_ids = np.array([catdict[xx][catkeys['ID']] for xx in catdict.keys()])
            np.savetxt(self.match_cache,np.array(list(zip(desi_ids.astype('int'), lotss_ids.astype('int')))),fmt='%i')
        return catdict, catkeys

    def query_obj(self, lotssid):
        if os.path.isfile(self.match_cache):
            matched_dat = np.loadtxt(self.match_cache, dtype='int')
            ids, target = matched_dat[:,0], matched_dat[:,1]
        else:    
            ids = np.array(list(self.catdict.keys()))
            target = np.array([self.catdict[xx][self.catkeys['ID']] for xx in self.catdict.keys()])
        tid = np.where(target == lotssid)[0]
        return ids[tid[0]] if len(tid) else None

    def reverse_query_obj(self, targetid):
        if os.path.isfile(self.match_cache):
            matched_dat = np.loadtxt(self.match_cache, dtype='int')
            target, ids  = matched_dat[:,0], matched_dat[:,1]
        else:  
            ids = np.array([self.catdict[xx][self.catkeys['ID']] for xx in self.catdict.keys()])
            target = np.array([self.catdict[xx][self.catkeys['TARGETID']] for xx in self.catdict.keys()])
        tid = np.where(target == targetid)[0]
        return ids[tid[0]] if len(tid) else None

    def get_object(self, targetid):
        tdat = self.catdict[targetid]
        tdict = {key: tdat[self.catkeys[key]] for key in ['ID', 'TARGETID', 'SURVEY', 'PROGRAM',
                                                    'HEALPIX', 'TARGET_RA', 'TARGET_DEC', 'Z', 'ZERR',
                                                    'ZWARN', 'Z_BEST', 'Z_SPEC', 'SPECTYPE', 'SUBTYPE']}
        return tdict

class TileHandler:
    def __init__(self, tile_dir='/beegfs/car/sdas/desi/healpix/sv3/', 
        coadd_cache ='/beegfs/car/sdas/desi/healpix/sv3/targetid_coadd_matches.txt'):
        self.tile_dir = tile_dir
        self.coadd_cache = coadd_cache

    def list_tiles(self):
        if os.path.isfile(self.coadd_cache):
            coadd_cached = np.loadtxt(self.coadd_cache, dtype='str')
            return coadd_cached[:,0].astype('int'), coadd_cached[:,1].astype('str')
        else:
            objs, coadds = np.array([]), np.array([])
            tiles = [d for d in os.listdir(self.tile_dir) if os.path.isdir(os.path.join(self.tile_dir, d))]
    
            for tt in tiles:
                ldates = [d for d in os.listdir(os.path.join(self.tile_dir, tt)) if os.path.isdir(os.path.join(self.tile_dir, tt, d))]
                for dd in ldates:
                    parent_dir = os.path.join(self.tile_dir, tt, dd)
                    all_files = os.listdir(parent_dir)
                    coadd_files = [file for file in all_files if 'coadd' in file.lower()]
                    for coadd_file in coadd_files:
                        thdu = fits.open(os.path.join(parent_dir, coadd_file))
                        tdata = thdu[1].data
                        coadds = np.concatenate([coadds, np.full(len(tdata), os.path.join(parent_dir, coadd_file))])
                        objs = np.concatenate([objs, np.array(list(str(ff[0]) for ff in tdata))])

            np.savetxt(self.coadd_cache,np.array(list(zip(objs.astype('str'), coadds))),fmt='%s')
            return objs.astype('int'), coadds

    def check_spectra(self, cat_handler, targetid=None, lotssid=None):
        '''
        Given targetid, check if that spectra is cached/locally available
        1: Available, 0: not available, -1: faulty inputs
        '''
        if targetid is not None and np.isfinite(targetid):
            thisobj = cat_handler.get_object(targetid)
        elif lotssid is not None and np.isfinite(lotssid):
            targetid = cat_handler.query_obj(lotssid)
            thisobj = cat_handler.get_object(targetid)
        else:
            return -1
        objs, coadds = self.list_tiles()
        res = np.where(objs == thisobj['TARGETID'])[0]
        return int(len(res) > 0)            

    def which_coadd(self, lotssid, cat_handler):
        targetid = cat_handler.query_obj(lotssid)
        thisobj = cat_handler.get_object(targetid)
        objs, coadds = self.list_tiles()
        res = np.where(objs == thisobj['TARGETID'])[0]
        return coadds[res[0]] if len(res) else None

    def get_coadd(self, lotssid, cat_handler):
        coaddfile = self.which_coadd(lotssid, cat_handler)
        targetid = cat_handler.query_obj(lotssid)
        # Selecting the particular spectra of the targetid
        coadd_obj = desispec.io.read_spectra(coaddfile)
        coadd_tgts = coadd_obj.target_ids().data
        # Check if this target exists in the healpix coadd fits file
        row = (coadd_tgts == targetid)
        return coadd_obj[row]

    def get_spec(self, lotssid, cat_handler):
        coaddfile = self.which_coadd(lotssid, cat_handler)
        targetid = cat_handler.query_obj(lotssid)
        # Selecting the particular spectra of the targetid
        coadd_obj = desispec.io.read_spectra(coaddfile)
        coadd_tgts = coadd_obj.target_ids().data
        # Check if this target exists in the healpix coadd fits file
        row = (coadd_tgts == targetid)
        coadd_spec = coadd_obj[row]
        spec_combined = coaddition.coadd_cameras(coadd_spec)
        wave, flux, flux_err, ivar = spec_combined.wave['brz'], spec_combined.flux['brz'][0], 1/np.sqrt(spec_combined.ivar['brz'][0]), spec_combined.ivar['brz'][0]
        return wave, flux, flux_err, ivar


def plot_desi_spec(obj_details, wave, flux, flux_err, outfile=None):
    plt.figure()
    fig, ax = plt.subplots(1,1, figsize = (20, 6))
    survey, program, spectype, z = obj_details['SURVEY'], obj_details['PROGRAM'], obj_details['SPECTYPE'], obj_details['Z']   
    # Plot the combined spectrum in maroon
    ax.plot(wave, flux, color = 'maroon', alpha = 0.5)
    # Over-plotting smoothed spectra 
    ax.plot(wave, convolve(flux, Gaussian1DKernel(5)), color = 'k', lw = 2.0)
    ax.fill_between(wave, y1=flux - flux_err, y2=flux + flux_err,zorder=1, alpha=0.5, color='gray')
    ax.set_xlim([3500, 9900])
    ax.set_xlabel('$\lambda$ [$\AA$]')
    ax.set_ylabel('$F_{\lambda}$ [$10^{-17} erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]')
    ax.set_title('LOTSS : {0}, DESI : {1}, Z = {2:.3f}'.format(obj_details['ID'],
                                                              obj_details['TARGETID'],obj_details['Z']))
    trans = ax.get_xaxis_transform()
    ax.annotate(f'{survey}, {program}', xy = (6000, 0.85), xycoords = trans, fontsize = 16)
    ax.annotate(f'SPECTYPE : {spectype}', xy = (8000, 0.85), xycoords = trans, fontsize = 16)
    if outfile is not None:
        if outfile == '':
            pname = str(obj_details['ID'])+'_'+str(obj_details['TARGETID'])+'_'\
                  +obj_details['SURVEY']+'_'+obj_details['PROGRAM']+'_'+str(obj_details['HEALPIX'])+'.jpg'
        else: pname = outfile
        print(pname)
        plt.savefig(pname, bbox_inches='tight', dpi=300)

    # plt.close()