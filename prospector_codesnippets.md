## Import data for radio selected galaxy catalogue

		def data_import(fits_fname='EN1_sedfit_v1.0.fits'):
		    from astropy.io import fits
		    import numpy as np

		    fitshdu = fits.open(fits_fname)
		    fitsarr = fitshdu[1].data
		    fitscols = fitshdu[1].columns
		    fitskeys = {fitscols[ii].name:ii for ii in range(len(fitscols)) }

		    fluxkeys = np.array([fitskeys[ff] for ff in fitskeys if '_f' in ff])
		    flux_e_keys = fluxkeys +1
		    fitsdict = {}
		    cnt = 0
		    for ths in tqdm_notebook(fitsarr):
		        if (np.isfinite(ths[fitskeys['Z_BEST']])  
		                    and ths[fitskeys['FLAG_GOOD']]==True 
		                    and ths[fitskeys['FLAG_OVERLAP']]==7):   
		            fluxarr = np.array([ths[jj]/3631.  for jj in fluxkeys])
		            fluxerr = np.array([ths[jj]/3631.  for jj in flux_e_keys]) 
		            fitsdict[cnt] = ( {
		                'radio_id' : int(ths[fitskeys['radioID']]), 
		                'sour_id' : int(ths[fitskeys['ID']]), 
		                'sour_name' : str(ths[fitskeys['Source_Name']]), 
		                'z_best' : ths[fitskeys['Z_BEST']], 
		                'flux_maggies' : fluxarr, 
		                'flux_err_maggies' : fluxerr,
		                'nbands' : np.count_nonzero(np.isfinite(fluxarr))
		            } )
		            cnt+=1


		    filterlist = np.array([ff[:-2] for ff in fitskeys if '_f' in ff])

		    return fitsdict, filterlist
