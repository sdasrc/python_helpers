import os, sys
import numpy as np


def get_beam(img, wcs, beam_deg):
    '''
    Input : full image array, wcs from the image header, and [bmaj,bmin] in degrees
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.visualization import make_lupton_rgb
    from astropy.coordinates import Angle
    import astropy.units as u
    from astropy.wcs import WCS
    
    # Get the shape of the image
    image_shape = img.shape

    # Define the pixel coordinates for the four corners of the image
    pixel_coords = np.array([[0, 0], [0, image_shape[1]], 
                             [image_shape[0], 0], [image_shape[0], image_shape[1]]])

    # Convert pixel coordinates to RA and DEC using wcsV5
    ra_dec_coords = wcs.all_pix2world(pixel_coords, 0)

    # Extract the RA and DEC values for the four corners
    ra_values = ra_dec_coords[:, 0]
    dec_values = ra_dec_coords[:, 1]

    # Find the minimum and maximum RA and DEC
    imsizex_deg = np.max(ra_values) - np.min(ra_values) # imsize in degree
    imsizex_cell = image_shape[0]
    imsizey_deg = np.max(dec_values) - np.min(dec_values) # imsize in degree
    imsizey_cell = image_shape[1]
    
    return np.array([beam_deg[0]*(imsizex_cell/imsizex_deg), beam_deg[1]*(imsizey_cell/imsizey_deg)])


def get_levs(tpeak, nlevs = 10, cutoff = 0, ascending=True):
    import numpy as np
    '''
    returns contour levels starting at 90\% of peak
    going down by a factor of two
    till either the cuff (in abs flux) or number of 
    levels are met. 
    '''
    pcntrs = []
    acntrs = []
    pstr, astr = '', ''
    if not nlevs: nlevs = 100
    for ii in range(nlevs):
        pcntrs.append(90/2**ii)
        acntrs.append(tpeak*(90/2**ii))
        if tpeak*(90/2**ii) < cutoff: break

    pcntrs.append(-1*pcntrs[-1])
    acntrs.append(-1*acntrs[-1])

    if ascending:
        pcntrs.reverse()
        acntrs.reverse()

    for ii in range(len(pcntrs)-1):
        pstr=pstr+'{0:.3g}'.format(pcntrs[ii])+','   
        astr=astr+'{0:.3g}'.format(acntrs[ii])+','      

    pstr=pstr+'{0:.3g}'.format(pcntrs[-1])
    astr=astr+'{0:.3g}'.format(acntrs[-1]/100)

    print('abs : ',astr)
    print('percent : ',pstr)

    return np.array(pcntrs), np.array(acntrs)/100