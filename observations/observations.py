from astropy.io import fits
import numpy as np
import os.path as osp
import logging


# s_logger = logging.getLogger(__file__)
HSC_PIX = 0.17


class MockObservedImage:
    """
    
    """
    stages = ['raw', 'ab_mag', 'jansky', 'observed']
    def __init__(self, filename=None):
        self.image = None
        self.size = None
        self.stage = None
        self.image_filename = filename
        self.logger = logging.getLogger(__name__)
        if self.image_filename:
            self.set_image()
        

    def set_image(self, filename=None):
        """
        
        """
        if not self.image_filename and not filename:
            self.logger.error(f"No filename specified")
            return False
        elif not self.image_filename:
            if not osp.isfile(filename):
                self.logger.error(f"Bad filename: {filename}")
                return False
            self.image_filename = filename

        with fits.open(self.image_filename) as l_hdu:
            self.image = l_hdu[1].data
            l_head = l_hdu[0].header
            d_head = l_hdu[1].header

        self.flux_mag = np.float32(l_head["FLUXMAG0"])
        self.size = (d_head["NAXIS1"], d_head["NAXIS2"])
        self.stage = 'raw'
        return True

    def raw_to_abmag(self):
        if not self.image:
            self.logger.error(f"Not image already loaded!")
            return False
        elif self.stage != 'raw':
            self.logger.error(f"Image is not in raw format!")
            return False
        
        self.ab_mag_image = 2.5*np.log(self.flux_mag/self.image)
        self.stage = 'ab_mag'

    def abmag_to_jansky(self):
        if not self.image:
            self.logger.error(f"Not image already loaded!")
            return False
        elif self.stage != 'ab_mag':
            self.logger.error(f"Image is not in AB magnitud format!")
            return False
        
        self.jansky_image = 10**(23 - (self.ab_mag_image + 48.6)/2.5)
        self.stage = 'jansky'

    def jansky_to_obs(self):
        if not self.image:
            self.logger.error(f"Not image already loaded!")
            return False
        elif self.stage != 'jansky':
            self.logger.error(f"Image is not in AB magnitud format!")
            return False
            
        self.obs_image = self.jansky_image/HSC_PIX
        self.stage = 'observed'

    def add_noise(self, sigma=1.0):
        self.obs_image += np.random.normal(1,sigma,self.shape)




