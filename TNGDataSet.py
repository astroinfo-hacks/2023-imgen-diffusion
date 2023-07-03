import os
import glob
import tensorflow as tf
import h5py as h5py
import yaml as tfds
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd




_DESCRIPTION = """
#Data representing the TNG50, TNG100, and TNG300 Simulations
"""

_CITATION = ""
_URL = "https://github.com/astroinfo-hacks/2023-imgen-diffusion"


def ReturnFiles(dir):
   for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        yield (os.path.join(root, name))


class TNGDataSet(tfds.core.GeneratorBasedBuilder):
  """Eagle galaxy dataset"""  

  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {'1.0.0': 'Initial release.',}
  MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    N_TIMESTEPS = 280
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        homepage=_URL,
        citation=_CITATION,
        # Two features: image with 3 channels (stellar light, velocity map, velocity dispersion map)
        #  and redshift value of last major merger
        features=tfds.features.FeaturesDict({
            #'noiseless_griz': tfds.features.Tensor(shape=(128, 128, 4), dtype=tf.float32),
            #'stellar_light': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            #'velocity_map': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            #'velocity_dispersion_map': tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
            "sed": tfds.features.Tensor(shape=(143,), dtype=tf.float32),
            "time": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            #"SFR_halfRad": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            #"SFR_Rad": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "SFR_Max": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            #"Mstar_Half": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            "Mstar": tfds.features.Tensor(shape=(N_TIMESTEPS,), dtype=tf.dtypes.float32),
            'mass_quantiles': tfds.features.Tensor(shape=(9,), dtype=tf.float32),
            'last_over_max': tf.float32,
            #'last_major_merger': tf.float32,
            'object_id': tf.int32
        }),
        supervised_keys=('noiseless_griz', 'last_major_merger'), 
    )

  def _split_generators(self, p):
    """Returns generators according to split"""
    return {tfds.Split.TRAIN: self._generate_examples(str(dl.manual_dir))}

  def _generate_examples(self, root_path):
    """Yields examples."""

    
    # read EAGLE hd5y + filter names + filter wavelength
    hf = h5py.File(root_path+'/dataMagnitudes_2000kpc_EMILES_PDXX_DUST_CH_028_z000p000.hdf5', 'r')
    wl = np.loadtxt(root_path+"/wl.csv")
    text_file = open(root_path+"/fnames.csv", "r")
    fname_list = text_file.readlines()
    sfh = hf.get('Data/SFhistory')
    tbins = hf.get('Data/SFbins')
    time = (tbins[1:] + tbins[:-1] )/2.

    mstar  = hf.get('Data/StellarMassNew')  
    
    # sfh
    sfh = hf.get('Data/SFhistory')
    nobjects = sfh.shape[0]
    tbins = hf.get('Data/SFbins')
    time = (tbins[1:] + tbins[:-1] )/2.
    deltat=tbins[1:] - tbins[:-1]
   
        
   

    for i in range(len(mstar)):
        object_id = i

        if True:
            
            
            if np.log10(mstar[i])<9.5:
                continue

            # sed
    
            mag = [] 
            for f in fname_list:
                mag.append(hf['Data'][f.strip()][1])
            app_mag = np.array(mag)+5*(np.log10(20e6)-1) #assume at 20pc
            flux = 10**(.4*(-app_mag+8.90)) #convert to Jy
            
            example = {'sed': flux}
            
            #example.update({'sed': np.array(flux).astype('float32')})

        
            #mstar growth
     
    
            mgrowth = np.cumsum(deltat*sfh[i])
            example.update({'Mstar': np.array(mgrowth).astype('float32')})
   

            #sfh
    
            example.update({'time': np.array(time).astype('float32')})
            example.update({'SFR_Max': np.array(sfh[i]).astype('float32')})
            
            
            #quantiles
            mass_history_summaries = find_summaries(example['Mstar'],
                                                example['time'])
            last_over_max = example['Mstar'][0]/np.max(example['Mstar'])
            example.update({'mass_quantiles': mass_history_summaries,
                        'last_over_max': last_over_max,
                        'object_id': object_id})
            
            
            


    

            yield object_id, example
        else:
            continue      