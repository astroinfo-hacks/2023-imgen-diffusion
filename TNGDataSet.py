import os
from random import choice
import tensorflow as tf
import h5py as h5py
import tensorflow_datasets as tfds
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union
from etils import epath
from pathlib import Path


def Logger(str,verbosity=0):
    """

      Simple Logger
      By default logs .. increase verbosity in case of well .. more verbose logs

      use 
      os.environ["DIFF_TRACE"]="1"
      in your notebook to change verbosity

    """
    if int(os.environ.get("DIFF_TRACE",0)) > verbosity:
        print(str)

_DESCRIPTION = """
#Data representing the TNG50, TNG100, and TNG300 Simulations
"""

_CITATION = ""
_URL = "https://github.com/astroinfo-hacks/2023-imgen-diffusion"

histo_grame = {}

def ScaleImage(img):
    return img[0:500,0:500].astype('float32')

class SubsplitDictionaries:
    def __init__(self):
        self.train_dict = []
        self.valid_dict = []
        self.test_dict = []
        self.full_list = []

    @staticmethod
    def CreateKey(EXTNAME, ORIGIN, SIMTAG, SNAPNUM, SUBHALO):
        return "{0}_{1}_{2}_{3}_{4}".format(EXTNAME, ORIGIN, SIMTAG, SNAPNUM, SUBHALO)
    
    # Sanity check (This is not needed .. TFDS does it already .. the key is sent back with the yield
    def AlreadyExisting(self,key_with_camera,Create=False):
        res =  key_with_camera not in self.full_list
        if Create:
          self.full_list.append(key_with_camera)
        return res

    def FindOrCreate(self,key, CAMERA,subsplit):
        #key_with_camera = "{0}_{1}".format(key,CAMERA)
        #assert (self.AlreadyExisting(key_with_camera,True))
        if subsplit == tfds.Split.TRAIN:
            if key not in self.valid_dict and key not in self.test_dict:
                Logger("Key {0} with camera code {1} will be assigned to training".format(key,CAMERA),3)
                if key not in self.train_dict:
                    self.train_dict.append(key)
                return False
        elif subsplit == tfds.Split.VALIDATION:
            if key not in self.train_dict and key not in self.test_dict:
                Logger("Key {0} with camera code {1} will be assigned to validation".format(key,CAMERA),3)
                if key not in self.valid_dict:
                    self.valid_dict.append(key)
                return False
        elif subsplit == tfds.Split.TEST:
            if key not in self.train_dict and key not in self.valid_dict:
                Logger("Key {0} with camera code {1} will be assigned to testing".format(key,CAMERA),3)
                if key not in self.test_dict:
                    self.test_dict.append(key)
                return False
        return True


def GetParent(path):
   return str(Path(path).parent.absolute())

class TNGDataSet(tfds.core.GeneratorBasedBuilder):
    """Eagle galaxy dataset"""  

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {'1.0.0': 'Initial release.',}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Nothing to download. Dataset was generated at first call."

    def __init__(
            self,input_dir, 
            train_percent=0.8, 
            val_percent=0.1, 
            test_percent=0.1,
            Scaler_fcn=ScaleImage ,
            generation_verbosity=1000,
            Image_Size=(150,150),
            band_filters=['CFHT_MEGACAM.U', 'SUBARU_HSC.G' ,'SUBARU_HSC.R' ,'CFHT_MEGACAM.R' ,'SUBARU_HSC.I', 'SUBARU_HSC.Z' ,'SUBARU_HSC.Y'],
            **kwargs):
        super(TNGDataSet,self).__init__(**kwargs)
        self.internal_dict = SubsplitDictionaries()
        self.isPopulated = False
        self.list_of_fits = []
        # Notify user if he is using wierd percentages
        if (train_percent + val_percent + test_percent) != 1:
            Logger("TNGDataSet WARNING : Only {0}% of the data will be assigned".format(str((train_percent + val_percent + test_percent))))
        # register percenteges
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent
        # Number of entries yielded
        self.nb_of_train = 0
        self.nb_of_val = 0
        self.nb_of_test = 0
        # Define the simage helpers
        self.Scaler_fcn = Scaler_fcn
        self.Image_Size = Image_Size
        # Define selected filters
        self.band_filters = band_filters
        # helper
        self.hit_count = {}
        self.generation_verbosity = generation_verbosity
        self.input_dir = input_dir
    
    def PopulateFileList(self,fit_path):
        if self.isPopulated:
            return
        else:
            assert len(self.list_of_fits) == 0

        for root, dirs, files in os.walk(fit_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                if os.path.splitext(file_path)[1] != ".fits":
                    continue
                self.list_of_fits.append(file_path)
        # Make sure that some data exists
        assert len(self.list_of_fits) > 0
        self.isPopulated = True
        # nb_of_entries  is number of files times number of selected filters
        self.nb_of_entries = len(self.list_of_fits) * len(self.band_filters)
        # Called only once .. so not verbose
        Logger("File list is populated .. there is {0} files with {1} entries".format(len(self.list_of_fits),self.nb_of_entries))
    
    def EnoughSamples(self,split_type):
        if split_type == tfds.Split.TRAIN:
            return self.nb_of_train >= (self.nb_of_entries * self.train_percent)
        elif split_type == tfds.Split.VALIDATION:
            return self.nb_of_val >= (self.nb_of_entries * self.val_percent)
        elif split_type == tfds.Split.TEST:
            return self.nb_of_test >= (self.nb_of_entries * self.test_percent)
        
    def IncrementSamples(self,split_type):
        if split_type == tfds.Split.TRAIN:
            self.nb_of_train += 1
            if self.nb_of_train % self.generation_verbosity == 0:
                Logger("Number of Training samples is {0} ".format(self.nb_of_train),1)
                Logger("Remaining to add {0}".format(int(self.nb_of_entries * self.train_percent)),1)
        elif split_type == tfds.Split.VALIDATION:
            self.nb_of_val += 1
            if self.nb_of_val % self.generation_verbosity == 0:
                Logger("Number of Validation samples is {0}".format(self.nb_of_val),1)
                Logger("Remaining to add {0}".format(int(self.nb_of_entries * self.val_percent)),1)
        elif split_type == tfds.Split.TEST:
            self.nb_of_test += 1
            if self.nb_of_test % self.generation_verbosity == 0:
                Logger("Number of Test samples is {0}".format(self.nb_of_test),1)
                Logger("Remaining to add {0}".format(int(self.nb_of_entries * self.test_percent)),1)


    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            homepage=_URL,
            citation=_CITATION,
            # Two features: image with 3 channels (stellar light, velocity map, velocity dispersion map)
            #  and redshift value of last major merger
            features=tfds.features.FeaturesDict({
                "img" : tfds.features.Tensor(shape=(150,150) , dtype=tf.float32),
                "EXTNAME" : tf.string,
                "ORIGIN" : tf.string,
                "SIMTAG" : tf.string,
                "SNAPNUM" : tf.int32,
                "SUBHALO" : tf.int32,
                "CAMERA" : tf.string,
                "REDSHIFT" : tf.float32,
                "FILTER" : tf.string,
                "FOVSIZE" : tf.float32,
                "BUNIT" : tf.string,
                "NAXIS1" : tf.int32,
                "NAXIS2" : tf.int32
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns generators according to split"""
        return {tfds.Split.TRAIN: self._generate_examples(GetParent(dl_manager.download_dir) ,tfds.Split.TRAIN) ,
                tfds.Split.VALIDATION: self._generate_examples(GetParent(dl_manager.download_dir) ,tfds.Split.VALIDATION),
                tfds.Split.TEST: self._generate_examples(GetParent(dl_manager.download_dir) ,tfds.Split.TEST)}

    def _generate_examples(self, fit_path,split_type):
        """Yields examples."""

        # Only populated the first time
        self.PopulateFileList(self.input_dir)

        while len(self.list_of_fits) > 0 and not self.EnoughSamples(split_type):
            # Select randomly a file
            fit_file = choice(self.list_of_fits)

            # very verbose
            Logger("File : {0} is being treated".format(fit_file),2)

            fitm = fits.open(fit_file)
            for fit_elem in fitm:
                example = {}

                example["EXTNAME"] = fit_elem.header["EXTNAME"]
                # Filter the extension :)
                if example["EXTNAME"] not in self.band_filters:
                    continue

                example["FILTER"] = fit_elem.header["FILTER"]
                example["ORIGIN"] = fit_elem.header["ORIGIN"]
                example["SIMTAG"] = fit_elem.header["SIMTAG"]
                example["SNAPNUM"] = fit_elem.header["SNAPNUM"]
                example["SUBHALO"] = fit_elem.header["SUBHALO"]
                example["CAMERA"] = fit_elem.header["CAMERA"]

                key = SubsplitDictionaries.CreateKey(example["EXTNAME"],example["ORIGIN"],example["SIMTAG"],
                                                    example["SNAPNUM"],example["SUBHALO"])
                # Extremely verbose
                if self.internal_dict.FindOrCreate(key,example["CAMERA"],split_type):
                    continue
                
                example["REDSHIFT"] = fit_elem.header["REDSHIFT"]
                example["FOVSIZE"] = fit_elem.header["FOVSIZE"]
                example["BUNIT"] = fit_elem.header["BUNIT"]
                example["NAXIS1"] = fit_elem.header["NAXIS1"]
                example["NAXIS2"] = fit_elem.header["NAXIS2"]
                example["img"] = self.Scaler_fcn(fit_elem.data)

                # Register a hit for this file
                if self.hit_count.get(fit_file,0) != 0:
                    self.hit_count[fit_file] += 1
                else:
                    self.hit_count[fit_file] = 1
                self.IncrementSamples(split_type)
                Logger("Hit count for {0} is {1}".format(fit_file,self.hit_count[fit_file]),2)

                key_with_camera = "{0}_{1}".format(key,example["CAMERA"])
                yield key_with_camera , example

            # Done with this file
            # If we took all data with selected filters this file is done remove it
            if self.hit_count.get(fit_file,0) == len(self.band_filters):
                self.list_of_fits.remove(fit_file)



def loadTNGDataset(
            input_dir, 
            output_dir,
            train_percent=0.8, 
            val_percent=0.1, 
            test_percent=0.1,
            Scaler_fcn=None ,
            generation_verbosity=1000,
            Image_Size=(150,150),
            band_filters=['CFHT_MEGACAM.U', 'SUBARU_HSC.G' ,'SUBARU_HSC.R' ,'CFHT_MEGACAM.R' ,'SUBARU_HSC.I', 'SUBARU_HSC.Z' ,'SUBARU_HSC.Y'],
            subsets = [tfds.Split.TRAIN,tfds.Split.VALIDATION, tfds.Split.TEST]):
    
    # A Scaling must exist
    assert(ScaleImage is not None)

    arg_dict = {
    'input_dir':input_dir,
    'train_percent':train_percent,
    'val_percent':val_percent, 
    'test_percent':test_percent , 
    'Image_Size':Image_Size , 
    'generation_verbosity' : generation_verbosity , 
    'Scaler_fcn':Scaler_fcn,
    'band_filters' : band_filters
    }
    #subsets = [tfds.Split.TRAIN,tfds.Split.VALIDATION]
    ds = tfds.load('TNGDataSet', split=subsets, data_dir=output_dir, builder_kwargs=arg_dict)

    return ds