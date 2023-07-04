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

_DESCRIPTION = """
#Data representing the TNG50, TNG100, and TNG300 Simulations
"""

_CITATION = ""
_URL = "https://github.com/astroinfo-hacks/2023-imgen-diffusion"

histo_grame = {}

def ScaleImage(img):
    return img

class SubsplitDictionaries:
    def __init__(self):
        self.train_dict = []
        self.valid_dict = []
        self.test_dict = []
        self.full_list = []

    @staticmethod
    def CreateKey(EXTNAME, ORIGIN, SIMTAG, SNAPNUM, SUBHALO):
        return "{0}_{1}_{2}_{3}_{4}".format(EXTNAME, ORIGIN, SIMTAG, SNAPNUM, SUBHALO)
    
    # Sanity check
    def AlreadyExisting(self,key_with_camera,Create=False):
        res =  key_with_camera not in self.full_list
        if Create:
          self.full_list.append(key_with_camera)
        return res

    def FindOrCreate(self,key, CAMERA,subsplit):
        key_with_camera = "{0}_{1}".format(key,CAMERA)
        assert (self.AlreadyExisting(key_with_camera,True))
        if subsplit == tfds.Split.TRAIN:
            if key not in self.valid_dict and key not in self.test_dict:
                assert key not in self.train_dict
                self.train_dict.append(key)
                return False
        elif subsplit == tfds.Split.VALIDATION:
            if key not in self.train_dict and key not in self.test_dict:
                assert key not in self.valid_dict
                self.valid_dict.append(key)
                return False
        elif subsplit == tfds.Split.TEST:
            if key not in self.train_dict and key not in self.valid_dict:
                assert key not in self.test_dict
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
  
  def __init__(self,**kwargs):
      super(TNGDataSet,self).__init__(**kwargs)
      self.internal_dict = SubsplitDictionaries()
      self.isPopulated = False
      self.list_of_fits = []
    
  def PopulateFileList(self,fit_path):
      if self.isPopulated:
          return
      else:
        assert len(self.list_of_fits) == 0

        for root, dirs, files in os.walk(fit_path, topdown=False):
            for name in files:
                self.list_of_fits.append(os.path.join(root, name))
        # Make sure that some data exists
        assert len(self.list_of_fits) > 0
        self.isPopulated = True

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
            "img" : tfds.features.Tensor(shape=(500,500) , dtype=tf.float32),
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
    self.PopulateFileList(fit_path)
    # Select randomly a file
    while len(self.list_of_fits) > 0:
        fit_file = choice(self.list_of_fits)
        fitm = fits.open(fit_file)
        for fit_elem in fitm:
            example = {}

            example["EXTNAME"] = fit_elem.header["EXTNAME"]
            example["ORIGIN"] = fit_elem.header["ORIGIN"]
            example["SIMTAG"] = fit_elem.header["SIMTAG"]
            example["SNAPNUM"] = fit_elem.header["SNAPNUM"]
            example["SUBHALO"] = fit_elem.header["SUBHALO"]
            example["CAMERA"] = fit_elem.header["CAMERA"]

            key = SubsplitDictionaries.CreateKey(example["EXTNAME"],example["ORIGIN"],example["SIMTAG"],
                                                example["SNAPNUM"],example["SUBHALO"])
            if self.internal_dict.FindOrCreate(key,example["CAMERA"],split_type):
                continue
            
            example["REDSHIFT"] = fit_elem.header["REDSHIFT"]
            example["FILTER"] = fit_elem.header["FILTER"]
            example["FOVSIZE"] = fit_elem.header["FOVSIZE"]
            example["BUNIT"] = fit_elem.header["BUNIT"]
            example["NAXIS1"] = fit_elem.header["NAXIS1"]
            example["NAXIS2"] = fit_elem.header["NAXIS2"]
            example["img"] = ScaleImage(fit_elem.data)

            self.list_of_fits.remove(fit_file)

            yield "I" , example