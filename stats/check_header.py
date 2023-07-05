from astropy.io import fits
import numpy as np
import os
import json
import logging
import sys
from glob import glob

TEMPLATE = '%(asctime)s %(levelname)5s [%(name)s %(lineno)d] %(message)s'
logging.basicConfig(format=TEMPLATE, stream=sys.stdout, level=logging.INFO)
s_logger = logging.getLogger(__name__)


# data_path = "/scratch/astroinfo2023/diffusion/"
data_path = "./"
output_path = "header.txt"

files = glob(os.path.join(data_path, "0*", "*.fits")) + \
        glob(os.path.join(data_path), "data", "0*", "*.fits")

np.random.shuffle(files)

files = files[:200]

header_stats = {}


def get_kws(hdu_list):
    for hdu in hdu_list:
        print(hdu.header)


def get_images_size(hdu_infos):
    assert isinstance(hdu_infos, list)

    my_size = hdu_infos[0][5]
    for ext in range(1,len(hdu_infos)):
        if my_size != hdu_infos[ext][5]:
            print(f"Error: size of the extension {ext} is {hdu_infos[ext][5]}"
                  f" and does not match {my_size}")
            return False
    if my_size[0] != my_size[1]:
        print(f"Warning: image is not square: {my_size}")
        return np.max(my_size)
    return my_size[0]


def get_kw_values(hdu_list, kw_name):
    assert isinstance(hdu_list, fits.hdu.HDUList)
    values = []
    for hdu in hdu_list:
        if kw_name not in hdu.header:
            print(f"Error: RedShift not present in image!")
            return False
        values.append(hdu.header[kw_name])
    return values


# def get_wavelength(hdu_list):
# #def get_fov_size(hdu_list):


s_logger.info(f"Processing {__file__}")
general_stats = {"size": {}}
for file in files:
    s_logger.info(f"Processing file {file}")
    if not file.endswith(".fits"):
        continue

    with fits.open(os.path.join(data_path, file)) as hdr_1:

        image_size = get_images_size(hdr_1.info(False))
        if not image_size:
            s_logger.warning(f"Image {file} has not consistent images")
            continue
        if image_size not in general_stats["size"].keys():
            general_stats["size"][image_size] = 1
        else:
            general_stats["size"][image_size] += 1

        s_logger.info(f"Image in file {file} has size {image_size}")
        # redshifts = get_kw_values(hdr_1, "REDSHIFT")
        # wavelengths = get_kw_values(hdr_1, "WLPIVOT")
        # fov_sizes = get_kw_values(hdr_1, "FOVSIZE")

# print(redshifts)
# print(fov_sizes)
print(general_stats)

for key, val in general_stats["size"].items():
    print(f"There are {general_stats['size'][key]} images with size ({key}, {key})")

