from astropy.io import fits
import numpy as np
import os
import csv

data_path = "../data/091"
output_path = "./parameter_stat.csv"

files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

header_stats = {}

for file in files:
    if not file.endswith(".fits"):
        continue

    header = fits.getheader(os.path.join(data_path, file))

    for key, value in header.items():
        if key in header_stats:
            if value not in header_stats[key]:
                header_stats[key][value] = 1
            else:
                header_stats[key][value] += 1
        else:
            header_stats[key] = {value: 1}

# Write header stats to a CSV file
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(['Keyword'] + ['Frequency'])

    for key, value_counts in header_stats.items():
        for value, count in value_counts.items():
            writer.writerow([key] + ["{:s} ({:s})".format(str(value), str(count))])
