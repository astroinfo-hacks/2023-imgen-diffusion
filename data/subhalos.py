import illustris_python as il
import numpy as np
from astropy.io import fits

basePath = '/virgo/simulations/IllustrisTNG/TNG50-1/output/'
fields = ['SubhaloFlag','SubhaloMass','SubhaloHalfmassRad']

cols=[]
for i in range(72,91,1):
    subhalos = il.groupcat.loadSubhalos(basePath,i,fields=fields)
    col1=fits.Column(name='SubhaloMass', format='D', array=subhalos['SubhaloMass'] * 1e10 / 0.704)
    cols.append(col1)
    col2=fits.Column(name='SHID', format='D', array=subhalos['count'])
    cols.append(col2)
    col3=fits.Column(name='SubhaloHalfMassRad', format='D', array=subhalos['SubhaloHalfmassRad'])
    cols.append(col3)
    col4=fits.Column(name='SubhaloFlag', format='D', array=subhalos['SubhaloFlag'])
    cols.append(col4)
    
    cols_to_write=fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto("/u/mhuertas/TNG50/cats/TNG50_0"+str(i)+"_physprop.fit",overwrite='True')
