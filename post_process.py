#----------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------

import numpy as np
from numpy import ma
import h5py as h

from scipy import ndimage as ndi
from skimage.morphology import watershed
from scipy.ndimage import label as label_scipy
from scipy.misc import imsave

from helper import *

#----------------------------------------------------------------------------
# ENERGY MAP
#----------------------------------------------------------------------------

energy = np.array(h.File('./inputs/test_energy.h5', 'r')['main'])[0]
energy.shape

#----------------------------------------------------------------------------
# CONNECTED COMPONENTS
#----------------------------------------------------------------------------

# Apply connected components to get initial segmentation map. 

seg = get_seg(energy, None, 16) 

# Remove small segmentation labels (with voxel count < 1000)

nlabels, count = np.unique(seg, return_counts=True)

indices = np.argsort(count)
nlabels = nlabels[indices]
count = count[indices]


least_index = np.where(count >= 1000)[0][0] 

count = count[least_index:]
nlabels = nlabels[least_index:]

rl = np.arange(seg.max() + 1).astype(seg.dtype)

for i in range(seg.max() + 1):
    if i not in nlabels:
        rl[i] = 0

seg = rl[seg]

# Save initial segmentation

f = h.File(f'./inputs/test_seg_cc.h5', 'w')
f.create_dataset('main', data=seg)
f.close()

#----------------------------------------------------------------------------
# WATERSHED
#----------------------------------------------------------------------------

energy = np.array(h.File('./inputs/test_energy.h5', 'r')['main'])[0].astype(np.float32)

threshold = 150

# Extract markers from energy map

energy_thres = energy - threshold

markers_unlabelled = (energy_thres > 0).astype(int)

# Label markers for watershed

markers, ncomponents = label_scipy(markers_unlabelled)

# Remove small markers, to prevent oversegmentation

labels_d, count_d = np.unique(markers, return_counts=True) 

rl = np.arange(markers.max() + 1).astype(markers.dtype)
pixel_threshold = 100


for i in range(len(labels_d)):
    if count_d[i] < pixel_threshold:
        rl[labels_d[i]] = 0

markers = rl[markers]

# Mask for watershed from CC output

mask = (seg > 0).astype(int) 

# Watershed with markers and mask 

labels = watershed(-energy, mask=mask, markers=markers) 

# Save Watershed segmentation map

f = h.File(f'./inputs/test_seg_{threshold}.h5', 'w')
f.create_dataset('main', data=labels)
f.close()

# Get watershed labels for Neuroglancer

np.unique(labels)


#----------------------------------------------------------------------------
# VISUALIZATION
#----------------------------------------------------------------------------

from neuroG import NeuroG

ng = NeuroG(port=8891)


ng.addLayer('inputs/test_volm.h5', 'h5py', name="Input Image")
ng.addLayer('inputs/test_energy.h5', 'h5py', name="Output")
ng.addLayer(f'inputs/test_seg_{threshold}.h5', 'h5py', name="Segmentation", isLabel=True)
ng.addLayer(f'inputs/test_seg_cc.h5', 'h5py', name="Seg CC", isLabel=True)

ng.viewer


#----------------------------------------------------------------------------
# SAVE OUTPUT TO PNG
#----------------------------------------------------------------------------

# def seg2Vast(seg):                                                                                   
#     return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)

# labels_wshed = np.array(h.File('./inputs/test_seg_150.h5', 'r')['main'])

# for i in range(labels_wshed.shape[0]):
#     png = seg2Vast(labels_wshed[i])
#     # save png to file
#     imsave(f'./slice20/{i}.png', png)