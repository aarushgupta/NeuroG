import neuroglancer
import numpy as np
import sys
import tifffile
import h5py

import warnings
warnings.filterwarnings("ignore")

class NeuroG:
    """

Class wrapper for Neuroglancer (https://github.com/google/neuroglancer)

Supports loading h5, numpy and compressed numpy files.

Args:
	ip (string): IP Address to bind the instance to
	port (int): Port to bind the instance to
	res (tuple):  Resolution of the data. This argument sets a global resolution value.
				  Custom resolution for every layer can be used by using the res argument 
				  in the addLayer method.
	label_dtype (dtype): Datatype of the segmentation labels

Examples:

	>>> ng = NeuroG(ip='locahost', port=8889)
	>>> ng = NeuroG(ip='172.25.14.127')
	>>> ng.viewer

Misc:
	viewer: instance attribute containing the link to the visualization

	"""
    def __init__(self, ip='localhost', port=98100, res=[6,6,30], label_dtype=np.uint16):
        super(NeuroG, self).__init__()

        self.port = port
        self.ip = ip
        self.res = res

        neuroglancer.set_server_bind_address(bind_address=self.ip, bind_port=self.port)
        self.viewer = neuroglancer.Viewer()

        self.label_dtype = label_dtype

    def addLayer(self, path, fileType, res=None, isLabel=False, name="Image", verbose=False):

    	'''

	Add new visualization layer to the instance

    Args:
        path (string): Path to the file (for loading the file), or variable name to preloaded data
        fileType (string, None): Extension of the file to be loaded (None for preloaded file)
        res (tuple): Override resolution set for the whole object
        name (string): Name to be displayed in the layer tab of neuroglancer.
        verbose (bool): Print additional information about the layer

    Examples:

	>>>  ng.addLayer('./inputs/test/seg.h5', 'h5py', None, True, "Segmentation Map")
	>>>  ng.addLayer('./inputs/train/img.npz', 'npArray', None, False, "Input Image")
	>>>  ng.addLayer(a, None, None, False, name='Image')


        '''

        assert fileType in ['h5py', 'npArray', None]

        if res == None:
            res = self.res
        if verbose:
            print(f"Load {name}")

        if fileType == 'h5py':
            temp = np.array(h5py.File(path,'r')['main'])
        elif fileType == 'npArray':
            temp = np.load(path)['arr_0']
        elif fileType == None:
            temp = path #path is the array then

        if isLabel:
            with self.viewer.txn() as s:
                s.layers.append(
                    name=name,
                    layer=neuroglancer.LocalVolume(
                        data=temp.astype(self.label_dtype),
                        voxel_size=res))
        else:
            with self.viewer.txn() as s:
                s.layers.append(
                    name=name,
                    layer=neuroglancer.LocalVolume(
                        data=temp,
                        voxel_size=res))

        del temp



