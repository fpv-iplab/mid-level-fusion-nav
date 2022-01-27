import numpy as np
import math
import os
import csv
from pyquaternion import Quaternion as pyquat
from PIL import Image
import faiss
from habitat.tasks.utils import quaternion_rotate_vector

class useRealImages():
    def __init__(
        self,
        data_path,
        angle_sim_threshold,
        data_partition = False,
        preload_images = False,
        images_type = "rgb",
        is_train = False       
        ):
        self._angle_sim_threshold = angle_sim_threshold
        self._data_path = data_path
        self._images_path = os.path.join(data_path, "imgs")
        self._preload_images = preload_images
        self._images_type = images_type
        self.is_train = is_train

        images_data = self._load_images_file(data_path)
        self._split_data(images_data, data_partition)
        
        self._init_search_indexes()
    
    def _load_images_file(self, file_path):
        rows=[]
        with open(os.path.join(file_path,"COLMAP_model.csv"), "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                rows.append(row)
        return rows

    def _split_data(self, images_data, data_partition):
        img_names = list()
        coords = list()
        quaternions = list()
        for row in images_data:
            img_names.append(row[0])
            coords.append( (np.float32(row[1]), np.float32(row[2])) )
            quaternions.append( pyquat( ( np.float32(row[3]),
                                          np.float32(row[4]),
                                          np.float32(row[5]),
                                          np.float32(row[6])) ) )
        if data_partition == False:
            self._img_names = img_names
            self._coords = np.asarray(coords)
            self._quaternions = np.asarray(quaternions)
            #heading vector ((cosA, sinA))
            heading_vec = []
            for q in self._quaternions:
                vec = self.quat2heading_vec(q)
                heading_vec.append( vec )
            self._heading_vec = np.asarray(heading_vec, dtype=np.float32)
        else:
            coords = np.asarray(coords)
            quaternions = np.asarray(quaternions)
            #heading vector ((cosA, sinA))
            heading_vec = []
            for q in quaternions:
                vec = self.quat2heading_vec(q)
                heading_vec.append( vec )
            heading_vec = np.asarray(heading_vec, dtype=np.float32)
            
            img_names = np.asarray(img_names)
            n_partitions = 8 # split the dataset in 8 chunks and choose not adjacent chunks for train/test sets for images diversity
            if self.is_train:
                partitions = [0, 2, 4, 6]
            else:
                partitions = [1, 3, 5, 7]
            n_images = len(img_names)
            splits_indexes=[ int(np.ceil( (n_images/n_partitions)*x )) for x in range(n_partitions+1) ] # 0-1000-2000,3000, ...
            partitions_indexes=[ np.arange(splits_indexes[i], splits_indexes[i+1]) for i in range(n_partitions) if i in partitions] #[[0, 1, ..., 999], [2000, ..., 2999], ...]
            partitions_indexes = [i for arr in partitions_indexes for i in arr]
            selected_images = img_names[partitions_indexes]
            selected_h_vec = heading_vec[partitions_indexes]
            selected_coords = coords[partitions_indexes]
            selected_quat = quaternions[partitions_indexes]

            self._img_names = np.asarray(selected_images)
            self._heading_vec = np.asarray(selected_h_vec, dtype=np.float32)
            self._coords = np.asarray(selected_coords, dtype=np.float32)
            self._quaternions = np.asarray(selected_quat)

        # if preload_images is True, images are read and loaded in ram all at once to avoid i/o operations from disk during train
        if self._preload_images:
            images = []
            for img in self._img_names:
                with open(os.path.join(self._images_path, img), 'rb') as f:
                    image = np.asarray(Image.open(f), dtype=np.uint8 )
                    images.append( image )
            self._loaded_images = images
    
    
    def _init_search_indexes(self):
        """
        NN images, filtered by angles and then by coordinates
        """
        self._rot_index=faiss.IndexFlatIP(2)
        print("Rotations index loaded:",self._rot_index.is_trained)
        self._rot_index.add(self._heading_vec)

    def quat2heading_vec(self, q, invert_cos=False, invert_sin=False):
        r""" because images of the real map don't always have the same rotation angles/quaternions of the virtual images,
        the problem can eventually be fixed at this level, eventually changing the sin or cos sign in order to 
        have a correct angle match
        """
        q = pyquat( w=q[0], x=q[1], y=q[2], z=q[3] )
        a = 2 * (q.w * q.z + q.x * q.z)
        b = 1 - 2 * (q.z * q.z + q.y * q.y)
        angle = np.arctan2(a,b)
        if angle < 0:
            angle = 2 * np.pi + angle # + because angle is negative
        cos = np.cos(angle)
        sin = np.sin(angle)
        if invert_cos:
            cos *= -1
        if invert_sin:
            sin *= -1
        return (cos, sin)

    def get_nearest_image(self, query_quat, query_coords):
        # filter by heading vector
        query_angle = self.quat2heading_vec(query_quat, invert_sin=True)
        k = len(self._heading_vec) # k-nn
        rot_sim, rot_idx=self._rot_index.search(
            np.array( query_angle, dtype=np.float32).reshape(1,-1), k )
        near_rot_indexes = rot_idx[rot_sim > self._angle_sim_threshold]
        near_coords=self._coords[ near_rot_indexes ]
        loc_index=faiss.IndexFlatL2(2)
        loc_index.add(near_coords)
        k=1 # k-nn
        _,near_loc_indexes=loc_index.search(
            np.array( query_coords, dtype=np.float32).reshape(1,-1), k )
        abs_image_index = near_rot_indexes[ np.squeeze(near_loc_indexes) ]
        
        # is the image preloaded or have to read it from the disk?
        if self._preload_images:
            image = Image.fromarray( self._loaded_images[abs_image_index] ) 
        else:
            image = Image.open( os.path.join(self._images_path,self._img_names[abs_image_index]) )
        return image, self._coords[abs_image_index], self._quaternions[abs_image_index]

    def resize_image(self, image, size=(256,256)):
        w, h = image.size
        ratio = w/h
        res_image = image.resize( (int(size[1]*ratio), size[1]), Image.BILINEAR ) # width, height
        new_w, _ = res_image.size
        
        if new_w > size[0]:
            exceed = int( (new_w - size[0])/2 )
            res_image = res_image.crop( (exceed, 0, new_w-exceed, size[1]) ) # starting x,y , final x,y
            # can happend that after removing left and right portions of the image, it's still slightly wider
            # (by few pixels) than the height. 
            if res_image.size[0] > size[1]:
                res_image = res_image.crop( (0, 0, size[0], size[1]) )
        return np.array(res_image)