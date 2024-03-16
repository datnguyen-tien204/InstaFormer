import os
import pdb
import random

from PIL import Image
import numpy as np
import torch


from .base_dataset import BaseDataset
from .image_folder import make_dataset 

class INIT_Dataset(BaseDataset):
    def __init__(self, cfg, train_mode=True):
        BaseDataset.__init__(self, cfg, train_mode)

        self.A_paths = sorted(make_dataset(self.dir_A, cfg.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, cfg.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
    
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[random.randint(0, self.B_size - 1)]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        
        do_flip = self.train_mode and random.random() > 0.5
        
        if self.use_box:
            direct = 'gt_box'           
            path_split_A = A_path.split('/')
            path_split_B = B_path.split('/')
            
            box_path_A = os.path.join('/',*path_split_A[:-1],direct,(path_split_A[-1][:-4]+'.txt')) #gt_box for INIT box, box for Faster R-CNN box
            box_path_B = os.path.join('/',*path_split_B[:-1],direct,(path_split_B[-1][:-4]+'.txt'))
            
            A_Box = torch.zeros(self.num_box, 5)
            A_Box[:, 0] = -1
            B_Box = torch.zeros(self.num_box, 5)
            B_Box[:, 0] = -1

            try:                                                                                                    
                with open(box_path_A, 'r') as f:                                                                                
                    for i, line in enumerate(f.readlines()):
                        if i >= self.num_box:
                            break
                    
                        param = line.split(' ')

                        if float(param[3]) - float(param[1]) < 0.03:
                            pass
                        else:
                            A_Box[i,0] = int(param[0]) 
                            A_Box[i,1] = float(param[1]) 
                            A_Box[i,2] = float(param[2]) 
                            A_Box[i,3] = float(param[3])
                            A_Box[i,4] = float(param[4])
            except:
                pass
             
            try:
                with open(box_path_B, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        if i >= self.num_box:
                            break
                    
                        param = line.split(' ')

                        if float(param[1]) - float(param[1]) < 0.03:
                            pass
                        else:
                            B_Box[i,0] = int(param[0]) 
                            B_Box[i,1] = float(param[1]) 
                            B_Box[i,2] = float(param[2]) 
                            B_Box[i,3] = float(param[3])
                            B_Box[i,4] = float(param[4])
            except:
                pass 

            if self.train_mode:
                A, A_Box = self.horiz_flip(A, A_Box)
                B, B_Box = self.horiz_flip(B, B_Box)

            return {'A': A, 'B': B, 'A_box': A_Box, 'B_box': B_Box}       

        else:
            if self.train_mode:
                A = self.horiz_flip(A)
                B = self.horiz_flip(B)
            
            return {'A': A, 'B': B}
            # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)