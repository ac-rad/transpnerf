import torch
from torch.functional import norm
import torch.nn.functional as F
import cv2
import numpy as np
from matplotlib import pyplot as plt


# read Depth-Anything depth anything
image = cv2.imread("./frame_00032.png", cv2.IMREAD_ANYDEPTH) 


# apply surface normal filter from https://github.com/Ruthrash/surface_normal_filter with a few modifications
image = image.astype(np.float32)
depths = torch.from_numpy(image[:, :, np.newaxis])

depths_reshape = depths.permute(2, 0, 1).unsqueeze(1) 
nb_channels = 1

delzdelxkernel = torch.tensor([[0.00000, 0.00000, 0.00000],
                                [-1.00000, 0.00000, 1.00000],
                                [0.00000, 0.00000, 0.00000]]) 
delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
delzdelx = F.conv2d(depths_reshape, delzdelxkernel, padding=1)

delzdelykernel = torch.tensor([[0.00000, -1.00000, 0.00000],
                                [0.00000, 0.00000, 0.00000],
                                [0.0000, 1.00000, 0.00000]]) 
delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)

delzdely = F.conv2d(depths_reshape, delzdelykernel, padding=1)

delzdelz = torch.ones(delzdely.shape) 

surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
surface_norm = torch.div(surface_norm,  norm(surface_norm, dim=2)[:,:,None,:,:])

# normalize between [0, 255] and distrubute to correct axis for visualization.
surface_norm_viz = torch.mul(torch.add(surface_norm, 1.00000),127 )

normal_output = surface_norm_viz.numpy()[0,0,:,:,:]
normal_img = np.zeros((756,1008,3),dtype=float)
normal_img[:,:,0] = normal_output[0,:,:]  
normal_img[:,:,1] = normal_output[1,:,:]
normal_img[:,:,2] = normal_output[2,:,:]

cv2.imwrite("normal.jpg", normal_img)  