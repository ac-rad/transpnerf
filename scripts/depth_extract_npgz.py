from matplotlib import pyplot as plt
import numpy as np
import gzip

# extract from numpy gz file and invert
with gzip.open(f'r_13.npy.gz', 'rb') as f:
    depth_array_est = 1/np.load(f)

# plot
plt.imshow(depth_array_est, cmap='gray')
plt.axis('off')
plt.savefig('depth.png', bbox_inches='tight', pad_inches=0)
plt.show()


