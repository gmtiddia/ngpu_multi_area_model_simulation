import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.gridspec as gridspec

fr_E=img.imread("firing_rate/L4E2NEST.png")
fr_I=img.imread("firing_rate/L4I2NEST.png")
cv_E=img.imread("cv_isi/L4E2NEST.png")
cv_I=img.imread("cv_isi/L4I2NEST.png")
co_E=img.imread("correlation/L4E2NEST.png")
co_I=img.imread("correlation/L4I2NEST.png")


fig, axs = plt.subplots(3, 2)
plt.subplots_adjust(wspace=-0.7, hspace=0.0, left=0.05, right=0.95) 

axs[0,0].imshow(fr_E)
axs[0,0].axis('off')
axs[0,1].imshow(fr_I)
axs[0,1].axis('off')
axs[1,0].imshow(cv_E)
axs[1,0].axis('off')
axs[1,1].imshow(cv_I)
axs[1,1].axis('off')
axs[2,0].imshow(co_E)
axs[2,0].axis('off')
axs[2,1].imshow(co_I)
axs[2,1].axis('off')
fig.set_size_inches(24, 15)
plt.savefig("dist_sample_ms.png", bbox_inches = 'tight', pad_inches = 0)
plt.show()




