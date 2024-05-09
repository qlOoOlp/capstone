from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os


a=np.load('/home/hong/capstone/vlmaps/data_demo/custom_0508/depth/000000.npy')
img = Image.open("/home/hong/capstone/vlmaps/data_demo/custom_0508/depth_img/000000.png")
ans = np.array(img)

print(a)
print(ans)

plt.imshow(a, cmap='gray')#, vmin=0, vmax=255)
plt.colorbar()  # 옆에 컬러바를 추가하여 값의 범위를 표시
plt.show()
