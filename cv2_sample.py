import cv2
import numpy as np
from PIL import Image

# https://habr.com/ru/articles/469355/

def smooth(I):
    J = I.copy()
    J[1:-1] = J[1:-1] // 2 + J[:-2] // 4 + J[2:] // 4
    J[:, 1:-1] = J[:, 1:-1] // 2 + J[:, :-2] // 4 + J[:, 2:] // 4
    return J


num = '4'

foto = Image.open(f'img/{num}.png')
new_size = (28, 28)
foto.thumbnail(new_size)
foto.save(f'img/{num}_28.png')

# read image as grey scale
grey_img = cv2.imread(f"img/{num}_28.png", cv2.IMREAD_GRAYSCALE)

# save image
status = cv2.imwrite(f"img/{num}_28_inp.png", grey_img)
print("Image written to file-system : ", status)
np.savetxt(f"img/{num}_28_inp.csv", grey_img, delimiter=",", fmt='%d')

denoise = smooth(grey_img)

# save image
status = cv2.imwrite(f"img/{num}_28_out.png", denoise)
print("Image written to file-system : ", status)
np.savetxt(f"img/{num}_28_out.csv", denoise, delimiter=",", fmt='%d')

# np.savetxt('test1.txt', grey_img, fmt='%d')
