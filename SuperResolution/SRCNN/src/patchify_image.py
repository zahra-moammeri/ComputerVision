@ -0,0 +1,103 @@



from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import patchify
import numpy as np
import matplotlib.gridspec as gridspec
import glob as glob
import os
import cv2

SHOW_PATCHES = False
STRIDE = 14
SIZE = 32   # Patch size



def show_patches(patches):
  plt.figure(figsize=(patches.shape[0], patches.shape[1]))
  gs = gridspec.GridSpec(patches.shape[0], patches.shape[1])
  gs.update(wspace= 0.01, hspace=0.02)
  counter = 0
  for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
      ax = plt.subplot(gs[counter])
      plt.imshow(patches[i, j, 0, :, :, :])
      plt.axis('off')
      counter += 1
  plt.show()

"""
The show_patches function accepts the image patches and displays
them using Matplotlib if SHOW_PATCHES is True.
"""



def create_patches(
    input_paths, out_hr_path, out_lr_path,
):
    os.makedirs(out_hr_path, exist_ok=True)
    os.makedirs(out_lr_path, exist_ok=True)
    all_paths = []
    for input_path in input_paths:
        all_paths.extend(glob.glob(f"{input_path}/*"))
    print(f"Creating patches for {len(all_paths)} images")
    for image_path in tqdm(all_paths, total=len(all_paths)):
        image = Image.open(image_path)
        image_name = image_path.split(os.path.sep)[-1].split('.')[0]
        w, h = image.size
        # Create patches of size (32, 32, 3)
        patches = patchify.patchify(np.array(image), (32, 32, 3), STRIDE)
        if SHOW_PATCHES:
            show_patches(patches)


            """
            It will return a square matrix depending on the stride and patch size
            which will contain 32×32 dimensional images.
            We start iterating over the rows and columns of that matrix.
            Then we extract each of the corresponding patches and save it to disk using OpenCV
            after converting to BGR format.
            """

        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0, :, :, :]
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{out_hr_path}/{image_name}_{counter}.png",
                    patch
                )
                # Convert to bicubic and save.
                h, w, _ = patch.shape
                low_res_img = cv2.resize(patch, (int(w*0.5), int(h*0.5)),
                                        interpolation=cv2.INTER_CUBIC)
                # Now upscale using BICUBIC.
                high_res_upscale = cv2.resize(low_res_img, (w, h),
                                            interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(
                    f"{out_lr_path}/{image_name}_{counter}.png",
                    high_res_upscale
                )


        """
        We create the corresponding low resolution patch on line 61. Because we are using 2x bicubic upscaling
        to again obtain the original size blurry path, so, we just multiply the width and height of the patch by 0.5
        to downscale them. Finally, We save these low resolution (same 32×32 dimensional, but blurry) patches to disk as well.
        """
        

if __name__ == '__main__':
    create_patches(
        ['../input/T91'],
        '../input/t91_hr_patches',
        '../input/t91_lr_patches'
        )