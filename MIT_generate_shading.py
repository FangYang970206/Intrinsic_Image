import os
import numpy as np
import cv2


data_path = 'D:\\fangyang\\MIT_intrinsic_image'
names = ['apple', 'box', 'cup1', 'cup2', 'deer', 'dinosaur', 'frog1', 'frog2', 
         'panther', 'paper1', 'pear', 'phone', 'potato', 'squirrel', 'sun', 
         'teabag1', 'teabag2', 'turtle']
name_suffixs = ['_light01', '_light02', '_light03', '_light04','_light05',
                '_light06', '_light07', '_light08', '_light09','_light10']

for name in names:
    refl_name = name + '_reflectance.png'
    refl = cv2.imread(os.path.join(data_path, refl_name))/255.
    shad = np.zeros(refl.shape)
    for n_s in name_suffixs:
        inp = cv2.imread(os.path.join(data_path, name+n_s+'.png'))/255.
        for c in range(refl.shape[2]):
            for w in range(refl.shape[1]):
                for h in range(refl.shape[0]):
                    if inp[h][w][c] == 0:
                        shad[h][w][c] = 0
                    else:
                        shad[h][w][c] = inp[h][w][c] / refl[h][w][c]
        shad = shad*255
        shad = np.clip(shad, 0, 255)
        cv2.imwrite(os.path.join(data_path, name+'_shad'+n_s+'.png'), shad[:, :, 0])