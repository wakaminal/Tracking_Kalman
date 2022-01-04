import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import numpy as np
import pickle

i = 0
for filename in glob.glob('/Users/nivedha.sivakumar94/Desktop/Tracker/normalized_frames/video_3/Binary/*.png'):
    print('Saving image: ' + format(i))
    img_name = filename[76:]
    img = misc.imread(filename)
    if len(img_name) == 7:
        new_name = 'new' + img_name
    elif len(img_name) == 6:
        suffix = format(0) + img_name
        new_name = 'new' + suffix
    else:
        suffix = format(0) + format(0) + img_name
        new_name = 'new' + suffix
    misc.imsave(new_name, img)
    i += 1
