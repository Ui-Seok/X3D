import numpy as np
import glob

for i in glob.glob('./result/Burglary/*.npy'):
    check = np.load(i)
    if check.shape[0] != 32:
        print(i)
        print((check.shape))