import numpy as np
import glob
import shutil
import os

for i in glob.glob('./result/Robbery/*.npy'):
    check = np.load(i)
    print('============================')
    print(i)
    print(max(check[:]))
    if check.shape[0] != 32:
        print(i)
        print((check.shape))


# # Video move code
# result_path = './result/Robbery/'
# video_path = './dataset/UCF_Crime/Robbery/'

# get_files = os.listdir(result_path)
# video_files = os.listdir(video_path)
# file_list = []
# for i in range(len(get_files)):
#     file_list.append(get_files[i][:15] + '.mp4')

# for i in video_files:
#     if i in file_list:
#         shutil.move(video_path + i, './dataset/UCF_Crime')