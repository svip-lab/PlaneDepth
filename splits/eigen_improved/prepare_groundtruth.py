import os
from PIL import Image
import numpy as np

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

filenames = readlines("./test_files.txt")
new_filenames = []
GT_depths = []
count = 0
for index in range(len(filenames)):
    line = filenames[index].split()
    folder = line[0]
    if len(line) == 3:
        frame_index = int(line[1])
    else:
        frame_index = 0
    GT_path = os.path.join("/public/home/wangry3/datasets/kitti_depth/train/", line[0][11:], "proj_depth/groundtruth/image_02", line[1]+".png")
    if os.path.exists(GT_path):
        count += 1
        new_filenames.append(filenames[index]+"\n")
        GT_depth = np.array(Image.open(GT_path)) / 255.
        GT_depths.append(GT_depth)
    GT_path = GT_path.replace("train", "val")
    if os.path.exists(GT_path):
        count += 1
        new_filenames.append(filenames[index]+"\n")
        GT_depth = np.array(Image.open(GT_path)) / 255.
        GT_depths.append(GT_depth)
# GT_depths = np.array(GT_depths)
np.savez('gt_depths.npz',data=GT_depths)
with open("new_test_files.txt", 'w') as f:
    lines = f.writelines(new_filenames)
print(count)