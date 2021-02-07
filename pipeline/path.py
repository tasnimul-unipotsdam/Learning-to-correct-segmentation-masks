import os

MASK_DIR_PATH = "D:/PROJECTS/internship/femur_2d_segmentations/middle_slices"
CORRUPT_MASK_PATH = "D:/PROJECTS/internship/images"


mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

corrupt_mask_path =  [os.path.join(CORRUPT_MASK_PATH, x) for x in os.listdir(CORRUPT_MASK_PATH) if x.endswith('.png')]

print(len(mask_paths))
print(len(corrupt_mask_path))



