import matplotlib.pyplot as plt

from pipeline.dataloader import DataLoader
from pipeline.path import train_mask_path, train_corrupt_mask_path

train_dataset = DataLoader(train_corrupt_mask_path, train_mask_path, training=True).load_data()
# validation_dataset = DataLoader(validation_image_path, validation_mask_path, training=False).load_data()

image, mask = next(iter(train_dataset))

for i in range(4):
    plt.plot(2, 2)
    plt.imshow(mask[i])
    plt.show()
