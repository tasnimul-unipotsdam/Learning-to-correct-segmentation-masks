import numpy as np
import matplotlib.pyplot as plt

X_ts = np.load("D:/PROJECTS/internship/3D_data/test_input_image.npy")

Y_ts = np.load("D:/PROJECTS/internship/3D_data/test_label_image.npy")

Y_ts_hat = np.load("D:/PROJECTS/internship/MCDP_3D/Y_ts_hat_3D.npy")

U_ts = np.load("D:/PROJECTS/internship/MCDP_3D/U_ts_3D.npy")

dice = np.load("D:/PROJECTS/internship/MCDP_3D/dice_mcdp_3D.npy")

Ntest = 5

# threshold = 0.5
# Y_ts_hat[Y_ts_hat < threshold] = 0
# Y_ts_hat[Y_ts_hat >= threshold] = 1
#
# U_ts[U_ts < threshold] = 0
# U_ts[U_ts >= threshold] = 1

fig, axes = plt.subplots(5, 5, figsize=(8 * 8, Ntest * 8))
for i in range(Ntest):
    axes[i, 0].imshow(X_ts[i, :, 95, :, 0])
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    axes[i, 0].set_title('input image', {'fontsize': 16})

    axes[i, 1].imshow(Y_ts[i, :, 95, :, 0])
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])
    axes[i, 1].set_title('True mask', {'fontsize': 16})

    axes[i, 2].imshow(Y_ts_hat[i, :, 95, :, 0])
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])
    axes[i, 2].set_title('Pred mask, dice=' + str(np.round(dice[i], 2)), {'fontsize': 16})

    axes[i, 3].imshow(U_ts[i, :, 95, :, 0])
    axes[i, 3].set_xticks([])
    axes[i, 3].set_yticks([])
    axes[i, 3].set_title('Model Uncertainty', {'fontsize': 16})

    axes[i, 4].imshow(Y_ts_hat[i, :, 95, :, 0])
    axes[i, 4].imshow(U_ts[i, :, 95, :, 0], alpha=.5)
    axes[i, 4].set_xticks([])
    axes[i, 4].set_yticks([])
    axes[i, 4].set_title('Prediction & Uncertainty Overlay', {'fontsize': 16})

plt.savefig("uncertainty_plot_3D.jpg", dpi=200)
