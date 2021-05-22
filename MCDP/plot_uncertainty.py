import numpy as np
import matplotlib.pyplot as plt

Y_ts = np.load("D:/PROJECTS/internship/MCDP/Y_ts.npy")

X_ts = np.load("D:/PROJECTS/internship/MCDP/X_ts.npy")

Y_ts_hat = np.load("D:/PROJECTS/internship/MCDP/Y_ts_hat.npy")

U_ts = np.load("D:/PROJECTS/internship/MCDP/U_ts.npy")

U_ts_foreground = np.load("D:/PROJECTS/internship/MCDP/U_ts_foreground.npy")

U_ts_background = np.load("D:/PROJECTS/internship/MCDP/U_ts_background.npy")

dice = np.load("D:/PROJECTS/internship/MCDP/dice.npy")

X_ts = X_ts[:20]

Ntest = len(X_ts)

Y_ts = Y_ts[:20]

Y_ts_hat = Y_ts_hat[:20]

U_ts = U_ts[:20]

dice = dice[:20]


fig, axes = plt.subplots(Ntest, 6, figsize=(4 * 7, Ntest * 4))
for i in range(Ntest):
    axes[i, 0].imshow(X_ts[i, :, :, 0], cmap='gray')
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    axes[i, 0].set_title('input image', {'fontsize': 16})

    axes[i, 1].imshow(Y_ts[i, :, :, 0], cmap='gray')
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])
    axes[i, 1].set_title('True mask', {'fontsize': 16})

    axes[i, 2].imshow(Y_ts_hat[i, :, :, 0], cmap='gray')
    axes[i, 2].set_xticks([])
    axes[i, 2].set_yticks([])
    axes[i, 2].set_title('Pred mask, dice=' + str(np.round(dice[i], 2)), {'fontsize': 16})

    axes[i, 3].imshow(U_ts[i, :, :, 0], cmap='gray')
    axes[i, 3].set_xticks([])
    axes[i, 3].set_yticks([])
    axes[i, 3].set_title('Model Uncertainty', {'fontsize': 16})

    axes[i, 4].imshow(U_ts_foreground[i, :, :, 0], cmap='gray')
    axes[i, 4].set_xticks([])
    axes[i, 4].set_yticks([])
    axes[i, 4].set_title('FG Uncertainty', {'fontsize': 16})

    axes[i, 5].imshow(U_ts_background[i, :, :, 0], cmap='gray')
    axes[i, 5].set_xticks([])
    axes[i, 5].set_yticks([])
    axes[i, 5].set_title('BG Uncertainty', {'fontsize': 16})

plt.savefig("all_uncertainty_plot.jpg", bbox_inces='tight',
            dpi=100)


