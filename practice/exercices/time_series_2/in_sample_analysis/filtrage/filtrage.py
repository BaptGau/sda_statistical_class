import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from pandas import Series
from demos.utlis import setup_plot, get_noisy_time_series

if __name__ == "__main__":

    data = get_noisy_time_series()
    colors = setup_plot()

    # todo: utiliser et afficher les filtres suivants: moyenne mobile (par produit de convolution), filtre gaussien, filtre de Savitzky-Golay

    window = 12
    sigma = 3

    to_be_plotted = {
        "data": data,
        "moyenne mobile": data.rolling(window=window).mean(),
        "filtre gaussien": gaussian_filter1d(data, sigma=sigma),
        "savitzky-golay": savgol_filter(data, window_length=window, polyorder=2),
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axes = axes.flatten()
    for i, (title, sample) in enumerate(to_be_plotted.items()):
        axes[i].plot(data.index, sample, color=colors[i])
        if title != "data":
            axes[i].plot(data, color=colors[0], alpha=0.3)
        axes[i].set_title(title)
    plt.grid(True)
    plt.show()
