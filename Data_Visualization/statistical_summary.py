import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def make_histogram(x,number_of_bins):

    num_bins = number_of_bins
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

def make_cdf(x,number_of_bins):

    # Anyone of the following two can be used to draw cumulative distributed functions.
    # plt.hist(x, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8, color='k')
    plt.hist(x, bins=number_of_bins, normed=True, cumulative=True, label='CDF DATA', histtype='step', alpha=0.55,
             color='purple')  # bins and (lognormal / normal) datasets are pre-defined

    plt.show()