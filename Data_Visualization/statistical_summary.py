import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from random import randint
import collections

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

def make_histogram_twoset(x,y,number_of_bins):

    num_bins = number_of_bins
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.hist(y, num_bins, facecolor='blue', alpha=0.5)

    plt.show()

def make_cdf_twoset(x,y,number_of_bins):

    # Anyone of the following two can be used to draw cumulative distributed functions.
    # plt.hist(x, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8, color='k')
    plt.hist(x, bins=number_of_bins, normed=True, cumulative=True, label='CDF DATA', histtype='step', alpha=0.55,
             color='green')  # bins and (lognormal / normal) datasets are pre-defined
    plt.hist(y, bins=number_of_bins, normed=True, cumulative=True, label='CDF DATA', histtype='step', alpha=0.55,
             color='blue')

    plt.show()

def make_histogram_multipledataSet(array_of_dataset, number_of_bin):

    colors = ["red","blue","green","yellow","gray","orange","purple","red","blue","green"]

    for i in range(0,len(array_of_dataset)):
        plt.hist(array_of_dataset[i], number_of_bin, facecolor=colors[1], alpha=0.5)

    plt.show()

def make_plot(input, semilog_y = False):
    freq = collections.Counter(input)

    flower_count=[]
    frequency =[]
    for key,val in freq.items():
        flower_count.append(key)
        frequency.append(val)

    if (semilog_y == True):
        plt.semilogy(flower_count,frequency)

    else :
        plt.plot(flower_count,frequency)

    plt.show()

def CountFrequency(arr):
    freq = collections.Counter(arr)
    for key, value in freq.items():
        print(key, " : ", value)