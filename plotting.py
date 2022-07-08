
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import itertools


def plotMuscleAnalysis(totalmuscleNum, volumes, lengths, surface_areas):
    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    c, s = 'k', 3
    names = ['length', 'surface area', 'volume']
    units = ['mm', 'mm^2', 'mm^3']
    values = [lengths, surface_areas, volumes]
    for i,(j,k) in enumerate(itertools.combinations(np.arange(len(values)), 2)):
        axs[i].scatter(values[j], values[k], c=c, s=s)
        m, b = np.polyfit(values[j], values[k], 1)
        axs[i].plot(np.unique(values[j]), np.poly1d([m, b])(np.unique(values[j])), 
                    label='$y = {:.2f}x {:+.2f}$'.format(m, b), c=c)
        axs[i].set_xlabel('{} [${}$]'.format(names[j], units[j]))
        axs[i].set_ylabel('{} [${}$]'.format(names[k], units[k]))
        axs[i].axis('tight')
        axs[i].legend()
    plt.suptitle('Individual muscle measurements - $n={}$'.format(totalmuscleNum))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('plotting_output/MuscleAnalysis.png')
    
    
df = pd.read_csv('analysis_output/muscles_p1.csv')
plotMuscleAnalysis(len(df), df['volumes'], df['lengths'], df['surface areas'])


def plotTrunkAnalysis(trunkArea, muscleArea, fasciclesArea):
    trunkStart = np.where(trunkArea > 1e-3)[0][0]
    muscleStart = np.where(muscleArea > 1e-3)[0][0]
    fasciclesStart = np.where(fasciclesArea > 1e-3)[0][0]
    plt.figure(figsize=(5, 10), dpi=200)
    plt.plot(trunkArea[trunkStart:], 
             (np.arange(trunkStart, len(trunkArea))-trunkStart)*(self.voxelSize*2/10), label='trunk')
    plt.plot(muscleArea[muscleStart:], 
             (np.arange(muscleStart, len(muscleArea))-trunkStart)*(self.voxelSize*2/10), label='muscle')
    plt.plot(fasciclesArea[fasciclesStart:], 
             (np.arange(fasciclesStart, len(fasciclesArea))-trunkStart*2)*(self.voxelSize/10), label='fascicles')
    plt.title('Area of each trunk region')
    plt.xlabel('area [$cm^2$]')
    plt.ylabel('distance from tip [$cm$]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plotting_output/TrunkAnalysis.png')
