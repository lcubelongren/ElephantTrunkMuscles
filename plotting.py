
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
import itertools


workingDir = 'D:\Luke\ElephantTrunkMuscles'


def plotMetricsAll(df):
    totalmuscleNum = len(df['index'])
    normalized_df = (df - df.min()) / (df.max() - df.min())
    ax = sns.stripplot(data=normalized_df.drop(columns=['index']), size=1)
    ax.set_title('Individual muscle measurements - $n={}$'.format(totalmuscleNum))
    ax.set_ylabel('normalized value')
    fig = ax.get_figure()
    fig.savefig(workingDir + '\plotting_output\MetricsAll.png')


def plotMetricsSize(totalmuscleNum, lengths, surface_areas, volumes):
    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    c, s = 'k', 3
    names = ['length', 'surface area', 'volume']
    units = ['mm', 'mm^2', 'mm^3']
    values = [lengths, surface_areas, volumes]
    for i,(j,k) in enumerate(itertools.combinations(np.arange(len(values)), 2)):
        axs[i].scatter(values[j], values[k], c=c, s=s)
        m, b = np.polyfit(x=values[j], y=values[k], deg=1)
        axs[i].plot(np.unique(values[j]), np.poly1d([m, b])(np.unique(values[j])), 
                    label='$y = {:.2f}x {:+.2f}$'.format(m, b), c=c)
        axs[i].set_xlabel('{} [${}$]'.format(names[j], units[j]))
        axs[i].set_ylabel('{} [${}$]'.format(names[k], units[k]))
        axs[i].axis('tight')
        axs[i].legend()
    plt.suptitle('Individual muscle measurements - $n={}$'.format(totalmuscleNum))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(workingDir + '\plotting_output\MetricsSize.png')


def plotMetricsDR(df):
    totalmuscleNum = len(df['index'])
    normalized_df = (df - df.min()) / (df.max() - df.min())
    X = normalized_df.drop(columns=['index', 'OrientationPhi'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    kpca = KernelPCA(n_components=None, kernel='rbf', gamma=10, fit_inverse_transform=True, alpha=0.1)
    X_kpca = kpca.fit_transform(X)
    fig, axs = plt.subplots(ncols=2, figsize=(8,4))
    axs[0].scatter(X_pca[:,0], X_pca[:,1], s=2)
    axs[0].set_title('PCA')
    axs[1].scatter(X_kpca[:,0], X_kpca[:,1], s=2)
    axs[1].set_title('Kernel PCA')
    [axs[i].set_xlabel('$PC_1$') for i in range(2)]
    [axs[i].set_ylabel('$PC_2$') for i in range(2)]
    [axs[i].axis('square') for i in range(2)]
    plt.suptitle('Dimensionality reduction - $n={}$'.format(totalmuscleNum))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2)
    plt.figtext(0.5, 0.03, 'Included metrics: ' + str(X.columns.values), 
                ha='center', fontsize=9, bbox={'alpha': 0.5, 'pad': 5})
    plt.savefig(workingDir + '\plotting_output\MetricsDR.png')


df_fname = workingDir + r'\analysis_output\muscles_p1-combined.Label-Analysis.csv'
df = pd.read_csv(df_fname, skiprows=1).drop(columns=['Materials'])
df = df[np.isfinite(df).all(True)]  # remove rows with inf
plotMetricsAll(df)
plotMetricsSize(len(df['index']), df['Length3d'], df['VoxelFaceArea'], df['Volume3d'])
plotMetricsDR(df)



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
