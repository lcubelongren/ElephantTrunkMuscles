
from tkinter import N
from turtle import distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FactorAnalysis
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering, OPTICS, BisectingKMeans
from sklearn.mixture import GaussianMixture
import itertools

from sklearn.random_projection import SparseRandomProjection


workingDir = 'D:\Luke\ElephantTrunkMuscles'

plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=14)


def plotMetricsAll(df):
    totalmuscleNum = len(df['index'])
    normalized_df = (df - df.min()) / (df.max() - df.min())
    fig, axs = plt.subplots(1, 1, figsize=(22,4))
    ax = sns.stripplot(data=normalized_df.drop(columns=['index']), size=1, color='k', ax=axs)
    ax.set_title('All metrics of individual muscles - $n={}$'.format(totalmuscleNum))
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
    plt.suptitle('Shape metrics of individual muscles - $n={}$'.format(totalmuscleNum))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(workingDir + '\plotting_output\MetricsSize.png')


def plotMetricsDR(df):
    totalmuscleNum = len(df['index'])
    normalized_df = (df - df.min()) / (df.max() - df.min()) + df.mean()
    X = normalized_df.drop(columns=['index'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    kpca = KernelPCA(n_components=None, kernel='rbf', gamma=0.01, fit_inverse_transform=True, alpha=1)
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
    plt.subplots_adjust(top=0.85, bottom=0.3)
    plt.figtext(0.5, 0.03, 'Included metrics: ' + str(X.columns.values), 
                ha='center', fontsize=9, bbox={'alpha': 0.5, 'pad': 5})
    plt.savefig(workingDir + '\plotting_output\MetricsDR.png')
    return X_pca


def plotMetricsClustering(X, index):
    totalmuscleNum = len(X)
    
    #clustering = DBSCAN(eps=0.1, min_samples=50).fit(X)
    #components = clustering.components_
    #df = pd.DataFrame()
    #df['index'] = index
    #df['MuscleClass'] = clustering.labels_
    #df['MuscleClass'] = np.where(np.logical_and(X[:,0] < 0, X[:,1] > 0), np.ones(len(clustering.labels_)) * -1, clustering.labels_)

    df = pd.DataFrame()
    df['index'] = index
    df['MuscleClass'] = np.zeros(totalmuscleNum)

    pointsX = np.linspace(X[:,0].min(), X[:,0].max(), 2)
    pointsY = [-0.025, 0.35]

    A = np.vstack([pointsX, np.ones(len(pointsX))]).T
    m, c = np.linalg.lstsq(A, pointsY, rcond=None)[0]
    checkPoint = lambda x, y: m * x + c - y
    df['MuscleClass'] = [int(checkPoint(x, y) > 0) for x,y in zip(X[:,0], X[:,1])]

    df.to_csv(workingDir + r'/analysis_output/{}.csv'.format('muscles_p1-classes'), columns=['MuscleClass', 'index'])
    plt.figure(dpi=400)
    markers = ['<', '>']
    for i in np.unique(df['MuscleClass']):
        mask = df['MuscleClass'] == i
        plt.scatter(X[mask,0], X[mask,1], marker=markers[i], c='k', s=5)
    plt.plot(pointsX, pointsY, c='k', ls='--', lw=1, label='$y={:.2f}x+{:.2f}$'.format(m, c))
    #plt.scatter(components[:,0], components[:,1], c='r', s=2)
    #plt.suptitle('Hierarchical clustering - $n={}$'.format(totalmuscleNum))
    plt.xlabel('$PC_1$')
    plt.ylabel('$PC_2$')
    plt.axis('square')
    plt.legend()
    plt.xlim([-0.6, 0.9])
    plt.ylim([-0.2, 1.3])
    plt.tight_layout()
    plt.savefig(workingDir + '\plotting_output\MetricsClustering.png')


def plotMetricsTip():
    df_fname_dorsal = workingDir + r'/analysis_output/muscles_p1-dorsal.Label-Analysis---dorsal.csv'
    df_fname_ventral = workingDir + r'/analysis_output/muscles_p1-ventral.Label-Analysis---ventral.csv'
    df_dorsal = dfLoad(df_fname_dorsal)
    df_ventral = dfLoad(df_fname_ventral)
    df_length = pd.concat([df_dorsal[['Length3d']].rename(columns={'Length3d': 'Dorsal'}), 
                           df_ventral[['Length3d']].rename(columns={'Length3d': 'Ventral'})], axis=1)
    df_volume = pd.concat([df_dorsal[['Volume3d']].rename(columns={'Volume3d': 'Dorsal'}), 
                           df_ventral[['Volume3d']].rename(columns={'Volume3d': 'Ventral'})], axis=1)
    fig, axs = plt.subplots(2, 1, figsize=(8,6))
    sns.boxplot(data=df_length, fliersize=0, orient='h', ax=axs[0])
    sns.stripplot(data=df_length, size=4, color='k', orient='h', ax=axs[0])
    axs[0].set_xlabel(r'length [$mm$]')
    axs[0].set_xlim([0, 12])
    sns.boxplot(data=df_volume, fliersize=0, orient='h', ax=axs[1])
    sns.stripplot(data=df_volume, size=4, color='k', orient='h', ax=axs[1])
    axs[1].set_xlabel(r'volume [$mm^3$]')
    axs[1].set_xlim([0, 0.55])
    fig.tight_layout()
    fig.savefig(workingDir + '\plotting_output\MetricsTip.png')

df_fname = workingDir + r'/analysis_output/muscles_p1-dorsal.Label-Analysis.csv'

def dfLoad(df_fname):
    df = pd.read_csv(df_fname, skiprows=1)
    df = df[df['Volume3d'] != 0]  # remove indices with zero volume
    df = df.drop(columns=['Materials'])
    return df


def plotInsertionTip():
    voxelSize = 0.0179999  # mm
    regions = ['dorsal', 'ventral']
    fig, axs = plt.subplots(2, 1, sharex=False, figsize=(8,8))
    for i,region in enumerate(regions):
        df = pd.read_csv(workingDir + '/analysis_output/muscles_p1-InsertionTip_{}.csv'.format(region))

        centerPlaneDists = df['centerPlaneDists'] * voxelSize
        outputPlaneDists = df['outputPlaneDists'] * voxelSize
        inputPlaneDists = df['inputPlaneDists'] * voxelSize

        if not i:
            subtractTipDist = np.max([df['centerTipDists'], df['outputTipDists'], df['inputTipDists']])

        centerTipDists = (df['centerTipDists'] - subtractTipDist) * -voxelSize
        outputTipDists = (df['outputTipDists'] - subtractTipDist) * -voxelSize
        inputTipDists = (df['inputTipDists'] - subtractTipDist) * -voxelSize

        PlaneDistsArray = np.array([inputPlaneDists, centerPlaneDists, outputPlaneDists])
        TipDistsArray = np.array([inputTipDists, centerTipDists, outputTipDists])

        axs[i].plot(TipDistsArray, PlaneDistsArray, c='k', lw=0.2)
        axs[i].scatter(centerTipDists, centerPlaneDists, c='k', s=2, label='barycenter')
        axs[i].scatter(outputTipDists, outputPlaneDists, c='b', s=2, label='start')
        axs[i].scatter(inputTipDists, inputPlaneDists, c='r', s=2, label='end')
        axs[i].set_title('{}'.format(region))
        #axs[i].set_xlim([-0.5,15.5])
    axs[0].set_ylim([0,10])
    axs[1].set_ylim([0,14])
    #[axs[i].set_aspect('equal', adjustable='box') for i in range(2)]
        

    axs[1].set_xlabel(r'distance from tip [$mm$]')
    axs[0].legend()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel(r'distance from medial plane [$mm$]')
    fig.savefig(workingDir + '\plotting_output\InsertionTip.png')


#plotMetricsAll(df)
#plotMetricsSize(len(df['index']), df['Length3d'], df['Area3d'], df['Volume3d'])
#X = plotMetricsDR(df)
#plotMetricsClustering(X, index=df['index'])
#plotMetricsTip()
#plotInsertionTip()



region = 'dorsal'
df_Metrics = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_{0}\muscles_p1-{0}.Label-Analysis_Metrics.csv'.format(region), skiprows=1)
df_DistanceTransform = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_{0}\muscles_p1-{0}.Label-Analysis_DistanceTransform.csv'.format(region), skiprows=1)
df_PropagationDistance = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_{0}\muscles_p1-{0}.Label-Analysis_PropagationDistance.csv'.format(region), skiprows=1)

print(df_DistanceTransform.shape, df_PropagationDistance.shape, df_Metrics.shape)

df = pd.DataFrame()
#df['DistanceTransformMinimum'] = df_DistanceTransform['Minimum']
#df['DistanceTransformMean'] = df_DistanceTransform['Mean']
#df['DistanceTransformMaximum'] = df_DistanceTransform['Maximum']
df['DistanceTransformDelta'] = df_DistanceTransform['Maximum'] - df_DistanceTransform['Minimum']
#df['PropagationDistanceMinimum'] = df_PropagationDistance['Minimum']
#df['PropagationDistanceMean'] = df_PropagationDistance['Mean']
#df['PropagationDistanceMaximum'] = df_PropagationDistance['Maximum']
df['PropagationDistanceDelta'] = df_PropagationDistance['Maximum'] - df_PropagationDistance['Minimum']
df['Length3d'] = df_Metrics['Length3d']
df['Volume3d'] = df_Metrics['Volume3d']
df['index'] = df_Metrics['index']
plotMetricsAll(df)
X = plotMetricsDR(df)
plotMetricsClustering(X, df['index'])





# def plotTrunkAnalysis(trunkArea, muscleArea, fasciclesArea):
#     trunkStart = np.where(trunkArea > 1e-3)[0][0]
#     muscleStart = np.where(muscleArea > 1e-3)[0][0]
#     fasciclesStart = np.where(fasciclesArea > 1e-3)[0][0]
#     plt.figure(figsize=(5, 10), dpi=200)
#     plt.plot(trunkArea[trunkStart:], 
#              (np.arange(trunkStart, len(trunkArea))-trunkStart)*(self.voxelSize*2/10), label='trunk')
#     plt.plot(muscleArea[muscleStart:], 
#              (np.arange(muscleStart, len(muscleArea))-trunkStart)*(self.voxelSize*2/10), label='muscle')
#     plt.plot(fasciclesArea[fasciclesStart:], 
#              (np.arange(fasciclesStart, len(fasciclesArea))-trunkStart*2)*(self.voxelSize/10), label='fascicles')
#     plt.title('Area of each trunk region')
#     plt.xlabel('area [$cm^2$]')
#     plt.ylabel('distance from tip [$cm$]')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('plotting_output/TrunkAnalysis.png')
