
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
tip_voxelSize = 0.0179999  # mm

plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=14)


class TipDorsal:
    def __init__(self):
        self.region = 'dorsal'

    def plotMetricsClustering(self, df):
        totalmuscleNum = len(df['index'])
        normalized_df = (df - df.min()) / (df.max() - df.min()) + df.mean()
        X = normalized_df.drop(columns=['index'])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        self.X = X_pca

        df_class = pd.DataFrame()
        df_class['MuscleClass'] = np.zeros(totalmuscleNum)
        df_class['index'] = df['index']

        pointsX = np.linspace(self.X[:,0].min(), self.X[:,0].max(), 2)
        pointsY = [-0.025, 0.35]

        A = np.vstack([pointsX, np.ones(len(pointsX))]).T
        m, c = np.linalg.lstsq(A, pointsY, rcond=None)[0]
        checkPoint = lambda x, y: m * x + c - y
        df_class['MuscleClass'] = [int(checkPoint(x, y) > 0) for x,y in zip(self.X[:,0], self.X[:,1])]

        df_class.to_csv(workingDir + r'/analysis_output/{}.csv'.format('muscles_p1-classes'), columns=['MuscleClass', 'index'])
        plt.figure(dpi=400)
        markers = ['<', '>']
        colors = ['tab:orange', 'tab:blue']
        for i in np.unique(df_class['MuscleClass']):
            mask = df_class['MuscleClass'] == i
            plt.scatter(self.X[mask,0], self.X[mask,1], marker=markers[i], c=colors[i], s=5)
        plt.plot(pointsX, pointsY, c='k', ls='--', lw=1, label='$y={:.2f}x+{:.2f}$'.format(m, c))
        plt.xlabel('$PC_1$')
        plt.ylabel('$PC_2$')
        plt.axis('square')
        plt.legend(loc='upper right')
        plt.xlim([-0.6, 0.9])
        plt.ylim([-0.2, 1.3])
        plt.tight_layout()
        plt.savefig(workingDir + '\plotting_output\MetricsClustering.png')

    def run(self):
        df_Metrics = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_{0}\muscles_p1-{0}.Label-Analysis_Metrics.csv'.format(self.region), skiprows=1)
        df_DistanceTransform = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_{0}\muscles_p1-{0}.Label-Analysis_DistanceTransform.csv'.format(self.region), skiprows=1)
        df_PropagationDistance = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_{0}\muscles_p1-{0}.Label-Analysis_PropagationDistance.csv'.format(self.region), skiprows=1)
        df = pd.DataFrame()
        df['DistanceTransformDelta'] = df_DistanceTransform['Maximum'] - df_DistanceTransform['Minimum']
        df['PropagationDistanceDelta'] = df_PropagationDistance['Maximum'] - df_PropagationDistance['Minimum']
        df['Length3d'] = df_Metrics['Length3d']
        df['Volume3d'] = df_Metrics['Volume3d']
        df['index'] = df_Metrics['index']

        self.plotMetricsClustering(df)

TD = TipDorsal()
TD.run()


# ------------------------------------


class TipVentral:
    def __init__(self):
        self.region = 'ventral'

    def plotMetricsClustering(self, df):
        totalmuscleNum = len(df['index'])
        normalized_df = (df - df.min()) / (df.max() - df.min()) + df.mean()
        X = normalized_df.drop(columns=['index'])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        self.X = X_pca

        df_class = pd.DataFrame()
        df_class['MuscleClass'] = np.zeros(totalmuscleNum)
        df_class['index'] = df['index']

        pointsX = np.linspace(self.X[:,0].min(), self.X[:,0].max(), 2)
        pointsY = [0.3, 0.3]

        A = np.vstack([pointsX, np.ones(len(pointsX))]).T
        m, c = np.linalg.lstsq(A, pointsY, rcond=None)[0]
        checkPoint = lambda x, y: m * x + c - y
        df_class['MuscleClass'] = [int(checkPoint(x, y) > 0) for x,y in zip(self.X[:,0], self.X[:,1])]

        df_class.to_csv(workingDir + r'/analysis_output/{}.csv'.format('muscles_p1-classes'), columns=['MuscleClass', 'index'])
        plt.figure(dpi=400)
        markers = ['^', 'v']
        colors = ['tab:orange', 'tab:blue']
        for i in np.unique(df_class['MuscleClass']):
            mask = df_class['MuscleClass'] == i
            plt.scatter(self.X[mask,0], self.X[mask,1], marker=markers[i], c=colors[i], s=5)
        plt.plot(pointsX, pointsY, c='k', ls='--', lw=1, label='$y={:.2f}x+{:.2f}$'.format(m, c))
        plt.xlabel('$PC_1$')
        plt.ylabel('$PC_2$')
        plt.axis('square')
        plt.legend(loc='upper right')
        plt.xlim([-0.6, 0.7])
        plt.ylim([-0.3, 1.0])
        plt.tight_layout()
        plt.savefig(workingDir + '\plotting_output\MetricsClustering.png')

    def run(self):
        df_Metrics = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_{0}\muscles_p1-{0}.Label-Analysis_Metrics.csv'.format(self.region), skiprows=1)
        df = pd.DataFrame()
        df['OrientationPhi'] = df_Metrics['OrientationPhi']
        df['OrientationTheta'] = df_Metrics['OrientationTheta']
        df['index'] = df_Metrics['index']

        self.plotMetricsClustering(df)

TV = TipVentral()
#TV.run()


# ------------------------------------


class TipShared:
    def __init__(self):
        pass

    def plotMetricsTip(self):
        df_metrics_dorsal = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_dorsal\muscles_p1-dorsal.Label-Analysis_Metrics.csv', skiprows=1)
        df_metrics_ventral = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_ventral\muscles_p1-ventral.Label-Analysis_Metrics.csv', skiprows=1)
        df_classes_dorsal = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_dorsal\muscles_p1-dorsal.classes.csv')
        df_classes_ventral = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_ventral\muscles_p1-ventral.classes.csv')

        df_length = pd.concat([df_metrics_dorsal[['Length3d']].rename(columns={'Length3d': 'Dorsal'}), 
                               df_metrics_ventral[['Length3d']].rename(columns={'Length3d': 'Ventral'})], axis=1) * tip_voxelSize
        df_volume = pd.concat([df_metrics_dorsal[['Volume3d']].rename(columns={'Volume3d': 'Dorsal'}), 
                               df_metrics_ventral[['Volume3d']].rename(columns={'Volume3d': 'Ventral'})], axis=1) * tip_voxelSize**3

        fig, axs = plt.subplots(2, 1, figsize=(8,6))
        sns.boxplot(data=df_length, fliersize=0, color='w', orient='h', ax=axs[0])
        sns.stripplot(data=df_length, size=4, color='k', orient='h', ax=axs[0])
        axs[0].set_xlabel(r'length [$mm$]')
        axs[0].set_xlim([0, 12])

        sns.boxplot(data=df_volume, fliersize=0, color='w', orient='h', ax=axs[1])
        sns.stripplot(data=df_volume, size=4, color='k', orient='h', ax=axs[1])
        axs[1].set_xlabel(r'volume [$mm^3$]')
        axs[1].set_xlim([0, 0.55])
        fig.tight_layout()
        fig.savefig(workingDir + '\plotting_output\MetricsTip.png')

    def run(self):
        self.plotMetricsTip()

TS = TipShared()
#TS.run()

