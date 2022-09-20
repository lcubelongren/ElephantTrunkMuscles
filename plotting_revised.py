
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats


workingDir = 'D:\Luke\ElephantTrunkMuscles'
tip_voxelSize = 0.0179999  # mm

plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=18)


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
        plt.figure(figsize=(5,5), dpi=500)
        markers = ['<', '>']
        colors = ['tab:orange', 'tab:blue']
        for i in np.unique(df_class['MuscleClass']):
            mask = df_class['MuscleClass'] == i
            plt.scatter(self.X[mask,0], self.X[mask,1], marker=markers[i], c=colors[i], s=20)
        plt.plot(pointsX, pointsY, c='k', ls='--', lw=1, label='$y={:.2f}x+{:.2f}$'.format(m, c))
        plt.xlabel('$PC_1$')
        plt.ylabel('$PC_2$')
        plt.axis('square')
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
#TD.run()


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
        plt.figure(figsize=(5,5), dpi=500)
        markers = ['^', 'v']
        colors = ['tab:orange', 'tab:blue']
        for i in np.unique(df_class['MuscleClass']):
            mask = df_class['MuscleClass'] == i
            plt.scatter(self.X[mask,0], self.X[mask,1], marker=markers[i], c=colors[i], s=20)
        plt.plot(pointsX, pointsY, c='k', ls='--', lw=1, label='$y={:.2f}x+{:.2f}$'.format(m, c))
        plt.xlabel('$PC_1$')
        plt.ylabel('$PC_2$')
        plt.axis('square')
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


plt.rc('font', size=16)


class TipShared:
    def __init__(self):
        pass

    def plotMetricsTip(self):
        df_metrics_dorsal = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_dorsal\muscles_p1-dorsal.Label-Analysis_Metrics.csv', skiprows=1)
        df_metrics_ventral = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_ventral\muscles_p1-ventral.Label-Analysis_Metrics.csv', skiprows=1)
        df_metrics_ventral['index'] += len(df_metrics_dorsal['Length3d'])

        df_classes_dorsal = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_dorsal\muscles_p1-dorsal.classes.csv')
        df_classes_ventral = pd.read_csv(r'D:\Luke\current\IMPORTANT\tip_ventral\muscles_p1-ventral.classes.csv')
        df_metrics_ventral['index'] += len(df_classes_dorsal['MuscleClass'])
        df_metrics_dorsal['MuscleClass'] = df_classes_dorsal['MuscleClass']
        df_metrics_ventral['MuscleClass'] = df_classes_ventral['MuscleClass']

        df_length = pd.concat([df_metrics_dorsal[['Length3d', 'MuscleClass']], 
                               df_metrics_ventral[['Length3d', 'MuscleClass']]]).reset_index()
        df_length['index'] = df_length.index
        df_length['Region'] = np.where(df_length.index < len(df_metrics_dorsal), 'Dorsal', 'Ventral')
        df_length['Length3d'] *= tip_voxelSize

        df_volume = pd.concat([df_metrics_dorsal[['Volume3d', 'MuscleClass']], 
                               df_metrics_ventral[['Volume3d', 'MuscleClass']]]).reset_index()
        df_volume['index'] = df_volume.index
        df_volume['Region'] = np.where(df_volume.index < len(df_metrics_dorsal), 'Dorsal', 'Ventral')
        df_volume['Volume3d'] *= tip_voxelSize**3

        dfs = [df_length, df_volume]
        metrics1 = ['Length3d', 'Volume3d']
        metrics2 = [r'length [$mm$]', r'volume [$mm^3$]']
        xlims = [[0, 12], [0, 0.55]]

        fig, axs = plt.subplots(2, 1, figsize=(6,6), dpi=500)
        for i in range(len(metrics1)):
            sns.boxplot(data=dfs[i].drop(columns=['MuscleClass']), x=metrics1[i], y='Region', width=0.9, fliersize=0, color='w', orient='h', ax=axs[i])
            markers = ['X', 'o']
            colors = ['tab:orange', 'tab:blue']
            for j in np.unique(dfs[i]['MuscleClass']):
                sns.stripplot(data=dfs[i].where(dfs[i]['MuscleClass'] == j), x=metrics1[i], y='Region', marker=markers[j], color=colors[j],
                              size=3.5, jitter=0.3, orient='h', ax=axs[i])
            axs[i].set_xlabel(metrics2[i])
            axs[i].set_ylabel(None)
            axs[i].set_xlim(xlims[i])
            axs[i].legend([],[], frameon=False)

            statistic, pvalue = stats.ttest_ind(dfs[i].where(dfs[i]['Region'] == 'Dorsal').dropna()[metrics1[i]], 
                                                dfs[i].where(dfs[i]['Region'] == 'Ventral').dropna()[metrics1[i]], 
                                                equal_var=False, alternative='two-sided')
            print(metrics1[i], pvalue)

        fig.tight_layout()
        fig.savefig(workingDir + '\plotting_output\MetricsTip.png')

    def run(self):
        self.plotMetricsTip()

TS = TipShared()
TS.run()

