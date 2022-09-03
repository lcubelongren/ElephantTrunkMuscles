
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from PIL import Image
from skimage import io, morphology, measure
import skan
import pandas as pd


workingDir = 'D:\Luke\ElephantTrunkMuscles'


def crop2labels(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return xmin, xmax, ymin, ymax, zmin, zmax


def combineMuscleImages(volDir, subfolder, volShape):
    print('processing partition:', subfolder)
    outputVol = np.zeros(volShape, dtype=np.uint16)
    print(volDir + '/' + subfolder)
    paths = Path(volDir + '/' + subfolder).rglob('*.h5am')
    print('combining a total of {} label volumes'.format(len(list(paths))))
    for p,path in enumerate(Path(volDir + '/' + subfolder).rglob('*.h5am')):  # (paths)
        print('adding image:', path.name)
        with h5py.File(path, 'r') as f:
            Vol = f['amira']['dataset:0']['timestep:0'][()].squeeze().astype(np.uint16)
            newVol = Vol + (p * (2**8 - 2)) - 1
            outputVol[:] = np.where(Vol > 1, newVol, outputVol)
    io.imsave('analysis_output/{}-combined.tif'.format(subfolder), outputVol, check_contrast=False)
    print('finished combining label files')
 

voxelSize = 0.0179999  # mm

#volDir = '/media/lukel/BT-Longren/TrunkMusclePaper/HOAS-BACKUP/segmentations/muscles'  # Linux machine
volDir = 'F:/TrunkMusclePaper/HOAS-BACKUP/segmentations/tip'  # Windows machine
subfolder = 'muscles_p1'  # tip
#subfolder = 'muscles_p2'  # medial
#subfolder = 'muscles_p3'  # posterior
volShape = (1075, 2150, 2150)  # muscles_p1
#combineMuscleImages(volDir, subfolder, volShape)  # save to a single .tif volume

# ----------

def combineMuscleClass(subfolder='muscles_p1'):
    print('mapping predicted class values to each muscle in partition {}'.format(subfolder))
    labelVol = np.fromfile(r'D:\Luke\current\muscles_p1_analysis-files\muscles_p1-final.raw', dtype=np.uint16)
    labelVol = np.reshape(labelVol, volShape)
    labels = np.unique(labelVol)
    classes0 = pd.read_csv(workingDir + '/analysis_output/{}-classes.csv'.format(subfolder))['MuscleClass']
    classes = np.append([0], np.array(classes0) + 1)
    from skimage.util._map_array import map_array
    outputVol = map_array(labelVol, labels, classes).astype(np.uint8)
    io.imsave(workingDir + '/analysis_output/{}-classes.tif'.format(subfolder), outputVol, check_contrast=False)


combineMuscleClass()

# ----------

# try:
#     df = pd.read_csv('analysis_output/{}.csv'.format(subfolder), index_col=0)
#     print('read existing .csv')
# except:
#     df = pd.DataFrame()
#     print('created new .csv')

# print('Loading and cropping volume')
# inputVol_full = io.imread('analysis_output/{}-combined.tif'.format(subfolder))
# xmin, xmax, ymin, ymax, zmin, zmax = crop2labels(inputVol_full)
# inputVol = inputVol_full[xmin:xmax, ymin:ymax, zmin:zmax]
# print('Input volume partition shape:', inputVol.shape)


class MuscleAnalysis:
    
    def __init__(self, df, voxelSize, subfolder):
        print('Running MuscleAnalysis')
        self.df = df
        self.voxelSize = voxelSize
        self.subfolder = subfolder

    def calculateVolumes(self, image):
        print('Calculating volumes of all muscles')
        values, counts = np.unique(image, return_counts=True)
        volumes = counts[1:] * self.voxelSize**3
        self.muscleNum = len(volumes)
        self.df = self.df.reindex(values[1:])
        self.df['volumes'] = volumes
        
    def calculateLengths(self, image):
        for i in range(1, self.muscleNum + 1):
            if i in [250, 773]: continue  # error: negative number in skan
            
            print('Calculating length of muscle: {}/{}'.format(i, self.muscleNum))
            try:
                if self.df.at[i, 'lengths'] > 0:
                    print('-> skipping, already filled')
                    continue
            except:
                pass
            skeleton = morphology.skeletonize_3d(image == i)
            S = skan.csr.Skeleton(skeleton, spacing=self.voxelSize)
            length = np.nanmax(S.path_lengths())
            # save each iteration
            self.df.at[i, 'lengths'] = length
            self.df.to_csv('analysis_output/{}.csv'.format(self.subfolder))
            
    def calculateSurfaceAreas(self, image):
        for i in range(1, self.muscleNum + 1):
            print('Calculating surface area of muscle: {}/{}'.format(i, self.muscleNum))
            try:
                if self.df.at[i, 'surface areas'] > 0:
                    print('-> skipping, already filled')
                    continue
            except:
                pass
            mask = image == i
            spacing = (self.voxelSize, self.voxelSize, self.voxelSize)
            verts, faces, normals, values = measure.marching_cubes(mask, spacing=spacing)
            area = measure.mesh_surface_area(verts, faces)
            # save each iteration
            self.df.at[i, 'surface areas'] = area
            self.df.to_csv('analysis_output/{}.csv'.format(self.subfolder))
            
    def run(self, image):
        self.calculateVolumes(image)
        self.calculateLengths(image)
        self.calculateSurfaceAreas(image)
        # post-process df
        #self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]  # remove index columns
        self.df.to_csv('analysis_output/{}.csv'.format(self.subfolder))
        print('Completed MuscleAnalysis')


#MA = MuscleAnalysis(df, voxelSize, subfolder)
#MA.run(inputVol)


class TrunkAnalysis:

    def __init__(self, voxelSize):
        self.trunkDir = '/media/lukel/BT-Longren/TrunkMusclePaper/HOAS-BACKUP/segmentations'
        self.voxelSize = voxelSize
        
    def processVolumes(self):
        print('Loading files...')
        f_trunk = h5py.File(self.trunkDir + '/whole_trunk/cvolume-filtered2.to-byte.Mask.h5am', 'r')
        f_muscle = h5py.File(self.trunkDir + '/whole_trunk/cvolume-filtered2.labels-interpolated.h5am', 'r')
        trunkVol = f_trunk['amira']['dataset:0']['timestep:0'][()].squeeze().astype('uint8', copy=False)
        muscleVol = f_muscle['amira']['dataset:0']['timestep:0'][()].squeeze().astype('uint8', copy=False)
        fasciclesVol = io.imread('analysis_output/muscles_p1-combined.tif')
        return trunkVol, muscleVol, fasciclesVol
        
    def calculateAreas(self, trunkVol, muscleVol, fasciclesVol):
        print('Calculating the area of each slice...')
        Area = lambda Vol: (Vol != 0).sum(axis=1).sum(axis=1) * (self.voxelSize/10)**2  # cm
        trunkArea, muscleArea, fasciclesArea = Area(trunkVol), Area(muscleVol), Area(fasciclesVol)
        return trunkArea, muscleArea, fasciclesArea
        
    def run(self):
        trunkVol, muscleVol, fasciclesVol = self.processVolumes()
        trunkArea, muscleArea, fasciclesArea = self.calculateAreas(trunkVol, muscleVol, fasciclesVol)
        return trunkArea, muscleArea, fasciclesArea
        
    def plot(self, trunkArea, muscleArea, fasciclesArea):
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

        
#TA = TrunkAnalysis()
#trunkArea, muscleArea, fasciclesArea = TA.run()




