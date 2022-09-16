
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from PIL import Image
from skimage import io, morphology, measure
import skan
import pandas as pd
from scipy.spatial import distance


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

def combineMuscleClass(subfolder='muscles_p1', region='ventral'):
    print('mapping predicted class values to each muscle in partition {}'.format(subfolder))
    labelVol = io.imread(workingDir + '/VolumeOperations_output/muscles_p1-{}.tif'.format(region))
    labels = np.unique(labelVol)
    classes0 = pd.read_csv(workingDir + '/analysis_output/{}-classes.csv'.format(subfolder))['MuscleClass']
    classes = np.append([0], np.array(classes0) + 1)
    from skimage.util._map_array import map_array
    outputVol = map_array(labelVol, labels, classes).astype(np.uint8)
    io.imsave(workingDir + '/analysis_output/{}-classes.tif'.format(subfolder), outputVol, check_contrast=False)


combineMuscleClass()


# ----------


def processOrientation():
    print('processing orientation properties of individual fascicles')
    df = pd.read_csv(workingDir + '/analysis_output/muscles_p1-final.Label-Analysis---orientation.csv', skiprows=1)
    df = df[df['Volume3d'] != 0]  # remove indices with zero volume
    # volume shape info
    physical_size = np.array((38.6818, 38.6818, 19.3319))
    physical_from = np.array((7.09196, 12.3839, 29.7178))
    shape = (2150, 2150, 1075)
    voxelSize = 0.0179999  # mm
    # convert df to units of voxel
    factor = physical_size // voxelSize
    df_voxel = df[['BaryCenterX', 'BaryCenterY', 'BaryCenterZ',
                   'Length3dInputX', 'Length3dInputY', 'Length3dInputZ',
                   'Length3dOutputX', 'Length3dOutputY', 'Length3dOutputZ']]
    df_voxel.loc[:, ('BaryCenterX', 'Length3dInputX', 'Length3dOutputX')] -= physical_from[0]
    df_voxel.loc[:, ('BaryCenterY', 'Length3dInputY', 'Length3dOutputY')] -= physical_from[1]
    df_voxel.loc[:, ('BaryCenterZ', 'Length3dInputZ', 'Length3dOutputZ')] -= physical_from[2]
    df_voxel.loc[:, :] //= voxelSize
    df_voxel.to_csv(workingDir + '/analysis_output/muscles_p1-orientation.csv')

    def returnOrientationVol():
        # place points at input (1), barycenter (2), and output (3)
        idxs_input = np.transpose([df_voxel['Length3dInputX'], df_voxel['Length3dInputY'], df_voxel['Length3dInputZ']]).astype(int)
        idxs_center = np.transpose([df_voxel['BaryCenterX'], df_voxel['BaryCenterY'], df_voxel['BaryCenterZ']]).astype(int)
        idxs_output = np.transpose([df_voxel['Length3dOutputX'], df_voxel['Length3dOutputY'], df_voxel['Length3dOutputZ']]).astype(int)
        orientationVol = np.zeros(shape, dtype=np.uint8)
        for i in range(len(df_voxel)):
            orientationVol[tuple(idxs_input[i])] = 1
            orientationVol[tuple(idxs_center[i])] = 2
            orientationVol[tuple(idxs_output[i])] = 3
        orientationVol = np.swapaxes(orientationVol, 0, 2)
        io.imsave(workingDir + '/analysis_output/muscles_p1-orientation.tif', orientationVol, check_contrast=False)
    #returnOrientationVol()

    def returnInsertionDistances():
        # distance from medial plane
        planeVol = np.fromfile(r'D:\Luke\current\tip_muscle_volumes-files\tip-medial-plane.distmap.raw', dtype=np.uint16)
        planeVol = np.reshape(planeVol, shape)
        # obtain all points on the plane
        planePoints = np.transpose(np.where(planeVol))
        # determine smallest distance for each point
        input_points = np.array([df_voxel['Length3dInputX'], df_voxel['Length3dInputY'], df_voxel['Length3dInputZ']])
        barycenter_points = np.array([df_voxel['BaryCenterX'], df_voxel['BaryCenterY'], df_voxel['BaryCenterZ']])
        output_points = np.array([df_voxel['Length3dOutputX'], df_voxel['Length3dOutputY'], df_voxel['Length3dOutputZ']])
        input2planeDistance, input2planePoint = np.zeros(len(df_voxel)), np.zeros((len(df_voxel), 3), dtype=tuple)
        barycenter2planeDistance, barycenter2planePoint = np.zeros(len(df_voxel)), np.zeros((len(df_voxel), 3), dtype=tuple)
        output2planeDistance, output2planePoint = np.zeros(len(df_voxel)), np.zeros((len(df_voxel), 3), dtype=tuple)
        tip2barycenterDistance = np.zeros(len(df_voxel))
        for i in range(len(df_voxel)):
            input_distances = np.linalg.norm(planePoints - input_points[:,i], axis=1)
            barycenter_distances = np.linalg.norm(planePoints - barycenter_points[:,i], axis=1)
            output_distances = np.linalg.norm(planePoints - output_points[:,i], axis=1)
            input2planeMin = np.argmin(input_distances)
            barycenter2planeMin = np.argmin(barycenter_distances)
            output2planeMin = np.argmin(output_distances)
            input2planeDistance[i], input2planePoint[i] = input_distances[input2planeMin], planePoints[input2planeMin]
            barycenter2planeDistance[i], barycenter2planePoint[i] = barycenter_distances[barycenter2planeMin], planePoints[barycenter2planeMin]
            output2planeDistance[i], output2planePoint[i] = output_distances[output2planeMin], planePoints[output2planeMin]
            tip2barycenterDistance[i] = planeVol[tuple(planePoints[barycenter2planeMin])]
        df_distance2plane = pd.DataFrame()
        df_distance2plane['input2planeDistance'] = input2planeDistance * voxelSize
        df_distance2plane['barycenter2planeDistance'] = barycenter2planeDistance * voxelSize
        df_distance2plane['output2planeDistance'] = output2planeDistance * voxelSize
        df_distance2plane['tip2barycenterDistance'] = tip2barycenterDistance * voxelSize
        input2planePointAll = input2planePoint * voxelSize
        barycenter2planePointAll = barycenter2planePoint * voxelSize
        output2planePointAll = output2planePoint * voxelSize
        df_distance2plane['input2planePointX'] = input2planePointAll[:,0]
        df_distance2plane['input2planePointY'] = input2planePointAll[:,1]
        df_distance2plane['input2planePointZ'] = input2planePointAll[:,2]
        df_distance2plane['barycenter2planePointX'] = barycenter2planePointAll[:,0]
        df_distance2plane['barycenter2planePointY'] = barycenter2planePointAll[:,1]
        df_distance2plane['barycenter2planePointZ'] = barycenter2planePointAll[:,2]
        df_distance2plane['output2planePointX'] = output2planePointAll[:,0]
        df_distance2plane['output2planePointY'] = output2planePointAll[:,1]
        df_distance2plane['output2planePointZ'] = output2planePointAll[:,2]
        df_distance2plane.to_csv(workingDir + '/analysis_output/muscles_p1-distance2plane.csv')
    #returnInsertionDistances()


#processOrientation()


# ----------


def processInsertionTip(region):
    df = pd.read_csv(r'D:\Luke\current\muscles_p1_insertion-{0}-files\muscles_p1-{0}.Label-Analysis.csv'.format(region), skiprows=1)
    df = df[df['Volume3d'] != 0]  # remove indices with zero volume
    # convert to voxel units
    physical_from = np.array((7.09196, 12.3839, 29.7178))
    voxelSize = 0.0179999  # mm
    df_voxel = df.copy()[['BaryCenterX', 'BaryCenterY', 'BaryCenterZ',
                          'Length3dInputX', 'Length3dInputY', 'Length3dInputZ',
                          'Length3dOutputX', 'Length3dOutputY', 'Length3dOutputZ']]
    df_voxel.loc[:, ('BaryCenterX', 'Length3dInputX', 'Length3dOutputX')] -= physical_from[0]
    df_voxel.loc[:, ('BaryCenterY', 'Length3dInputY', 'Length3dOutputY')] -= physical_from[1]
    df_voxel.loc[:, ('BaryCenterZ', 'Length3dInputZ', 'Length3dOutputZ')] -= physical_from[2]
    df_voxel.loc[:, :] //= voxelSize
    #
    planeVol = np.fromfile(r'D:\Luke\current\muscles_p1_insertion-{0}-files\tip-medial-plane_{0}.distances.raw'.format(region), dtype=np.uint16)
    planeVol = np.reshape(planeVol, volShape)
    planeVol = np.swapaxes(planeVol, 0, 2)
    planePoints = np.transpose(np.where(planeVol))
    # determine closest point from muscle points to medial plane
    inputPoints = np.transpose([df_voxel['Length3dInputX'], df_voxel['Length3dInputY'], df_voxel['Length3dInputZ']])
    centerPoints = np.transpose([df_voxel['BaryCenterX'], df_voxel['BaryCenterY'], df_voxel['BaryCenterZ']])
    outputPoints = np.transpose([df_voxel['Length3dOutputX'], df_voxel['Length3dOutputY'], df_voxel['Length3dOutputZ']])
    #
    def calcDistances(planePoints, musclePoints):
        PlaneDistsAll = distance.cdist(planePoints, musclePoints, 'euclidean')
        MinIdxs = np.argmin(PlaneDistsAll, axis=0)
        PlaneDists = np.array([PlaneDistsAll[jdx,idx] for idx,jdx in enumerate(MinIdxs)])
        # determine value on the plane, distance to tip
        TipPoints = np.array([tuple(planePoints[idx]) for idx in MinIdxs])
        TipDists = np.array([planeVol[tuple(idx)] for idx in TipPoints])
        return PlaneDists, TipPoints, TipDists
    #
    inputPlaneDists, inputTipPoints, inputTipDists = calcDistances(planePoints, inputPoints)
    centerPlaneDists, centerTipPoints, centerTipDists = calcDistances(planePoints, centerPoints)
    outputPlaneDists, outputTipPoints, outputTipDists = calcDistances(planePoints, outputPoints)
    #
    df_out = pd.DataFrame()
    df_out['Volume3d'] = df['Volume3d'] // voxelSize
    df_out['Length3d'] = df['Length3d'] // voxelSize
    df_out['inputPlaneDists'] = inputPlaneDists
    df_out['centerPlaneDists'] = centerPlaneDists
    df_out['outputPlaneDists'] = outputPlaneDists
    df_out['inputTipDists'] = inputTipDists
    df_out['centerTipDists'] = centerTipDists
    df_out['outputTipDists'] = outputTipDists
    df_out['intputTipPointX'] = inputTipPoints[:,0]
    df_out['intputTipPointY'] = inputTipPoints[:,1]
    df_out['intputTipPointZ'] = inputTipPoints[:,2]
    df_out['centerTipPointX'] = centerTipPoints[:,0]
    df_out['centerTipPointY'] = centerTipPoints[:,1]
    df_out['centerTipPointZ'] = centerTipPoints[:,2]
    df_out['outputTipPointX'] = outputTipPoints[:,0]
    df_out['outputTipPointY'] = outputTipPoints[:,1]
    df_out['outputTipPointZ'] = outputTipPoints[:,2]
    df_out.to_csv(workingDir + '/analysis_output/muscles_p1-InsertionTip_{0}.csv'.format(region))


#processInsertionTip(region='dorsal')
#processInsertionTip(region='ventral')


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




