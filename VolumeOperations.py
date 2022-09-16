
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import io, morphology, measure
import pandas as pd


class LabelFieldOperations:

    def __init__(self, labelVol):
        self.labelVol = labelVol

    def MaskLabels(self, mask):
        print('masking incomplete fascicles...')
        outerVol = np.where(mask == False, self.labelVol, ~mask)
        unique = np.unique(outerVol)
        wholemask = np.where(np.isin(self.labelVol, unique), False, True)
        wholeVol = np.where(wholemask, self.labelVol, np.zeros(self.labelVol.shape))
        return wholeVol.astype(np.uint16)

    def AvgLabels(self, image):
        print('averaging values in image by label')
        labels = np.unique(self.labelVol)
        values = np.zeros(len(labels))
        for i in labels:
            print(i)
            imageMask = np.where(image == labels[i], image, np.zeros(image.shape))
            values[i] = np.sum(imageMask)
        return values


tipShape = (1075, 2150, 2150)


def runMaskLabels():
    labelVol = np.fromfile(r'D:\Luke\current\IMPORTANT\muscles_p1-combined.postprocessed.raw', dtype=np.uint16)
    labelVol = np.reshape(labelVol, tipShape)
    print('loading fascicles:', labelVol.shape)

    musclewholeVol = np.fromfile(r'D:\Luke\current\IMPORTANT\tip-muscle-whole.raw', dtype=np.uint8)
    musclewholeVol = np.reshape(musclewholeVol, tipShape)
    print('loading whole muscle mask:', musclewholeVol.shape)

    muscleregionsVol = np.fromfile(r'D:\Luke\current\IMPORTANT\tip-muscle-regions.raw', dtype=np.uint8)
    muscleregionsVol = np.reshape(muscleregionsVol, tipShape)
    print('loading muscle region masks:', muscleregionsVol.shape)

    LFO = LabelFieldOperations(labelVol)
    completetipVol = LFO.MaskLabels(mask=np.where(musclewholeVol, True, False))
    print('output complete fascicles in whole tip:', np.shape(completetipVol))
    dorsaltipVol = LFO.MaskLabels(mask=np.where(muscleregionsVol == 1, True, False))
    print('output complete fascicles in dorsal tip:', np.shape(dorsaltipVol))
    ventraltipVol = LFO.MaskLabels(mask=np.where(muscleregionsVol == 2, True, False))
    print('output complete fascicles in ventral tip:', np.shape(ventraltipVol))

    print('saving output...')
    workingDir = 'D:\Luke\ElephantTrunkMuscles'
    io.imsave(workingDir + '\VolumeOperations_output\muscles_p1-complete.tif', completetipVol, check_contrast=False)
    io.imsave(workingDir + '\VolumeOperations_output\muscles_p1-dorsal.tif', dorsaltipVol, check_contrast=False)
    io.imsave(workingDir + '\VolumeOperations_output\muscles_p1-ventral.tif', ventraltipVol, check_contrast=False)
runMaskLabels()


def runAvgLabels():
    dorsalLabelVol = io.imread(r'D:\Luke\ElephantTrunkMuscles\VolumeOperations_output\muscles_p1-dorsal.tif')
    print(dorsalLabelVol.shape)

    dorsalPropDistVol = np.fromfile(r'D:\Luke\current\IMPORTANT\test\tip-dorsal_PropagationDistance.raw', dtype=np.uint16)
    dorsalPropDistVol = np.reshape(dorsalPropDistVol, tipShape)
    print(dorsalPropDistVol.shape)

    LFO = LabelFieldOperations(dorsalLabelVol)
    dorsalPropDist = LFO.AvgLabels(dorsalPropDistVol)

    df = pd.DataFrame()
    df['dorsalPropDist'] = dorsalPropDist
    df.to_csv('D:\Luke\current\IMPORTANT\test\AvgLabels.csv')
#runAvgLabels()