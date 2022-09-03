
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import io, morphology, measure


class LabelFieldOperations:

    def __init__(self, labelVol):
        self.labelVol = labelVol

    def MaskLabels(self, mask):
        print('masking incomplete fascicles...')
        outerVol = np.where(mask == False, labelVol, ~mask)
        unique = np.unique(outerVol)
        wholemask = np.where(np.isin(labelVol, unique), False, True)
        wholeVol = np.where(wholemask, labelVol, np.zeros(labelVol.shape))
        return wholeVol.astype(np.uint16)


tipShape = (1075, 2150, 2150)

labelVol = np.fromfile(r'D:\Luke\current\muscles_p1_postprocessing-files\muscles_p1-combined.postprocessed.raw', dtype=np.uint16)
labelVol = np.reshape(labelVol, tipShape)
print('loading fascicles:', labelVol.shape)

musclewholeVol = np.fromfile(r'D:\Luke\current\tip_muscle_volumes-files\tip-muscle-whole.raw', dtype=np.uint8)
musclewholeVol = np.reshape(musclewholeVol, tipShape)
print('loading whole muscle mask:', musclewholeVol.shape)

muscleregionsVol = np.fromfile(r'D:\Luke\current\tip_muscle_volumes-files\tip-muscle-regions.raw', dtype=np.uint8)
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
