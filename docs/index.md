# Welcome to napari-roi-registration

napari-roi-registration is a plugin for simultaneous registration of multiple regions of interest (ROIs) in time lapse acquisition datasets (multiframe datasets) and for the subsequent processing of the registered data. The plugin works on single channel datasets as well as multichannel datasets.
The plugin consists of three widgets: the Background Widget, the Registration Widget and the Processing Widget. 

## Background Widget

The Background Widget allows a pre-processing of data, and it is typically useful when ambient light affects the measurements. It enables the removal of background on each frame of the time-lapse dataset. The background is calculated as the mean intensity of the pixels under a chosen area of one frame of the stack. The area of the frame is chosen drawing a label in a labels layer. 

### List of parameters

image: The image stack (time, channel, y, x) to be corrected.
labels: The labels layer where the user drew the label which will be used to calculate the background.

### How to use the Background Widget

1. Open the image you want to correct in the napari viewer. If you open more than one image in the viewer, select the image you want to correct in the image menu of the Background Widget. 