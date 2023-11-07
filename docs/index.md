# Welcome to napari-roi-registration

napari-roi-registration is a plugin for simultaneous registration of multiple regions of interest (ROIs) in time lapse acquisition datasets (multiframe datasets) and for the subsequent processing of the registered data. The plugin works on single channel datasets as well as multichannel datasets.
The plugin consists of three widgets: the Background Widget, the Registration Widget and the Processing Widget. 

## Background Widget

The Background Widget allows a pre-processing of data, and it is typically useful when ambient light affects the measurements. It enables the removal of background on each frame of the time-lapse dataset. The background is calculated as the mean intensity of the pixels under a chosen area of one frame of the stack. The area of the frame is chosen drawing a label in a labels layer. 

### List of parameters

**image**: The image stack (time, channel, y, x) to be corrected.  
**labels**: The labels layer where the user drew the label which will be used to calculate the background.

### How to use the Background Widget

1. Open the image you want to correct in the napari viewer. If you open more than one image in the viewer, select the image you want to correct in the image menu of the Background Widget. 

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/before_sub.png)

2. Create a new labels layer and draw a label on one frame of your stack on the area where you want to calculate the background. The background will be calculated as the mean intensity of the pixels under the drawn label.

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/background.png)

3. Press the **Subtract background** button. The mean intensity value will be subtracted from all frames of the dataset. If you are dealing with multichannel data, a different background value is calculated for each channel considering the pixels of the image under the label in each channel. 
A new image layer will appear in the viewer. This layer contains the stack of corrected images.

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/after_sub.png)

## Registration Widget

The Registration Widget executes registration of a non-limited number of user-defined regions of interest (ROIs). Using a labels layer, the user can draw several labels on the regions of the image which he wants to register. The widget constructs a rectangular ROI as the bounding box of the label around each of the labels and register the ROIs in all time frames. It is not mandatory to draw all the labels on one frame of the stack; it is possible to draw labels on different frames. 

### List of parameters

**image**: The image stack (time, channel, y, x) to be registered.  
**labels layer**: The labels layer where the user drew the labels that will define the rectangular ROIs to be registered.  
**selected channel**: The channel on which registration will be performed. It is not possible to perform registration simultaneously on all channels (as it will be explained later, it is possible to perform registration just on one channel and then process data in all channels using the same registration results).  
**mode**: The type of cv2 registration.  
**median filter size**: Size of the median filter applied to images before registration.  
**scale**: Rescaling factor of the image for the registration. Does not affect the registered image scale.  
**bbox zoom**: The bounding boxes enclosing the ROIs are zoomed by this factor. Normally the bounding box would be as large as the label but, if a bbox zoom greater than 1 is chosen, the portion of image to be registered will be bigger. For small labels, it is warmly suggested to choose a bbox zoom greater than 1 to get a more precise registration.  
**register entire image**: If True, the entire image is registered around the bounding box.  
**show registered stack**: If True, shows the registered stacks as new image layers. In the new layers, the images have been aligned to respect to the portion of the image under the label.  

### How to use the Registration Widget

1. Open the image you want to register in Napari. If you have already corrected your image using the background widget, your image will already be in the napari viewer under the name <original name of the image>_ corrected.  
If you open more than one image, select the image on which you want to perform the registration in the image menu of the Registration Widget.

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/before_reg.png)

2. Create a new labels layer and draw labels where you want to select a region of interest. Be careful with the colours of the labels. Each colour represents a different label which will correspond to a different ROI. If you use the same colour to label two different parts of an image, they will be considered as a single region of interest and will be registered in the same bounding box.  

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/double_label.png)

Moreover, labels are not supposed to overlap. If two labels are overlapping, only the colour that corresponds to the highest number will be considered in the overlapped area.

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/labels_numbers.png)

The registration process starts from the currently selected frame, which is the frame visualized in the viewer when the **Register ROIs** button is pressed, no matter in which frame have been drawn the labels. The result of the registration process can change according to the starting frame. So, if you are not satisfied with the obtained registration, try to change the starting frame simply changing the frame that you are visualizing.  
If **show registered stack** is selected, at the end of the registration process a new image layer containing the registered ROI stack will be created for each label.  
If **register entire image** is selected, the entire image will be registered, not only the portion of image in the bounding box.

3. Press the **Register ROIs** button: registration will be performed. When the registration process ends, two new layers will appear in the viewer. One layer contains the centroids of the drawn labels while the other contains the bounding boxes enclosing the ROIs.

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/after_reg.png)

## Processing Widget

The Processing Widget measures ROIs displacements and extracts the average intensity of the ROIs in each frame.

### List of parameters

**image**: The image to consider during processing. It is normally the previously registered image.  
**registered points**: The centroids of the ROIs to consider during processing. Obtained from the registration step. Displacement of the ROIs will be measured considering the coordinates of the centroids.  
**labels layer**: The labels previously used for registration. The intensity is calculated on the labels only.  
**selected channel**: The channel to be considered for processing of data. This channel may be different from the registration channel. Information related to intensity will be extracted considering images on the chosen processing channel even if registration has been performed on another channel. Rectangles and centroids obtained from registration will be used. This means that if you have one channel that is better than the others, you can perform registration on that channel only and then perform processing on every channel.  
**correct photobleaching**: If True photobleaching correction is applied to the intensities.  
**plot results**: If True, shows the plots of the collected data with matplotlib in the console.  
**save results**: If True, creates an excel file with processing results.  
**saving folder**: Folder to save the file to.  
**saving filename**: Name of the file to save.  

### How to use the Processing Widget

1. Pressing the **Process registered ROIs** button, the registered ROIs will be analysed. The intensity and the displacement of the registered ROIs will be calculated. Intensity is calculated considering only the pixels which are inside the bounding box and under the label. Pixels which are inside the bounding box but aren’t under the label will not contribute to calculation of intensity.   
If **plot results** is selected, plots of displacement vs time index and mean intensity vs time index will appear in the console.  
Choosing the **save results** option, an excel file containing ROIs positions, displacements, and intensities will be saved. 

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/blob/main/images/plots.png)



