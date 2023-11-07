"""
Created on Tue May 3 16:58:58 2022

@author: Giorgia Tortora and Andrea Bassi @Polimi

Napari widget for the registration of regions of interests (ROI) in a time lapse acquistion 
and processing of the intensity of the registered data.

The widget uses user-defined labels, constructs a rectangular ROI around the label 
and registers the ROI in each time frame.

Based on opencv registration functions.

Registration of multiple ROIs is supported.  

"""

from napari_roi_registration.registration_utils import plot_data, save_in_excel, normalize_stack, select_rois_from_stack, select_rois_from_image
from napari_roi_registration.registration_utils import align_with_registration, update_position, resize_stack,rescale_position, filter_images
from napari_roi_registration.registration_utils import calculate_spectrum, correct_decay
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import FunctionGui
from napari import Viewer
from napari.layers import Image, Points, Labels
import pathlib
import os
from napari.qt.threading import thread_worker
import warnings
from skimage.measure import regionprops
import time

def max_projection(label_layer):
    '''
    Compresses a 3D label layer into a 2D array and returns the values.
    Selects the label with the highest value in case of overlap.
    '''
    values = np.asarray(label_layer.data).astype(int)
    if values.ndim == 3:
        values = np.max(values, axis = 0)
    elif values.ndim == 4:
        values = np.max(values, axis = (0,1))
    return(values)
    

def get_labels_values(labels_data):
    labels_props = regionprops(label_image=labels_data) # intensity_image=image0)    
    label_values = []
    for prop in labels_props:
        label_values.append(prop['label'])
    return label_values


def get_labels_color(labels_data):
    colors_values = get_labels_values(labels_data)
    labels_layer = Labels(labels_data) # TODO use an active label on viewer
    colors = labels_layer.get_color(colors_values)
    return colors
    

@magic_factory(call_button="Subtract background")
def subtract_background(viewer: Viewer, image: Image, 
                        labels: Labels):
    '''
    Subtracts a background from each plane of a stack image
    background is calulated as the mean intensity over one or more layers
    
    Parameters
    ----------
    image: napari.layers.Image
        Select the image to subtract the background from.
    labels: napari.layers.Labels
        Use the labels to select the area from which to get the background.
    '''
    
    result_name = image.name + '_corrected'
    
    def update_image(new_image):
        warnings.filterwarnings('ignore')
        try: 
            # if the layer exists, update the data
            viewer.layers[result_name].data = new_image
        except KeyError:
            # otherwise add it to the viewer
            viewer.add_image(new_image, name=result_name)
            
      
    @thread_worker(connect={'yielded': update_image})
    def _subtract_background():
        
        try:
            subtract_background.enabled = False
            warnings.filterwarnings('ignore')
            AMAX=2**16-1
            #check_if_suitable_layer(image, labels, roitype='registration')
            original = np.asarray(image.data, dtype = int)
            labels_data = max_projection(labels)
            mask = np.asarray(labels_data>0).astype(int)
            corrected = np.zeros_like(original)
            if original.ndim == 4:
                [st, sc, sy, sx] = original.shape
                for plane_idx in range(st):
                    for channel_idx in range(sc):
                        plane = original[plane_idx, channel_idx, :, :]
                        print(f'Correcting background on frame {plane_idx}')
                        props = regionprops(label_image=mask,
                                            intensity_image=plane)
                        background = props[0]['mean_intensity'] #only one region is defined in mask
                        diff = np.clip(plane-background, a_min=0, a_max=AMAX).astype('uint16')
                        corrected[plane_idx, channel_idx, :,:] = diff
                        yield corrected
            elif original.ndim == 3:
                [st, sy, sx] = original.shape
                for plane_idx in range(st):
                    plane = original[plane_idx, :, :]
                    print(f'Correcting background on frame {plane_idx}')
                    props = regionprops(label_image=mask,
                                        intensity_image=plane)
                    background = props[0]['mean_intensity'] #only one region is defined in mask
                    diff = np.clip(plane-background, a_min=0, a_max=AMAX).astype('uint16')
                    corrected[plane_idx, :,:] = diff
                    yield corrected
                
        finally:
            subtract_background.enabled = True
             
    _subtract_background()
    
def get_rois_props(label_data, t=0, c=0, bbox_zoom = 1):
    centroids = []  
    roi_sizes_x = []
    roi_sizes_y = []
    props = regionprops(label_image=label_data)#, intensity_image=image0)
    for prop in props:
        yx = prop['centroid']
        bbox = prop['bbox']
        _sizey = int(np.abs(bbox[0]-bbox[2])*bbox_zoom)
        _sizex = int(np.abs(bbox[1]-bbox[3])*bbox_zoom)  
        centroids.append([t, c, yx[0], yx[1]])
        roi_sizes_y.append(_sizey)
        roi_sizes_x.append(_sizex)
    return centroids, roi_sizes_y, roi_sizes_x   

def create_point(viewer: Viewer, center, name):
    try:
        viewer.layers[name].add(np.array(center))
    except:
        viewer.add_points(np.array(center),
                          edge_color='green',
                          face_color=[1,1,1,0],
                          name = name
                          )

def create_rectangles(centers, sys, sxs):
    rectangles = []
    for center,sy,sx in zip(centers,sys,sxs):
        cz=center[0]
        cc=center[1]
        cy=center[2]
        cx=center[3]
        hsx = sx//2
        hsy = sy//2
        rectangle = [ [cz, cc, cy+hsy, cx-hsx], # up-left
                      [cz, cc, cy+hsy, cx+hsx], # up-right
                      [cz, cc, cy-hsy, cx+hsx], # down-right
                      [cz, cc, cy-hsy, cx-hsx]  # down-left
                      ]
        rectangles.append(rectangle)
    return np.array(rectangles)



def init_the_widget(this_widget: FunctionGui):
    @this_widget.image.changed.connect
    def _on_image_changed(image: Image):
        """
        Function to be executed on image change on the registration or processing widget. 
        Sets the registration channel maximum value to the number of channels available in the selected image layer 
        """
        data = image.data 
        stack_dim = data.ndim
        if stack_dim == 3:
            this_widget.selected_channel.value = 0
            this_widget.selected_channel.visible = False
        elif stack_dim == 4:
            this_widget.selected_channel.visible = True
            this_widget.selected_channel.max = data.shape[1]-1
            
    
@magic_factory(call_button="Register ROIs",
           widget_init=init_the_widget,
           selected_channel={'visible':False},
           mode={"choices":['Translation','Affine','Euclidean','Homography']})
def register_rois(viewer: Viewer, image: Image,
                    labels_layer: Labels,
                    selected_channel:int = 0,
                    mode: str = 'Translation',
                    median_filter_size:int = 3,
                    scale = 0.5,
                    bbox_zoom = 1,
                    register_entire_image:bool = False,
                    show_registered_stack:bool = False
                    ):
    '''
    Registers rectangular rois chosen on image as the bound box of the labels.
    Based on cv2 registration.

    Parameters
    ----------
    image: napari.layers.Image
        The image stack (time,channel,y,x) to be registered.
    labels_layer: Labels
        The labels will define the rectangular ROIs to be registered.
    selected_channel:int
        The channel on which registration will be performed.
    mode: str
        The type of cv2 registration.
    median_filter_size:int
        Size of the median filter applied before registration.
    scale: float
        Rescaling factor of the image for the registation
        Does not affect the registered image scale.
    bbox_zoom: int
        The bounding boxes enclosing the ROIs are zoomed by this factor.
    register_entire_image:bool
        If True, the entire image is registered around the bounding box.
    show_registered_stack:bool
        If True,  shows the registered stacks as new image layers.
    '''
    
    if labels_layer is None or image is None:
        return
    
    print('Starting registration...')
    # remove registration points if present
    labels = max_projection(labels_layer)
    label_colors = get_labels_color(labels)
    # labels = max_projection(labels_layer)
    initial_time_index = viewer.dims.current_step[0]
    widgets_shared_variables.initial_time_index = initial_time_index
    print('... with initial time index:', widgets_shared_variables.initial_time_index)
    original_stack = np.asarray(image.data)
    stack_dim = original_stack.ndim
    
    if stack_dim == 3:
        stack = original_stack
        time_frames_num, sy, sx = stack.shape
    elif stack_dim == 4:
        time_frames_num,ch_num, sy, sx = original_stack.shape
        stack = original_stack [:, selected_channel,:,:]
    print('... in channel:', selected_channel)
    real_initial_positions, real_roi_sy, real_roi_sx = get_rois_props(labels, 
                                                                        initial_time_index,
                                                                        selected_channel,
                                                                        bbox_zoom)
    roi_num = len(real_initial_positions)
    
    def add_rois(params): 
        
        import numpy.matlib
        rectangles = params[0]
        _centers = params[1]
        if stack_dim == 3:
            rectangles = np.delete(rectangles,1,axis=3)
            _centers = np.delete(_centers,1,axis=2)
            rectangles = rectangles.reshape((roi_num*time_frames_num,4,3))
            centers = _centers.reshape((roi_num*time_frames_num,3))
        elif stack_dim == 4:
            rectangles = rectangles.reshape((roi_num*time_frames_num,4,4))
            centers = _centers.reshape((roi_num*time_frames_num,4))
        color_array= numpy.matlib.repmat(label_colors,len(rectangles)//roi_num,1)
        points_layer_name = f'centroids_{image.name}_{selected_channel}'
        rectangles_name = f'rectangles_{image.name}_{selected_channel}'
        if points_layer_name in viewer.layers:
            viewer.layers.remove(points_layer_name)
        if rectangles_name in viewer.layers:
            viewer.layers.remove(rectangles_name)
        shapes = viewer.add_shapes(np.array(rectangles[0]),
                            edge_width=3,
                            edge_color=color_array[0],
                            face_color=[1,1,1,0],
                            name = rectangles_name
                            )
        shapes.add_rectangles(np.array(rectangles[1:]),
                                edge_color=color_array[1:])
        viewer.add_points(np.array(centers),
                                edge_color='green',
                                face_color=[1,1,1,0],
                                name = points_layer_name
                                )
        if show_registered_stack: 
            for roi_idx in range(roi_num):
                pos = _centers[:,roi_idx,:]
                if stack_dim == 3:
                    stack = original_stack
                    y = pos[initial_time_index,1]
                    x = pos[initial_time_index,2]
                    no_channel_pos = pos
                elif stack_dim == 4:
                    stack = original_stack[:, selected_channel, :, :]
                    y = pos[initial_time_index,2]
                    x = pos[initial_time_index,3]
                    no_channel_pos = np.delete(pos,1,axis=1)
                sizey = real_roi_sy[roi_idx]
                sizex = real_roi_sx[roi_idx]
                if register_entire_image: #TODO change the function select_rois_from_stack to effectively get registration of entire image
                    sizex = min(sx-x,x)*2 
                    sizey = min(sy-y,y)*2
                if stack_dim == 3:
                    registered_image = select_rois_from_stack(original_stack, no_channel_pos, 
                                                            [int(sizey)], [int(sizex)])
                    registered_data = np.asarray(registered_image)
                elif stack_dim == 4:
                    registered_images = []
                    for c_idx in range(ch_num):
                        registered_image = select_rois_from_stack(original_stack[:,c_idx,:,:], no_channel_pos, 
                                                                [int(sizey)], [int(sizex)])
                        registered_images.append(registered_image)
                    registered_data = np.swapaxes(np.asarray(registered_images),0,1)
                registered_roi_name= f'registered_{image.name}_roi{roi_idx}'
                if registered_roi_name in viewer.layers:
                        viewer.layers.remove(registered_roi_name)
                viewer.add_image(registered_data, name= registered_roi_name)
        
        print('... ending registration.')
        
    @thread_worker(connect={'returned':add_rois})
    def _register_rois():    
        warnings.filterwarnings('ignore')        
        resized = resize_stack(stack, scale)
        resized, _vmin, _vmax = normalize_stack(resized)
        image0 = resized[initial_time_index,...]
        
        initial_positions = rescale_position(real_initial_positions,scale)
        roi_sy = [int(ri*scale) for ri in real_roi_sy]
        roi_sx = [int(ri*scale) for ri in real_roi_sx]
        previous_rois = select_rois_from_image(image0, initial_positions, roi_sy,roi_sx)
        previous_rois = filter_images(previous_rois, median_filter_size)

        rectangles = np.zeros([time_frames_num,roi_num,4,4]) 
        centers = np.zeros([time_frames_num,roi_num,4])
            # register forwards
        next_positions = initial_positions.copy()
        real_next_positions = rescale_position(next_positions,1/scale)  
        centers[initial_time_index, :, :] = np.array(real_next_positions)
        rectangles[initial_time_index, :, :, :] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
        for t_index in range(initial_time_index+1, time_frames_num, 1):
            next_rois = select_rois_from_image(resized[t_index, ...], next_positions, roi_sy, roi_sx)
            next_rois = filter_images(next_rois, median_filter_size)
            dx, dy, _wm = align_with_registration(next_rois, previous_rois, mode)
            next_positions = update_position(next_positions, dz=1, dx_list=dx, dy_list=dy)
            real_next_positions = rescale_position(next_positions, 1/scale)
            centers[t_index,:,:] = np.array(real_next_positions)
            rectangles[t_index,:,:,:] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
        # register backwards
        next_positions = initial_positions.copy()
        for t_index in range(initial_time_index-1, -1, -1):
            next_rois = select_rois_from_image(resized[t_index, ...], next_positions, roi_sy, roi_sx)
            next_rois = filter_images(next_rois, median_filter_size)
            dx, dy, _wm = align_with_registration(next_rois, previous_rois, mode)
            next_positions = update_position(next_positions, dz=-1, dx_list=dx, dy_list=dy)
            real_next_positions = rescale_position(next_positions, 1/scale)
            centers[t_index,:,:] = np.array(real_next_positions)
            rectangles[t_index,:,:,:] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
        return (rectangles, centers)
    _register_rois()
    
def calculate_intensity(image:Image,
                        roi_num:int,
                        points_layer:Points,
                        labels_layer:Labels,
                        channel
                        ):
    """
    Calculates the mean intensity,
    within rectangular Rois of size roi_size, centered in points_layer,
    taking into account only the pixels that are in one of the labels of labels_layer
    """
    initial_time_index = widgets_shared_variables.initial_time_index
    labels_data = max_projection(labels_layer)
    label_values = get_labels_values(labels_data)
    original_stack = np.array(image.data)
    locations = points_layer.data
    
    if original_stack.ndim == 3:
        stack = original_stack
        st, sy, sx = stack.shape
        new_locations = np.insert(locations,1,channel,axis=1)
        no_channel_locations = locations
    elif original_stack.ndim == 4:
        st, sc, sy, sx = original_stack.shape
        stack = original_stack[:, channel, :, :]
        no_channel_locations = np.delete(locations,1,axis=1)
        new_locations = locations

    _ , roi_sizey, roi_sizex = get_rois_props(labels_data)   
    rois = select_rois_from_stack(stack, no_channel_locations, roi_sizey, roi_sizex)
    initial_location = initial_time_index*roi_num
    label_rois = select_rois_from_image(labels_data,
                                        new_locations[initial_location:initial_location+roi_num],
                                        roi_sizey, roi_sizex)
    intensities = np.zeros([st, roi_num])
    for time_idx in range(st):
        for roi_idx in range(roi_num):
            
            label_value = label_values[roi_idx]
            mask_indexes = label_rois[roi_idx] == label_value 
            global_idx = roi_idx + time_idx*roi_num
            roi = rois[global_idx]
            selected = roi[mask_indexes]
            intensity = np.mean(selected)
            intensities[time_idx, roi_idx] = intensity

    return intensities


def measure_displacement(image, roi_num, points, channel):
    """
    Measure the displacement of each roi:
    dr: relative to its position in the previous time frame 
    deltar: relative to the initial position.
    """
    original_stack = image.data
    locations = points.data
    
    if original_stack.ndim == 3:
        stack = original_stack
        st, sy, sx = stack.shape
        new_locations = locations
    elif original_stack.ndim == 4:
        st, ch_num, sy, sx = original_stack.shape
        stack = original_stack[:, channel, :, :]
        new_locations = np.delete(locations,1,axis=1)
        
    reshaped = new_locations.reshape((st, roi_num, stack.ndim))
    xy = reshaped[...,1:] # remove the time index value
    xy0 = xy[0,...] # take the x,y cohordinates of the rois in the first time frame
    
    deltar = np.sqrt( np.sum( (xy-xy0)**2, axis=2) )
    rolled = np.roll(xy, 1, axis=0) #roll one roi    
    rolled[0,...] = xy0
    dxy = xy-rolled
    dr = np.sqrt( np.sum( (dxy)**2, axis=2) )
    return xy, deltar, dxy, dr


@magic_factory(call_button="Process registered ROIs",
               widget_init=init_the_widget,
               selected_channel={'visible':False},
               saving_folder={'mode': 'd'})
def process_rois(viewer: Viewer, 
                 image: Image,
                 registered_points: Points,
                 labels_layer: Labels,
                 selected_channel:int = 0,
                 correct_photobleaching: bool = False,
                 plot_results:bool = True,
                 save_results:bool = False,
                 saving_folder: pathlib.Path = os.getcwd(),
                 saving_filename:str = 'temp'
                 ):
    
    '''
    Parameters
    ----------
    image: napari.layers.Image
        The image to take into account during processing.
        It is normally a previously registered image. 
    registered_points: napari.layers.Points
        The center of the ROIs to take into account during processing
    labels_layer: napari.layers.Labels
        The labels previously used for registration. The intensity is calculated on the labels only. 
    selected_channel:int
        The channel to be considered for processing of data.
    correct_photobleaching: bool
        If True photobleaching correction is appleid to the the intensities.
    plot_results:bool
        If True, shows the plots of the collected data with matplotlib in the console.
    save_results:bool
        If True, creates an excel file with processing results.
    saving foldere: str
        Folder to save the file to.
    saving_filename:
        Name of the file to save.
    '''
    
    if registered_points is None or labels_layer is None or image is None:
        return
    warnings.filterwarnings('ignore')
    print('Starting processing ...')
    print('... in channel:', selected_channel)
    try:
        process_rois.enabled = False
        original_stack = image.data
        
        if original_stack.ndim == 3:
            stack = original_stack
            time_frames_num, sy, sx = stack.shape
        elif original_stack.ndim == 4:
            time_frames_num, ch_num, sy, sx = original_stack.shape
            stack = original_stack[:, selected_channel, :, :]
            
        locations = registered_points.data
        roi_num = len(locations) // time_frames_num
        intensities = calculate_intensity(image, roi_num, 
                                          registered_points,
                                          labels_layer, selected_channel)
        yx, deltar, dyx, dr = measure_displacement(image, roi_num, registered_points, selected_channel)
        
        if correct_photobleaching:
            intensities = correct_decay(intensities)
        spectra = calculate_spectrum(intensities)    
            
        if plot_results:
            label_values = max_projection(labels_layer)
            colors = get_labels_color(label_values)
            plot_data(deltar, colors, "time index", "displacement (px)")
            plot_data(intensities, colors, "time index", "mean intensity")
        
        #directory, filename = os.path.split(path)
        #newpath = directory +'\\'+image.name
        if save_results:
            filename = os.path.join(saving_folder,saving_filename +'.xlsx')
            save_in_excel(filename,
                          sheet_name = 'Roi',
                          x = yx[...,1], 
                          y = yx[...,0],
                          length = deltar,
                          dx = dyx[...,1],
                          dy = dyx[...,0],
                          dr = dr,
                          intensity = intensities,
                          spectra = spectra
                          )
    except Exception as e:
        raise(e)
    finally: 
        process_rois.enabled = True
    print(f'... processed {time_frames_num} frames.')


from dataclasses import dataclass
@dataclass
class Shared_variables:
    initial_time_index: int = 0

widgets_shared_variables = Shared_variables(initial_time_index=0)  


if __name__ == '__main__':

    import napari
    viewer = napari.Viewer()
    background_widget = subtract_background()
    register_widget = register_rois()
    processing_widget = process_rois()
    viewer.window.add_dock_widget(background_widget, name = 'Background Subtraction',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(register_widget, name = 'ROIs Registration',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(processing_widget, name = 'Processing',
                                  area='right')
    warnings.filterwarnings('ignore')
    napari.run() 