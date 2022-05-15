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
from napari import Viewer
from napari.layers import Image, Points, Labels
import pathlib
import os
from napari.qt.threading import thread_worker
import warnings
from skimage.measure import regionprops

def max_projection(label_layer):
    '''
    Compresses a 3D label layer into a 2D array and returns the values.
    Selects the label with the highest value in case of overlap.
    '''
    values = np.asarray(label_layer.data).astype(int)
    if values.ndim>2:
        values = np.max(values, axis = 0)
        
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
            original = np.asarray(image.data)
            labels_data = max_projection(labels)
            mask = np.asarray(labels_data>0).astype(int)
            corrected = np.zeros_like(original)
             
            for plane_idx, plane in enumerate(original.astype('int')):
                print(f'Correcting background on frame {plane_idx}')
                props = regionprops(label_image=mask,
                                    intensity_image=plane)
                background = props[0]['mean_intensity'] #only one region is defined in mask
                diff = np.clip(plane-background, a_min=0, a_max=AMAX).astype('uint16')
                corrected[plane_idx,:,:] = diff
                yield corrected
        finally:
            subtract_background.enabled = True
             
    _subtract_background()
    
def get_rois_props(label_data, t=0, bbox_zoom = 1):
    centroids = []  
    roi_sizes_x = []
    roi_sizes_y = []
    props = regionprops(label_image=label_data)#, intensity_image=image0)
    for prop in props:
        yx = prop['centroid']
        bbox = prop['bbox']
        _sizey = int(np.abs(bbox[0]-bbox[2])*bbox_zoom)
        _sizex = int(np.abs(bbox[1]-bbox[3])*bbox_zoom)  
        centroids.append([t, yx[0], yx[1]])
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
        cy=center[1]
        cx=center[2]
        hsx = sx//2
        hsy = sy//2
        rectangle = [ [cz, cy+hsy, cx-hsx], # up-left
                      [cz, cy+hsy, cx+hsx], # up-right
                      [cz, cy-hsy, cx+hsx], # down-right
                      [cz, cy-hsy, cx-hsx]  # down-left
                      ]
        rectangles.append(rectangle)
    return np.array(rectangles)
        
    
@magic_factory(call_button="Register ROIs",
           mode={"choices": ['Translation','Affine','Euclidean','Homography']})
def register_rois(viewer: Viewer, image: Image,
                    labels_layer: Labels,
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
        The image stack (time,y,x) to be registered.
    labels_layer: Labels
        The labels will define the rectangular ROIs to be registered.
    mode: str
        The type of cv2 registration.
    median_filter_size:int
        Size of the median filter applied before registration.
    scale: float
        Rescaling factor of the image for the registation
        Does not affect the registered image scale.
    bbox_zoom: int
        The bounding boxes encloding the ROIs are zoomed by this factor.
    register_entire_image:bool
        If True, the entire image is registered around the bounding box.
    show_registered_stack:bool
        If True,  shows the registered stacks as new image layers.
    '''
    print('Starting registration...')
    # remove registration points if present
    label_values = max_projection(labels_layer)
    label_colors = get_labels_color(label_values)
    labels = max_projection(labels_layer)
    initial_time_index = viewer.dims.current_step[0]
    widgets_shared_variables.initial_time_index = initial_time_index
    print('... with initial time index:', widgets_shared_variables.initial_time_index)
    real_initial_positions, real_roi_sy, real_roi_sx = get_rois_props(labels, 
                                                                      initial_time_index,
                                                                      bbox_zoom) 
    roi_num = len(real_initial_positions)
    stack = np.asarray(image.data)
    time_frames_num, sy, sx = stack.shape
        
    def add_rois(params):
            import numpy.matlib
            rectangles = params[0]
            _centers = params[1]
            rectangles = rectangles.reshape((roi_num*time_frames_num,4,3))
            centers = _centers.reshape((roi_num*time_frames_num,3))
            color_array= numpy.matlib.repmat(label_colors,len(rectangles)//roi_num,1)
            
            points_layer_name = f'centroids_{image.name}'
            rectangles_name = f'rectangles_{image.name}'
            if points_layer_name in viewer.layers:
                viewer.layers.remove(points_layer_name)
            if rectangles_name in viewer.layers:
                viewer.layers.remove(rectangles_name)
            shapes = viewer.add_shapes(np.array(rectangles[0]),
                              edge_width=1,
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
                    y = pos[initial_time_index,1]
                    x = pos[initial_time_index,2]
                    
                    sizey = real_roi_sy[roi_idx] 
                    sizex = real_roi_sx[roi_idx]
                    if register_entire_image:
                        sizex = min(sx-x,x)*2
                        sizey = min(sy-y,y)*2
                    
                    registered = select_rois_from_stack(stack, pos, 
                                                         [int(sizey)], [int(sizex)])
            
                    registered_roi_name= f'registered_{image.name}_roi{roi_idx}'
                    if registered_roi_name in viewer.layers:
                            viewer.layers.remove(registered_roi_name)
                    viewer.add_image(np.array(registered), name= registered_roi_name)
                    #im.translate = [0,int(y-sizey/2),int(x-sizex/2)]         
            print('... ending registration.')
        
    @thread_worker(connect={'returned':add_rois})
    def _register_rois():    
        try:
            register_rois.enabled = False
            warnings.filterwarnings('ignore')        
            resized = resize_stack(stack, scale)
            resized, _vmin, _vmax = normalize_stack(resized)
            image0 = resized[initial_time_index,...]
            
            initial_positions = rescale_position(real_initial_positions,scale)
            roi_sy = [int(ri*scale) for ri in real_roi_sy]
            roi_sx = [int(ri*scale) for ri in real_roi_sx]
            previous_rois = select_rois_from_image(image0, initial_positions, roi_sy,roi_sx)
            previous_rois = filter_images(previous_rois, median_filter_size)
    
            rectangles = np.zeros([time_frames_num,roi_num,4,3]) 
            centers = np.zeros([time_frames_num,roi_num,3])
             # register forwards
            next_positions = initial_positions.copy()
            real_next_positions = rescale_position(next_positions,1/scale)
            centers[initial_time_index,:,:] = np.array(real_next_positions)
            rectangles[initial_time_index,:,:,:] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
            for t_index in range(initial_time_index+1, time_frames_num, 1):
                next_rois = select_rois_from_image(resized[t_index,...], next_positions, roi_sy,roi_sx)
                next_rois = filter_images(next_rois, median_filter_size)
                dx, dy, _wm = align_with_registration(next_rois, previous_rois,
                                                 mode)
                next_positions = update_position(next_positions, dz = 1,
                                                 dx_list = dx, dy_list = dy)
                real_next_positions = rescale_position(next_positions,1/scale)
                centers[t_index,:,:] = np.array(real_next_positions)
                rectangles[t_index,:,:,:] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
            # register backwards  
            next_positions = initial_positions.copy()    
            for t_index in range(initial_time_index-1, -1, -1):
                next_rois = select_rois_from_image(resized[t_index,...], next_positions, roi_sy,roi_sx)
                next_rois = filter_images(next_rois, median_filter_size)
                dx, dy, _wm = align_with_registration(next_rois,previous_rois,
                                                  mode)
                next_positions = update_position(next_positions, dz = -1,
                                                  dx_list = dx, dy_list = dy)
                real_next_positions = rescale_position(next_positions,1/scale)
                centers[t_index,:,:] = np.array(real_next_positions)
                rectangles[t_index,:,:,:] = create_rectangles(real_next_positions, real_roi_sy, real_roi_sx)
        except Exception as e:
            print(e)
        finally:
            register_rois.enabled = True       
        return (rectangles, centers)
    _register_rois() 
    
    
def calculate_intensity(image:Image,
                        roi_num:int,
                        points_layer:Points,
                        labels_layer:Labels
                        ):
    """
    Calculates the mean intensity,
    within rectangular Rois of size roi_size, centered in points_layer,
    taking into account only the pixels that are in one of the labels of labels_layer
    """
    initial_time_index = widgets_shared_variables.initial_time_index
    labels_data = max_projection(labels_layer)
    label_values = get_labels_values(labels_data)
    stack = np.array(image.data)
    locations = points_layer.data
    st, _sy, _sx = stack.shape
    _ , roi_sizey, roi_sizex = get_rois_props(labels_data)   
    rois = select_rois_from_stack(stack, locations, roi_sizey, roi_sizex)
    initial_location = initial_time_index*roi_num
    label_rois = select_rois_from_image(labels_data,
                                        locations[initial_location:initial_location+roi_num],
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

    return intensities, initial_time_index


def measure_displacement(image, roi_num, points):
    """
    Measure the displacement of each roi:
    dr: relative to its position in the previous time frame 
    deltar: relative to the initial position.
    """
    stack = image.data
    st, sy, sx = stack.shape
    locations = points.data
    reshaped = locations.reshape((st, roi_num, stack.ndim))
    xy = reshaped[...,1:] # remove the time index value
    xy0 = xy[0,...] # take the x,y cohordinates of the rois in the first time frame
    
    deltar = np.sqrt( np.sum( (xy-xy0)**2, axis=2) )
    rolled = np.roll(xy, 1, axis=0) #roll one roi    
    rolled[0,...] = xy0
    dxy = xy-rolled
    dr = np.sqrt( np.sum( (dxy)**2, axis=2) )
    return xy, deltar, dxy, dr


@magic_factory(call_button="Process registered ROIs")
def process_rois(viewer: Viewer, image: Image, 
                 registered_points: Points,
                 labels_layer: Labels,
                 correct_photobleaching: bool,
                 plot_results:bool = True,
                 save_results:bool = False,
                 path: pathlib.Path = os.getcwd()+'temp.xls'
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
    correct_photobleaching: bool
        If True photobleaching correction is appleid to the the intensities.
    plot_results:bool
        If Ture, shows the plots of the collected data with matplotlib in the console.
    save_results:bool
        If True, creates an excel file with processing results.
    path: str
        Folder with filename of the excel file. 
    '''
    
    warnings.filterwarnings('ignore')
    print('Starting processing ...')
    try:
        process_rois.enabled = False
        time_frames_num, sy, sx = image.data.shape
        locations = registered_points.data
        roi_num = len(locations) // time_frames_num
        intensities, initial_time_index = calculate_intensity(image, roi_num, 
                                          registered_points,
                                          labels_layer)
        yx, deltar, dyx, dr = measure_displacement(image, roi_num, registered_points)
        
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
            save_in_excel(filename_xls = path,
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