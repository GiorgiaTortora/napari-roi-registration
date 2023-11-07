from napari_roi_registration import register_rois
from napari_roi_registration import subtract_background
import numpy as np
from napari.layers import Points
from napari.layers import Image

def test_subtract_background(make_napari_viewer, capsys):
    
    viewer = make_napari_viewer()
    im_data = np.random.random((10, 200, 100))   
    image_layer = viewer.add_image(im_data)
    
    label_data = np.zeros((200,100), dtype=np.int8)
    label_data[50:60, 50:60] = 1
    label_layer = viewer.add_labels(label_data)

    background_widget = subtract_background()
    
    background_widget(viewer, viewer.layers[0], viewer.layers[1])
    out, err = capsys.readouterr()
    assert err == ''
    
    
    
def test_register_rois(make_napari_viewer, capsys):
    
    viewer = make_napari_viewer()
        
    #im_data = np.random.random((10, 200, 100))
    im_data = np.zeros((10, 200, 100))
    
    im_data[:,52:58, 52:58] = 1
    image_layer = viewer.add_image(im_data)
    
    label_data = np.zeros((200,100), dtype=np.int8)
    label_data[50:60, 50:60] = 1
    #label_data[150:170, 70:80] = 2
    label_layer = viewer.add_labels(label_data)
    
    viewer.dims.current_step = (5, 0, 0)

    registration_widget = register_rois()
    
    #registration_widget(viewer, viewer.layers[0], viewer.layers[1])
    out, err = capsys.readouterr()
    assert err == '' 