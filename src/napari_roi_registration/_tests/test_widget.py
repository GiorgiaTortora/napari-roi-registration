from napari_roi_registration import register_rois, subtract_background
import numpy as np
from napari.layers import Points, Image

def test_subtract_background(make_napari_viewer, capsys):
    
    viewer = make_napari_viewer()
        
    image_layer = viewer.add_image(np.random.random((10, 200, 100)))
    
    label_data = np.zeros((200,100), dtype=np.int8)
    label_data[50:60, 50:60] = 1
    # label_data[150:170, 70:80] = 2
    label_layer = viewer.add_labels(label_data)

    background_widget = subtract_background()
    
    background_widget(viewer, viewer.layers[0], viewer.layers[1])
    
    # assert len(viewer.layers) == 3 
    # assert type(viewer.layers[2]) == Image
    
    
