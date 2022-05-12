from napari_roi_registration import register_rois
import numpy as np
from napari.layers import Points

def test_register_rois(make_napari_viewer, capsys):
    
    viewer = make_napari_viewer()
        
    image_layer = viewer.add_image(np.random.random((10, 200, 100)))
    
    label_data = np.zeros((200,100), dtype=np.int8)
    label_data[50:60, 50:60] = 1
    label_data[150:170, 70:80] = 2
    label_layer = viewer.add_labels(label_data)

    register_widget = register_rois()
    
    register_widget(image_layer, label_layer)
    
    assert len(viewer.layers) == 4 
    assert type(viewer.layers[3]) == Points
    
    
