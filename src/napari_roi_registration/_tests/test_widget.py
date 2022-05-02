from napari_roi_registration import subtract_background, register_rois, process_rois
import numpy as np

def test_subtract_background(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    my_widget = subtract_background()
    
def test_register_rois(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    my_widget = register_rois()
    
def test_process_rois(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    my_widget = process_rois()