from napari_roi_registration import processing_widget, register_widget, background_widget
import numpy as np

def test_subtract_background(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # background_widget = subtract_background()
    
def test_register_rois(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # register_widget = register_rois()
    
def test_process_rois(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    # processing_widget = process_rois()