from napari_roi_registration import register_rois

def test_register_rois(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    # layer = viewer.add_image(np.random.random((100, 100)))

    # register_widget = register_rois()
    
# def test_register_rois(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     # layer = viewer.add_image(np.random.random((100, 100)))

#     # register_widget = register_rois()
    
# def test_process_rois(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     # layer = viewer.add_image(np.random.random((100, 100)))

#     # processing_widget = process_rois()