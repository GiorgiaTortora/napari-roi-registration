from src.napari_roi_registration._widget import subtract_background, register_rois, process_rois
'''
Script that runs the napari plugin from the IDE. 
It is not executed when the plugin runs.
'''
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
    napari.run()
    