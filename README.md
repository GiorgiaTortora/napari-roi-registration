# napari-roi-registration

[![License](https://img.shields.io/pypi/l/napari-roi-registration.svg?color=green)](https://github.com/GiorgiaTortora/napari-roi-registration/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-roi-registration.svg?color=green)](https://pypi.org/project/napari-roi-registration)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-roi-registration.svg?color=green)](https://python.org)
[![tests](https://github.com/GiorgiaTortora/napari-roi-registration/workflows/tests/badge.svg)](https://github.com/GiorgiaTortora/napari-roi-registration/actions)
[![codecov](https://codecov.io/gh/GiorgiaTortora/napari-roi-registration/branch/main/graph/badge.svg)](https://codecov.io/gh/GiorgiaTortora/napari-roi-registration)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-roi-registration)](https://napari-hub.org/plugins/napari-roi-registration)

A Napari plugin for the registration of regions of interests (ROI) in a time lapse acquistion and processing of the intensity of the registered data.

The ROI are defined using a Labels layer. Registration of multiple ROIs is supported.  

The `Registration` widget uses the user-defined labels, constructs a rectangular ROI around each of them and registers the ROIs in each time frame.
The `Processing` widget measures the ROI displacements and extract the average intensity of the ROI, calculated in the label area.
The `Subtract background` widget subtracts a background on each frame, calculated as the mean intensity on a Labels layer. 
Tipically useful when ambient light affects the measurement.  

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/raw/main/images/roi_registration.gif)

## Installation

You can install `napari-roi-registration` via [pip]:

    pip install napari-roi-registration



To install latest development version :

    pip install git+https://github.com/GiorgiaTortora/napari-roi-registration.git

## Usage

### Registration Widget

1. Create a new Labels layer and draw one or more labels where you want to select a ROI (Region Of Interest). Each color in the same Labels layer represent a different label which is a different ROI.

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/raw/main/images/Picture1.png)

2. Push the `Register ROIs` button: registration of the entire stack will be performed. When the registration is finished two new layers will appear in the viewer. One layer contains the centroids of the drawn labels while the other contains the bounding boxes encloding the ROIs.

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/raw/main/images/Picture2.png)

### Processing Widget

Pushing the `Process registered ROIs` button processing of the registered ROIs will be performed. Information about the intensity of the registered data and the displacement of the ROIs will be given. In the IPhyton console the displacement vs time index and the mean intensity vs time index plots will appear.
Choosing the `save results` option an excel file containg information about the ROIs positions, displacement and intensity at each frame will be generated. 

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/raw/main/images/Picture3.png)

### Background Widget

1. Create a new Labels layer and draw a label on the area from which to get the background. 

![raw](https://github.com/GiorgiaTortora/napari-roi-registration/raw/main/images/Picture4.png)

2. Push the `Subtract background` button. A new image layer will appear in the viewer. This layer contains the image to which the background was subtracted.

## Contributing 

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-roi-registration" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/GiorgiaTortora/napari-roi-registration/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
