# navigable_area_reconstruction_from_image

Given an input image taken from a survaillance-alike video and pedestrian detection data, tries to calculate the perspective correction operation that rectifies the navigable area in the image. The rectified image represents the blueprint (map) of the navigable regions.

Uses concepts such as vanishing points, RANSAC, circular points and stratified rectification.

References:
* https://github.com/chsasank/Image-Rectification 
* Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.  
 "Auto-rectification of user photos." 2014 IEEE International Conference on Image Processing (ICIP). IEEE, 2014.
* Bazin, Jean-Charles, and Marc Pollefeys. "3-line RANSAC for orthogonal  
 vanishing point detection." 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012.
* Bose, Biswajit, and Eric Grimson. "Ground plane rectification by tracking moving objects." Proceedings of the Joint IEEE International Workshop on Visual Surveillance and Performance Evaluation of Tracking and Surveillance. 2003.

Requires: (Tested on Python 3.7 with Anaconda)
* OpenCV 4.2
* Numpy 
* Skimage 0.16.2

If you find this work useful, please cite

Yalım Doğan, Sinan Sonlu, and Uğur Güdükbay. An Augmented Crowd Simulation System Using Automatic Determination of Navigable Areas. Computers & Graphics, 95:141–155, April 2021.

```
@article{DOGAN2021141,
title = {An augmented crowd simulation system using automatic determination of navigable areas},
journal = {Computers & Graphics},
volume = {95},
pages = {141-155},
year = {2021},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2021.01.012},
url = {https://www.sciencedirect.com/science/article/pii/S0097849321000121},
author = {Yalım Doğan and Sinan Sonlu and Uğur Güdükbay},
keywords = {Pedestrian detection and tracking, Data-driven simulation, Three-dimensional reconstruction, Crowd simulation, Augmented reality, Deep learning},
abstract = {Crowd simulations imitate the group dynamics of individuals in different environments. Applications in entertainment, security, and education require augmenting simulated crowds into videos of real people. In such cases, virtual agents should realistically interact with the environment and the people in the video. One component of this augmentation task is determining the navigable regions in the video. In this work, we utilize semantic segmentation and pedestrian detection to automatically locate and reconstruct the navigable regions of surveillance-like videos. We place the resulting flat mesh into our 3D crowd simulation environment to integrate virtual agents that navigate inside the video avoiding collision with real pedestrians and other virtual agents. We report the performance of our open-source system using real-life surveillance videos, based on the accuracy of the automatically determined navigable regions and camera configuration. We show that our system generates accurate navigable regions for realistic augmented crowd simulations.}
}
```
