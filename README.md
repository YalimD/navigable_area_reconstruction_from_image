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

