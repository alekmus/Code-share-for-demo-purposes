# Code-share-for-demo-purposes

Purpose of this repository is to simply house coding samples that are either their own individual smaller script that don't warrant their own full repositories 
or are single modules of larger projects that aren't ready to be made public as a whole yet.

Currently the repo houses

1. A convolutional neural network script that classifies images into 14 non-exclusive classes. 
The Script utilizes Tensorflow and it's data module to efficiently perform data augmentation, batching and reading the dataset from disk.

2. A Python module that reads outlined data tables and finds their location in an image.
The module is part of a larger project and leverages OpenCV to detect the outlines of a data table using computer vision. 
The module uses open-close morphological operations to detect orthogonal lines and clusters their crossing points to find individual cells of a table.
