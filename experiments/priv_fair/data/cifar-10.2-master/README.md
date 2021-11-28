# CIFAR 10.2

This repository contains the CIFAR-10.2 dataset prepared and published in "Harder or Different? A Closer Look at Distribution Shift in Dataset Reproduction" by Shangyun Lu, Bradley Nott, Aaron Olson, Alberto Todeschini, Puya Vahabi, Yair Carmon and Ludwig Schmidt.

The files `cifar102_train.npz` and `cifar102_test.npz` contain the train and test sets of CIFAR-10.2. Each file contains the the following keys:
- `images`: an nx32x32x10 numpy uint8 array containing the image data, where n=10000 for the training set and n=2000 for the test set
- `labels`: a numpy int64 array containing labels (integers between 0 and 9)
- `label_names`: a list mapping CIFAR-10 label indices to their meaning
- `ti_indices`: an array of the image indices in the 80 Million Tiny Images
- `keywords`: a list of the TinyImages keyword for every image