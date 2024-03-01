# FewShotsLearning

## Context
CT scans provide highly precise 3D images of the human body (up to 0.5 mm resolution) and thus enable the capture of human anatomy. The goal of this challenge is to automatically segment anatomical structures of the human body, as well as tumors, on a CT scan, without semantically identifying associated organs. In other words, it is about identifying visible shapes on a CT scan. This object discovery problem is intuitive and simple for a human, who can easily identify new objects in a scene, and even on a CT scan, even if they have never seen that object before.

While supervised segmentation algorithms for these individual structures are now considered solved, it is not possible to use supervised learning to generalize to new unseen anatomical structures.

## Goal
The goal of this challenge is to segment structures using their shape, but without exhaustive annotations. The training data consists of two types of images:

1. CT scanner images with segmentation masks of individual anatomical structures and tumors.
   These act as the ground truth definition of what an anatomical structure is.
   However, they are not meant to be representative of all possible structures and their diversity, but can still be used as training material.
   This makes this problem a mixture of an out-of-domain learning problem (some structures in the test set are not present in the training set) and a few-shot learning problem (some structures are common between the training set and the test set, but there are few examples).
   
2. Raw CT scanner images, without any segmented structure.
   These can be used as additional training material, within the framework of unsupervised training.

The test set consists of new images with their corresponding segmented structures, and the metric measures the ability to correctly segment and separate different structures in an image.

Note: The segmented structures do not cover the entirety of the image; some pixels are not part of an identifiable structure, as seen in the image above. They are therefore considered as part of the background.

## Data Description
The input is a list of 2D images (i.e., a 3D numpy array) in grayscale, each corresponding to a slice of a CT scan (in the transverse plane) of size 512x512 pixels. The slices are shuffled, so there is no 3D information.

The output is a list of 2D matrices (i.e., a 3D numpy array) of size 512x512 with integer values (uint8). Each position (w, h) of each matrix $Y_{i,w,h}$ identifies a structure.

This problem can be viewed as an image-by-image pixel clustering problem, where each structure is a cluster in the image. The identifier of a structure in an image may not necessarily have the same number. For example, the structure associated with the liver may be mapped to label 4 in one image and to 1 in another image.

The training set consists of 2000 images, divided into two groups:

400 with fully segmented structures.
For these images, the corresponding output is a 2D matrix with pixel labels for segmented structures and tumors, with other pixels defined as 0.
1600 of them have no annotation at all.
For these images, the corresponding output is a 2D matrix filled with zeros.
Note: Segmentations of structures, in the form of images (in addition to the CSV), from the training set are provided in the supplementary materials.

The test set consists of 500 images with segmented structures. For these images, the label is a 2D matrix with segmented structures, and background pixels are valued at 0. Considering that an individual image with its label matrix is about 400KB, and we have 1500 images, we then have a dataset of approximately 600MB in total.

Note: The segmentation map is not dense, meaning that some pixels between structures are not part of segmented structures, as seen in the image above. These pixels are considered as part of the background.

Note: The use of additional radiological training data, pre-trained models on radiological data, or any other external radiological data source is not allowed. The only permitted source of radiological data is the training and test sets provided by the organizers. However, you are allowed to use external non-radiological data and models (DINO v2, SAM...).

The final metric is calculated by averaging the Rand index between each label and its associated prediction, while excluding background pixels (i.e., 0 in the label).
