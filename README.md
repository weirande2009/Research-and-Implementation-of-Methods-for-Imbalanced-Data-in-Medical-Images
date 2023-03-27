# Research-and-Implementation-of-Methods-for-Imbalanced-Data-in-Medical-Images
This is my graduation thesis of undergraduate and here is the abstract.

Semantic segmentation is one of the important tasks of medical image processing. It can 
automatically segment the target objects of medical image, so as to speed up the reading and analysis 
speed of doctors and reduce the burden of doctors. In the semantic segmentation of medical images, the 
imbalance of data categories is a very common problem, which brings great challenges to the learning of 
neural networks. In order to solve this problem, this paper explores the loss function, carrying out 
experiments on two nuclear segmentation datasets for a variety of loss functions aiming at imbalanced 
data, and uses a variety of semantic segmentation networks to get the most appropriate algorithm 
combination. At the same time, the class weight function is explored and a novel class weight function is 
proposed; This paper also explores the number of difficult samples for online bootstrapping loss function, 
and finds the most suitable number of difficult samples. Through the analysis of the problems of the above 
loss function, and combined with the high accuracy of the binary-classification model for background 
classification, a novel two-stage method of reusing background is proposed, which achieves better results 
on both datasets.
The work done in this paper is as follows:
(1) Loss functions and their combinations for imbalanced data are deeply explored, and good results 
are obtained in the experiment.
(2) Different class weight functions of focal loss are studied, and a novel class weight function for
extremely imbalanced data is proposed.
(3) Experiments are carried out on the number of difficult samples of online bootstrapping loss 
function, and the number of difficult samples most suitable for the unbalanced dataset is found.
(4) A novel two-stage method of reusing background is proposed, and the accuracy is improved on 
two datasets.
