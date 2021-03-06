# Kaggle-Planet-Amazon
PyTorch CNN Finetune suite for Kaggle competition - Planet: Understanding the Amazon from Space.

Please click [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) for background and data of this competition.

This code takes the [PyTorch pre-trained models](http://pytorch.org/docs/master/torchvision/models.html), replace their final layers with 17 classes and add a Sigmoid activation layer for output. 

Variations of the Resnet family (18, 34, 50, 101, 152), Densenet family (121, 161, 169, 201), inception_v3, VGG16 and VGG19 are included.

# Structure of the Code
Main code in PyTorch_Amazon.py:
   1. Setting of path and model parameters
   1. Splitting data for training and validation
   1. Create dataset_loader, with on-the-fly image augmentation with functions from Image_transformation.py
   1. Loading the model (imported from PyTorch_models.py), setting the learning rate schedule and optimizer. Here the learning rate for the last classifer is 10 times larger than previous layers. The code should automatically download the pre-trained weights upon first run. If this failed, it can be done manually - just follow the error messages.
   1. Train model and saved the best according to validation set performance.
   1. Generate prediction (for both train and test data) with Test-time augmentation. The former is needed for F2 score threshold optimisation.
   1. Generate submission.

# Performance
Many Kagglers managed to reach a LB score of over 0.93 on the public leaderboard (see [discussion](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/35797)), but mine hovered around 0.925. It turns out that there was a bug in my earlier version - instead of learning more aggressively on the final layer with 10x learning rate, I set the previous layers with the 10x rate. With ensembing I brought it just over the magic 0.93 on the public leader board, but fell just short on the private one. 

With the latest version I tested it on Densenet121 with 40 epochs, and reached 0.929 on public LB for a single model without Kfold, easily better than any of my single models previousy. With ensemble, this should easily pass the 0.93 mark on both public and private LB.

# Reference:
1. [PyTorch Tutorial on Transfer Learning](http://pytorch.org/tutorials/)
2. Discussions on the [PyTorch Forum](https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/11)
3. Kaggle Forum on this competition. Among others, kernels from [Mamy Ratsimbazafy](https://www.kaggle.com/mratsim/starting-kit-for-pytorch-deep-learning), [Anokas](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475), [Peter Giannakopoulos](https://www.kaggle.com/petrosgk/keras-vgg19-0-93028-private-lb) and [Heng CherKeng](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/33559)

# Requirement
Torch version: '0.1.12_2'
