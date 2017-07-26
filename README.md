# Kaggle-Planet-Amazon
Code for Kaggle competition - Planet: Understanding the Amazon from Space. PyTorch CNN Finetune suite.

For background and data for this competition, please clink on link below:
https://www.kaggle.com/c/planet-understanding-the-amazon-from-space


This code take the [PyTorch pre-trained models](http://pytorch.org/docs/master/torchvision/models.html), replace their final layers with 17 classes and add a Sigmoid activation layer for output. 

Variations of the Resnet family (18, 34, 50, 101, 152), Densenet family (121, 161, 169, 201), inception_v3, VGG16 and VGG19 are included.

