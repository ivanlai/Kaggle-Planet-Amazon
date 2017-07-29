# Reference and ideas from  http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function

import torch.nn as nn
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")


##################################################################
##  PyTorch Model implementations in
##  /usr/local/lib/python2.7/dist-packages/torchvision/models  ##
##################################################################

def resnet18(num_classes, pretrained=True, freeze=False):

    model = models.resnet18( pretrained=True)
    if freeze:
        model = freeze_all_layers(model)

    # Parameters of newly constructed modules have requires_grad=True by default
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Resnet18'


def resnet34(num_classes, pretrained=True, freeze=False):

    model = models.resnet34( pretrained=True)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Resnet34'


def resnet50(num_classes, pretrained=True, freeze=False):

    model = models.resnet50( pretrained=pretrained)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Resnet50'


def resnet101(num_classes, pretrained=True, freeze=False):

    model = models.resnet101( pretrained=pretrained)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Resnet101'


def resnet152(num_classes, pretrained=True, freeze=False):

    model = models.resnet152( pretrained=pretrained)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Resnet152'


##################################################################
def densenet121(num_classes, pretrained=True, freeze=False):

    model = models.densenet121( pretrained=pretrained)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Densenet121'


def densenet161(num_classes, pretrained=True, freeze=False):

    model = models.densenet161( pretrained=pretrained)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Densenet161'


def densenet169(num_classes, pretrained=True, freeze=False):

    model = models.densenet169(pretrained=pretrained)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Densenet169'


def densenet201(num_classes, pretrained=True, freeze=False):

    model = models.densenet201( pretrained=pretrained)
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Densenet201'


##################################################################
def inception_v3(num_classes, pretrained=True, freeze=False):

    model = models.inception_v3(pretrained=pretrained)
    model.aux_logits = False
    if freeze:
        model = freeze_all_layers(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = Add_Sigmoid(model)

    return model, 'Inception_v3'

##################################################################
def vgg16(num_classes, pretrained=True, freeze=False):
    # Credit: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/10

    model = models.vgg16(pretrained=True)
    if freeze:
        model = freeze_all_layers(model)
    mod = list(model.classifier.children())
    mod.pop()
    mod.append(nn.Linear(4096, 17))
    new_classifier = nn.Sequential(*mod)
    model.classifier = new_classifier

    model = Add_Sigmoid(model)

    return model, 'VGG16'

##################################################################
def vgg19(num_classes, pretrained=True, freeze=False):
    # Credit: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/10

    model = models.vgg19(pretrained=True)
    if freeze:
        model = freeze_all_layers(model)
    mod = list(model.classifier.children())
    mod.pop()
    mod.append(nn.Linear(4096, 17))
    new_classifier = nn.Sequential(*mod)
    model.classifier = new_classifier

    model = Add_Sigmoid(model)

    return model, 'VGG19'

##################################################################
class Add_Sigmoid(nn.Module):
    def __init__(self, pretrained_model):
        super(Add_Sigmoid, self).__init__()
        self.pretrained_model = pretrained_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.pretrained_model(x))



##################################################################
def freeze_all_layers(model):
    #Freeze all layers except last during training (last layer training set to true when it get redefined)
    for param in model.parameters():
        param.requires_grad = False
    return model

##################################################################