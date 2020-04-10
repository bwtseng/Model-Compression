#from .ImageNet import *
# That will import __all__, which list all the function will be incorporated. 
# "." mean current decoratory.
from . import ImageNet as imagenet_models
from .ImageNet import *
#del resnet50
#del alexnet
#del classifier
#del vgg
#del inception 
#del resnet_cifar
#del shufflenetv2


imagenet_function_dict = {
    'resnet': imagenet_models.create_resnet,
    'vgg': imagenet_models.create_vgg,
    'shufflenet': imagenet_models.create_shufflenet,
    'squeezenet': imagenet_models.create_squeezenet,
    'mobilenet': imagenet_models.create_mobilenet,
    'inception': imagenet_models.inception_v3,
    'alexnet': imagenet_models.create_alexnet,
}

del imagenet_models

from . import Cifar10 as cifar10_models
from .Cifar10 import *
cifar10_function_dict = {
    'resnet': cifar10_models.create_resnet_cifar10,
    'vgg': cifar10_models.create_vgg_cifar,
}
del cifar10_models


from . import MNIST as mnist_models
from .MNIST import *
mnist_function_dict = {
    'simplenet': mnist_models.create_mnistnet,
}
del mnist_models

from . import VWW 
from .VWW import *
vww_function_dict = {
    'mobilenet':VWW.create_mobilenet,
}
del VWW

data_function_dict = {
    'imagenet': imagenet_function_dict,
    'cifar10': cifar10_function_dict,
    'mnist': mnist_function_dict,
    'vww': vww_function_dict # just support mobilenetv1 and v2
    
} 