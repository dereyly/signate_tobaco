import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import pretrainedmodels
# import torch.nn.functional as F
# from nasnet import NASNetALarge

__all__ = ['se_resnext50_32x4d_multi','resnet50_multi','se_resnet101_multi', 'InceptionResNetV2_multi', 'resnet152_multi',
           'se_resnext101_32x4d_multi']



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class InResnMultiClasses(nn.Module):
    def __init__(self, model, num_classes=[1000], head_size=1536, is_embed=False):
        self.is_embed=is_embed
        self.n_multi=len(num_classes)
        self.inplanes = 64
        super(InResnMultiClasses, self).__init__()
        self.conv2d_1a = model.conv2d_1a
        self.conv2d_2a = model.conv2d_2a
        self.conv2d_2b = model.conv2d_2b
        self.maxpool_3a = model.maxpool_3a
        self.conv2d_3b = model.conv2d_3b
        self.conv2d_4a = model.conv2d_4a
        self.maxpool_5a = model.maxpool_5a
        self.mixed_5b = model.mixed_5b
        self.repeat = model.repeat
        self.mixed_6a = model.mixed_6a
        self.repeat_1 = model.repeat_1
        self.mixed_7a = model.mixed_7a
        self.repeat_2 = model.repeat_2
        self.block8 = model.block8
        self.conv2d_7b = model.conv2d_7b

        self.avgpool = nn.AvgPool2d(4)
        self.drop = nn.Dropout(p=0.25)
        for idx, n_cls in enumerate(num_classes):
            self.add_module('classifier{}'.format(idx), nn.Linear(head_size, n_cls))

        self.classifiers = []
        for module_name, classifier in self.named_modules():
            if 'classifier' in module_name:
                self.classifiers.append(classifier)


    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)

        z = self.avgpool(x)
        z = z.view(z.size(0), -1)
        z = self.drop(z)
        if self.is_embed:
            return z
        else:
            return [clf(z) for clf in self.classifiers]


class ModelMultiClasses(nn.Module):
    def __init__(self, model, num_classes=[1000], head_size=2048, is_embed=False):
        self.is_embed=is_embed
        self.n_multi=len(num_classes)
        self.inplanes = 64
        super(ModelMultiClasses, self).__init__()
        self.is_layer0 = False
        if 'layer0' in model._modules:
            self.is_layer0=True
        if self.is_layer0:
            self.layer0 = model.layer0
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = nn.AvgPool2d(5)
        self.drop = nn.Dropout(p=0.25)
        for idx, n_cls in enumerate(num_classes):
            self.add_module('classifier{}'.format(idx), nn.Linear(head_size, n_cls))

        self.classifiers = []
        for module_name, classifier in self.named_modules():
            if 'classifier' in module_name:
                self.classifiers.append(classifier)


    def forward(self, x):
        if self.is_layer0:
            x = self.layer0(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        z = self.avgpool(x)
        z = z.view(z.size(0), -1)
        z = self.drop(z)
        if self.is_embed:
            return z
        else:
            return [clf(z) for clf in self.classifiers]



def se_resnext50_32x4d_multi(pretrained=True, **kwargs):
    model_name = 'se_resnext50_32x4d'  # could be fbresnet152 or inceptionresnetv2
    model_base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # dim_feats = model_base.last_linear.in_features
    model = ModelMultiClasses(model_base.cuda(), **kwargs)
    if pretrained is not None:
        # settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        # initialize_pretrained_model(model, num_classes, settings)
        #prefix = 'features.'
        prefix=''
        print("=> using pre-trained model se_resnext50_32x4d")
        print('use prefix ->' + prefix)
        pretrained_state = model_base.state_dict()
        model_state = model.state_dict()

        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        model.load_state_dict(model_state)
    return model

def se_resnext101_32x4d_multi(pretrained=True, **kwargs):
    model_name = 'se_resnext101_32x4d'  # could be fbresnet152 or inceptionresnetv2
    model_base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # dim_feats = model_base.last_linear.in_features
    model = ModelMultiClasses(model_base.cuda(), **kwargs)
    if pretrained is not None:
        # settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        # initialize_pretrained_model(model, num_classes, settings)
        #prefix = 'features.'
        prefix=''
        print("=> using pre-trained model se_resnext101_32x4d")
        print('use prefix ->' + prefix)
        pretrained_state = model_base.state_dict()
        model_state = model.state_dict()

        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        model.load_state_dict(model_state)
    return model

def resnet50_multi(pretrained=True, **kwargs):
    model_name = 'resnet50'  # could be fbresnet152 or inceptionresnetv2
    model_base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = ModelMultiClasses(model_base.cuda(), **kwargs)
    if pretrained is not None:
        # settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        # initialize_pretrained_model(model, num_classes, settings)
        prefix = ''
        print("=> using pre-trained model " +model_name)
        print('use prefix ->' + prefix)
        pretrained_state = model_base.state_dict()
        model_state = model.state_dict()

        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        model.load_state_dict(model_state)
    return model

def se_resnet101_multi(pretrained=True, **kwargs):
    model_name = 'se_resnet101'  # could be fbresnet152 or inceptionresnetv2
    model_base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # dim_feats = model_base.last_linear.in_features
    model = ModelMultiClasses(model_base.cuda(), **kwargs)
    if pretrained is not None:
        # settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        # initialize_pretrained_model(model, num_classes, settings)
        #prefix = 'features.'
        prefix=''
        print("=> using pre-trained model se_resnet101")
        print('use prefix ->' + prefix)
        pretrained_state = model_base.state_dict()
        model_state = model.state_dict()

        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        model.load_state_dict(model_state)
    return model

def senet154_multi(pretrained=True, **kwargs):
    model_name = 'senet154'  # could be fbresnet152 or inceptionresnetv2
    model_base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    # dim_feats = model_base.last_linear.in_features
    model = ModelMultiClasses(model_base.cuda(), **kwargs)
    if pretrained is not None:
        # settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        # initialize_pretrained_model(model, num_classes, settings)
        #prefix = 'features.'
        prefix=''
        print("=> using pre-trained model senet154")
        print('use prefix ->' + prefix)
        pretrained_state = model_base.state_dict()
        model_state = model.state_dict()

        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        model.load_state_dict(model_state)
    return model

def InceptionResNetV2_multi(pretrained=True, **kwargs):
    model_name = 'inceptionresnetv2'  # could be fbresnet152 or inceptionresnetv2
    model_base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = InResnMultiClasses(model_base.cuda(), **kwargs)
    if pretrained is not None:
        # settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        # initialize_pretrained_model(model, num_classes, settings)
        prefix = ''
        print("=> using pre-trained model InceptionResNetV2")
        print('use prefix ->' + prefix)
        pretrained_state = model_base.state_dict()
        model_state = model.state_dict()

        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        model.load_state_dict(model_state)
    return model


def resnet152_multi(pretrained=True, **kwargs):
    model_name = 'resnet152'  # could be fbresnet152 or inceptionresnetv2
    model_base = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = ModelMultiClasses(model_base.cuda(), **kwargs)
    if pretrained is not None:
        # settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        # initialize_pretrained_model(model, num_classes, settings)
        prefix = ''
        print("=> using pre-trained model resnet152")
        print('use prefix ->' + prefix)
        pretrained_state = model_base.state_dict()
        model_state = model.state_dict()

        for k, v in pretrained_state.items():
            key = prefix + k
            if key in model_state and v.size() == model_state[key].size():
                model_state[key] = v
                print(key)
            else:
                print('not copied --------> ', key)
        model.load_state_dict(model_state)
    return model