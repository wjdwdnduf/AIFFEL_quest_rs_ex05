'''
This file enables importing models directly from the 'models' directory.
It acts as a gateway to the ResNet and PlainNet definitions.
'''

# Import from resnet.py
from .resnet import ResNet34, ResNet50

# Import from plain.py
from .plain import PlainNet34, PlainNet50

# in-line comment: Ensure there are no typos in the names 'ResNet34' or 'ResNet50'