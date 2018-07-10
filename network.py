''' It's network init file '''

from netpy.nets import FeedForwardNet
from netpy.modules import LinearLayer,SigmoidLayer, FullRelation
from netpy.teachers import BackPropTeacher
import numpy as np

# Import file with settings
import conf

net = FeedForwardNet(name=conf.name)

# Add your layers here
layer1 = LinearLayer(1024)
layer2 = SigmoidLayer(1024 * 2)
layer3 = SigmoidLayer(1024 * 2)
layer4 = SigmoidLayer(1024)

# Add your layers to the net here
net.add_Input_Layer(layer1)
net.add_Layer(layer2)
net.add_Layer(layer3)
net.add_Output_Layer(layer4)


# Add your connections here
relation1 = FullRelation(layer1, layer2)
relation2 = FullRelation(layer2, layer3)
relation3 = FullRelation(layer3, layer4)

# Add your connections to the net here
net.add_Relation(relation1)
net.add_Relation(relation2)
net.add_Relation(relation3)
