from network import net
from netpy.tools.functions import iq_test
import numpy as np
import conf
from PIL import Image
import logging
import math as m
import datetime
import random

def deform_pic(image, phi, coff_scale):
    new_image = np.zeros((32,32))

    for i in range(0, 32):
        for j in range(0, 32):
            new_image[i][j] = 255

    for i in range(0, 32):
        for j in range(0, 32):
            
            x0 = i - 15.5
            y0 = j - 15.5

            x1 = int(((x0*m.cos(phi) - y0*m.sin(phi))/coff_scale) + 15.5)
            y1 = int(((x0*m.sin(phi) + y0*m.cos(phi))/coff_scale) + 15.5)

            if x1 < 0:
                x1 = 0
            elif x1 > 31:
                x1 = 31

            if y1 < 0:
                y1 = 0
            elif y1 > 31:
                y1 = 0

            new_image[i][j] = image[x1][y1]

    image = new_image
    return image

def random_():
    phi = random.uniform(-1, 1)
    scale = random.uniform(0.6, 1.4)

    num_canon = int(random.uniform(0,36))

    canon = Image.open("canon/can_"+str(num_canon)+".png").convert('L')
    canon = np.array(canon)

    test = deform_pic(canon, phi, scale)

    canon = canon.flatten()/255

    return test, canon

'''
def iq_test(net, X, Y):

    logging.basicConfig(filename=net.name+'_data/'+net.name+'.log', 
                       level=logging.INFO)

    output = []
    errors = np.array([])

    for i in range(0, len(X)):
        output.append(net.forward(X[i], training=True))

    for i in range(0, len(X)):
        errors = np.hstack((errors, abs(Y[i] - output[i])))


    err_ = sum(errors)/len(errors)
    sigma = m.sqrt((sum((errors - err_)** 2)/len(errors)))
    err_max = err_ + 3*sigma
    
    if err_max > 1:
        err_max = 1

    err_min = err_ - 3*sigma

    if err_max < 0:
        err_max = 0

    iq_min = (1 - err_max)*100
    iq_max = (1 - err_min)*100
    iq = (iq_min + iq_max)/2

    logging.info("%s | iq_test | iq: %s | iq_min: %s | iq_max: %s | nn_IQ: %s" %
                 (datetime.datetime.today().strftime("%Y-%m-%d-%H.%M.%S"),
                  iq, iq_min, iq_max, len(X)))

    return iq, iq_min, iq_max'''




net.load()

Y_ = []
X_ = []


for i in range(0, 1000):
    X, Y = random_()
   # Y = Image.open("canon/can_"+str(i)+".png").convert("L")
   # Y = np.array(Y)
   # Y = Y.flatten()/255
    
    Y_.append(Y)
    X_.append(net.forward(Y, training = True))


print(iq_test(net, X_, Y_))






