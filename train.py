from netpy.teachers import BackPropTeacherTest
from network import net
import numpy as np
import conf
from PIL import Image
import random
import math as m

net.load()

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

    return test, canon, num_canon

teacher = BackPropTeacherTest(net,
                              error = 'MSE',
                              learning_rate = 0.001,
                              alpha = 0.0001)

def teach():
    test, canon_arr, num_canon = random_()

    test_pic = Image.fromarray(test.astype('uint8')).convert('L')

    test = test.flatten()/255

    output = net.forward(test, True)*255
    output = np.reshape(output, (32,32))
    output = Image.fromarray(output.astype('uint8')).convert('L')
    output.save('data_set/output.png')
    test_pic.save('data_set/input.png')
    teacher.train(test, canon_arr, load_data = True, name_of_answer = num_canon)

for i in range(100):
    teach()
