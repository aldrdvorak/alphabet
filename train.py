from netpy.teachers import BackPropTeacherTest
from network import net
import numpy as np
import conf
from PIL import Image
import random
import math as m

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

def random_pic(num_canon):
    phi = random.uniform(-1, 1)
    scale = random.uniform(0.6, 1.4)

    canon = Image.open("canon/can_"+str(num_canon)+".png").convert('L')
    canon = np.array(canon)

    test = deform_pic(canon, phi, scale)

    canon = canon.flatten()/255
    test = test.flatten()/255


    return test, canon

def create_data_set(num_data):
    j = 0
    
    X = []
    Y = []
    name_of_answer_arr = []


    while j < num_data:
        for i in range(0, 36):
            test, canon = random_pic(i)

            X.append(test)
            Y.append(test)
            name_of_answer_arr.append(i)
        
            j += 1

    return X, Y, name_of_answer_arr

teacher = BackPropTeacherTest(net,
                              error = 'MSE',
                              learning_rate = 0.001,
                              alpha = 0.0001)

for i in range(10):
    X, Y, name_of_answer_array = create_data_set(100)

    teacher.train(X, Y, 100, load_data = True, 
                             name_of_answer = name_of_answer_array,
                             random_data = False,
                             save_output_data = True)
