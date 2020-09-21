# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import os
import pickle

pickle_results = open("../../models/Xbs_default_together.pickle","rb")
r = pickle.load(pickle_results)

Xb_train = r[0]
Xb_test = r[1]

img_rows, img_cols = 30, 184
input_shape = (img_cols, img_rows,1)

train_images = np.around(Xb_train[0]/255, decimals = 2)
test_images = np.around(Xb_test[0]/255, decimals = 2)
x_train = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)


train_images1 = np.around(Xb_train[1]/255, decimals = 2)
test_images1 = np.around(Xb_test[1]/255, decimals = 2)
x_train1 = train_images1.reshape(train_images1.shape[0], img_rows, img_cols, 1)


train_images2 = np.around(Xb_train[2]/255, decimals = 2)
test_images2 = np.around(Xb_test[2]/255, decimals = 2)
x_train2 = train_images2.reshape(train_images2.shape[0], img_rows, img_cols, 1)


train_images3 = np.around(Xb_train[3]/255, decimals = 2)
test_images3 = np.around(Xb_test[3]/255, decimals = 2)
x_train3 = train_images3.reshape(train_images3.shape[0], img_rows, img_cols, 1)


train_images4 = np.around(Xb_train[4]/255, decimals = 2)
test_images4 = np.around(Xb_test[4]/255, decimals = 2)
x_train4 = train_images4.reshape(train_images4.shape[0], img_rows, img_cols, 1)


pickle_out = open("../../models/x_train0_and_test_images.pickle","wb")
r = [x_train,test_images]
pickle.dump(r, pickle_out)
pickle_out.close()

pickle_out = open("../../models/x_train1_and_test_images.pickle","wb")
r = [x_train1,test_images1]
pickle.dump(r, pickle_out)
pickle_out.close()

pickle_out = open("../../models/x_train2_and_test_images.pickle","wb")
r = [x_train2,test_images2]
pickle.dump(r, pickle_out)
pickle_out.close()

pickle_out = open("../../models/x_train3_and_test_images.pickle","wb")
r = [x_train3,test_images3]
pickle.dump(r, pickle_out)
pickle_out.close()

pickle_out = open("../../models/x_train4_and_test_images.pickle","wb")
r = [x_train4,test_images4]
pickle.dump(r, pickle_out)
pickle_out.close()