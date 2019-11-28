Team: Tigers
Members: Mohanaditya R Reddi, Sucharitha V

>*Base Model:* 

validation accuracy-83.49..

>*MODEL DEFINITION:*
```
from keras.regularizers import l2
d_m=1
opt=Adam(lr=0.005)
l2val=0.0001
def scheduler(epoch, lr):
  if epoch>=30:
    return round(0.005 * 1/(1 + 0.319 * (epoch-29)), 10)

  else:
    return lr
'''    
def scheduler(epoch, lr):
  if epoch<=20:
    return round(0.0005 * 1/(1 + 0.319 * epoch), 10)
  elif epoch%5==0:
    return round(0.0005 * 1/(1 + 0.319 * epoch), 10)
  else:
    return round(0.0004 * 1/(1 + 0.319 * epoch), 10)
'''
###############
model1=Sequential()
#add dropout..........
model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), padding='same',depth_multiplier=5, activation='relu',input_shape=(32, 32, 3),use_bias=False,depthwise_regularizer=l2(l2val)))  #32,3
model1.add(BatchNormalization())
model1.add(Conv2D(32,1,activation='relu',use_bias=False,kernel_regularizer=l2(l2val)))                                                                                                   #32,3
model1.add(BatchNormalization())

model1.add(Dropout(0.05))
model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), padding='same',depth_multiplier=5, activation='relu',use_bias=False,depthwise_regularizer=l2(l2val)))                          #32,5
model1.add(BatchNormalization())
model1.add(Conv2D(32,1,activation='relu',use_bias=False,kernel_regularizer=l2(l2val)))                                                                                                   #32,5
model1.add(BatchNormalization())
model1.add(DepthwiseConv2D(kernel_size=3, strides=2, dilation_rate=1,padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False,depthwise_regularizer=l2(l2val)))            #15,7
model1.add(BatchNormalization())
model1.add(Conv2D(64,1,activation='relu',use_bias=False,kernel_regularizer=l2(l2val)))                                                                                                   #15,7
model1.add(BatchNormalization())

model1.add(Dropout(0.1))

model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), dilation_rate=1,padding='same',depth_multiplier=d_m, activation='relu',use_bias=False,depthwise_regularizer=l2(l2val)))        #15,11
model1.add(BatchNormalization())
model1.add(Conv2D(64,1,activation='relu',use_bias=False,kernel_regularizer=l2(l2val)))                                                                                                   #15,11
model1.add(BatchNormalization())

model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), dilation_rate=1,padding='same',depth_multiplier=d_m, activation='relu',use_bias=False,depthwise_regularizer=l2(l2val)))        #15,15
model1.add(BatchNormalization())
model1.add(Conv2D(64,1,activation='relu',use_bias=False,kernel_regularizer=l2(l2val)))                                                                                                   #15,15
model1.add(BatchNormalization())

model1.add(Dropout(0.1))

model1.add(DepthwiseConv2D(kernel_size=3, strides=2, dilation_rate=1,padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False,depthwise_regularizer=l2(l2val)))            #7,19
model1.add(BatchNormalization())

model1.add(Conv2D(128,1,activation='relu',use_bias=False,kernel_regularizer=l2(l2val)))                                                                                                  #7,19
model1.add(BatchNormalization())

model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), dilation_rate=1,padding='same',depth_multiplier=d_m, activation=None,use_bias=False,depthwise_regularizer=l2(l2val)))          #7,27
model1.add(Conv2D(256,1,activation=None,use_bias=False,kernel_regularizer=l2(l2val)))                                                                                                    #7,27

model1.add(AveragePooling2D(7))                                                                                                                                                           #1,51
model1.add(Conv2D(10,1,use_bias=False))                                                                                                                                                   #1,51
model1.add(Flatten())

model1.add(Activation('softmax'))

model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
```

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
depthwise_conv2d_9 (Depthwis (None, 32, 32, 15)        135       
_________________________________________________________________
batch_normalization_15 (Batc (None, 32, 32, 15)        60        
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 32, 32, 32)        480       
_________________________________________________________________
batch_normalization_16 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
dropout_4 (Dropout)          (None, 32, 32, 32)        0         
_________________________________________________________________
depthwise_conv2d_10 (Depthwi (None, 32, 32, 160)       1440      
_________________________________________________________________
batch_normalization_17 (Batc (None, 32, 32, 160)       640       
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 32, 32, 32)        5120      
_________________________________________________________________
batch_normalization_18 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
depthwise_conv2d_11 (Depthwi (None, 15, 15, 32)        288       
_________________________________________________________________
batch_normalization_19 (Batc (None, 15, 15, 32)        128       
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 15, 15, 64)        2048      
_________________________________________________________________
batch_normalization_20 (Batc (None, 15, 15, 64)        256       
_________________________________________________________________
dropout_5 (Dropout)          (None, 15, 15, 64)        0         
_________________________________________________________________
depthwise_conv2d_12 (Depthwi (None, 15, 15, 64)        576       
_________________________________________________________________
batch_normalization_21 (Batc (None, 15, 15, 64)        256       
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 15, 15, 64)        4096      
_________________________________________________________________
batch_normalization_22 (Batc (None, 15, 15, 64)        256       
_________________________________________________________________
depthwise_conv2d_13 (Depthwi (None, 15, 15, 64)        576       
_________________________________________________________________
batch_normalization_23 (Batc (None, 15, 15, 64)        256       
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 15, 15, 64)        4096      
_________________________________________________________________
batch_normalization_24 (Batc (None, 15, 15, 64)        256       
_________________________________________________________________
dropout_6 (Dropout)          (None, 15, 15, 64)        0         
_________________________________________________________________
depthwise_conv2d_14 (Depthwi (None, 7, 7, 64)          576       
_________________________________________________________________
batch_normalization_25 (Batc (None, 7, 7, 64)          256       
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 7, 7, 128)         8192      
_________________________________________________________________
batch_normalization_26 (Batc (None, 7, 7, 128)         512       
_________________________________________________________________
depthwise_conv2d_15 (Depthwi (None, 7, 7, 128)         1152      
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 7, 7, 256)         32768     
_________________________________________________________________
average_pooling2d_2 (Average (None, 1, 1, 256)         0         
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 1, 1, 10)          2560      
_________________________________________________________________
flatten_2 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0         
=================================================================
Total params: 67,235
Trainable params: 65,669
Non-trainable params: 1,566
_________________________________________________________________
```


>*LOGS:*
```
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:21: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:21: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., callbacks=[<keras.ca..., validation_data=(array([[[..., verbose=1, steps_per_epoch=97, epochs=50)`

Epoch 1/50

/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py:716: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
  warnings.warn('This ImageDataGenerator specifies '
/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py:724: UserWarning: This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
  warnings.warn('This ImageDataGenerator specifies '


Epoch 00001: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 50s 517ms/step - loss: 1.5647 - acc: 0.4405 - val_loss: 6.5255 - val_acc: 0.2425
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 483ms/step - loss: 1.1872 - acc: 0.5909 - val_loss: 5.5625 - val_acc: 0.2803
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 482ms/step - loss: 1.0474 - acc: 0.6470 - val_loss: 2.5029 - val_acc: 0.4488
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 486ms/step - loss: 0.9564 - acc: 0.6817 - val_loss: 2.2782 - val_acc: 0.4771
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 486ms/step - loss: 0.9008 - acc: 0.7037 - val_loss: 1.8261 - val_acc: 0.5232
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 483ms/step - loss: 0.8481 - acc: 0.7237 - val_loss: 1.8417 - val_acc: 0.5306
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 482ms/step - loss: 0.8083 - acc: 0.7405 - val_loss: 1.2038 - val_acc: 0.6549
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 479ms/step - loss: 0.7876 - acc: 0.7485 - val_loss: 1.6157 - val_acc: 0.5844
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 486ms/step - loss: 0.7506 - acc: 0.7643 - val_loss: 1.1499 - val_acc: 0.6820
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 482ms/step - loss: 0.7396 - acc: 0.7686 - val_loss: 1.4272 - val_acc: 0.6070
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 479ms/step - loss: 0.7245 - acc: 0.7742 - val_loss: 1.0775 - val_acc: 0.6841
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 477ms/step - loss: 0.7065 - acc: 0.7828 - val_loss: 0.9441 - val_acc: 0.7269
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 486ms/step - loss: 0.6988 - acc: 0.7860 - val_loss: 1.2842 - val_acc: 0.6610
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 476ms/step - loss: 0.6803 - acc: 0.7889 - val_loss: 1.0519 - val_acc: 0.6934
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 484ms/step - loss: 0.6735 - acc: 0.7938 - val_loss: 0.8298 - val_acc: 0.7638
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 480ms/step - loss: 0.6707 - acc: 0.7963 - val_loss: 0.9680 - val_acc: 0.7122
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 477ms/step - loss: 0.6544 - acc: 0.8013 - val_loss: 0.8488 - val_acc: 0.7571
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 479ms/step - loss: 0.6545 - acc: 0.8023 - val_loss: 0.7710 - val_acc: 0.7682
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 485ms/step - loss: 0.6474 - acc: 0.8039 - val_loss: 0.9866 - val_acc: 0.7361
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 474ms/step - loss: 0.6449 - acc: 0.8061 - val_loss: 0.8078 - val_acc: 0.7701
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 482ms/step - loss: 0.6361 - acc: 0.8081 - val_loss: 0.8274 - val_acc: 0.7515
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 478ms/step - loss: 0.6402 - acc: 0.8078 - val_loss: 1.1117 - val_acc: 0.7013
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 472ms/step - loss: 0.6280 - acc: 0.8113 - val_loss: 0.8686 - val_acc: 0.7471
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 479ms/step - loss: 0.6212 - acc: 0.8149 - val_loss: 0.8696 - val_acc: 0.7520
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 475ms/step - loss: 0.6115 - acc: 0.8162 - val_loss: 0.8347 - val_acc: 0.7566
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 47s 482ms/step - loss: 0.6063 - acc: 0.8200 - val_loss: 1.2197 - val_acc: 0.6599
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 474ms/step - loss: 0.6078 - acc: 0.8199 - val_loss: 0.8351 - val_acc: 0.7609
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 479ms/step - loss: 0.6075 - acc: 0.8200 - val_loss: 0.8876 - val_acc: 0.7515
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 474ms/step - loss: 0.5985 - acc: 0.8243 - val_loss: 1.0555 - val_acc: 0.7185
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.004999999888241291.
97/97 [==============================] - 46s 477ms/step - loss: 0.5996 - acc: 0.8237 - val_loss: 0.7312 - val_acc: 0.7930
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0037907506.
97/97 [==============================] - 46s 474ms/step - loss: 0.5759 - acc: 0.8325 - val_loss: 0.7379 - val_acc: 0.7914
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0030525031.
97/97 [==============================] - 46s 476ms/step - loss: 0.5514 - acc: 0.8402 - val_loss: 0.7033 - val_acc: 0.7965
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.002554931.
97/97 [==============================] - 46s 476ms/step - loss: 0.5378 - acc: 0.8438 - val_loss: 0.6372 - val_acc: 0.8188
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0021968366.
97/97 [==============================] - 46s 473ms/step - loss: 0.5249 - acc: 0.8487 - val_loss: 0.6511 - val_acc: 0.8099
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0019267823.
97/97 [==============================] - 46s 471ms/step - loss: 0.5073 - acc: 0.8537 - val_loss: 0.5857 - val_acc: 0.8308
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0017158545.
97/97 [==============================] - 46s 475ms/step - loss: 0.4949 - acc: 0.8594 - val_loss: 0.5538 - val_acc: 0.8425
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0015465512.
97/97 [==============================] - 46s 478ms/step - loss: 0.5009 - acc: 0.8551 - val_loss: 0.6087 - val_acc: 0.8276
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0014076577.
97/97 [==============================] - 46s 473ms/step - loss: 0.4869 - acc: 0.8593 - val_loss: 0.6172 - val_acc: 0.8246
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0012916559.
97/97 [==============================] - 46s 472ms/step - loss: 0.4813 - acc: 0.8631 - val_loss: 0.5996 - val_acc: 0.8315
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0011933174.
97/97 [==============================] - 46s 475ms/step - loss: 0.4762 - acc: 0.8615 - val_loss: 0.5694 - val_acc: 0.8373
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0011088933.
97/97 [==============================] - 45s 468ms/step - loss: 0.4688 - acc: 0.8652 - val_loss: 0.5786 - val_acc: 0.8370
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0010356255.
97/97 [==============================] - 46s 470ms/step - loss: 0.4671 - acc: 0.8656 - val_loss: 0.5983 - val_acc: 0.8301
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0009714397.
97/97 [==============================] - 46s 470ms/step - loss: 0.4568 - acc: 0.8698 - val_loss: 0.5673 - val_acc: 0.8376
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0009147457.
97/97 [==============================] - 46s 476ms/step - loss: 0.4610 - acc: 0.8679 - val_loss: 0.5461 - val_acc: 0.8459
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0008643042.
97/97 [==============================] - 46s 472ms/step - loss: 0.4515 - acc: 0.8710 - val_loss: 0.5647 - val_acc: 0.8440
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.000819135.
97/97 [==============================] - 46s 470ms/step - loss: 0.4569 - acc: 0.8692 - val_loss: 0.5857 - val_acc: 0.8358
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0007784524.
97/97 [==============================] - 45s 462ms/step - loss: 0.4500 - acc: 0.8699 - val_loss: 0.5929 - val_acc: 0.8306
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0007416197.
97/97 [==============================] - 45s 466ms/step - loss: 0.4425 - acc: 0.8714 - val_loss: 0.5418 - val_acc: 0.8465
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.000708115.
97/97 [==============================] - 45s 468ms/step - loss: 0.4418 - acc: 0.8722 - val_loss: 0.5443 - val_acc: 0.8472
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0006775068.
97/97 [==============================] - 45s 465ms/step - loss: 0.4400 - acc: 0.8738 - val_loss: 0.5820 - val_acc: 0.8383
Model took 2317.52 seconds to train

Accuracy on test data is: 83.83
Best accuracy on test data is: 84.72

```
