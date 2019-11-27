>*Base Model:* 

validation accuracy-83.49..

>*Model Definition*

```
d_m=1
opt=Adam(lr=0.005)

def scheduler(epoch, lr):
  if epoch>=30:
    return round(0.005 * 1/(1 + 0.319 * (epoch-29)), 10)

  else:
    return lr


###############
model1=Sequential()

model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), padding='valid',depth_multiplier=5, activation='relu',input_shape=(32, 32, 3),use_bias=False)) #30,3
model1.add(BatchNormalization())
model1.add(Conv2D(32,1,activation='relu',use_bias=False))  #30,3
model1.add(BatchNormalization())

model1.add(Dropout(0.05))
model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False)) #28,5
model1.add(BatchNormalization())
model1.add(Conv2D(32,1,activation='relu',use_bias=False)) #28,5
model1.add(BatchNormalization())
model1.add(DepthwiseConv2D(kernel_size=3, strides=1, dilation_rate=2,padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False)) #24,9
model1.add(BatchNormalization())
model1.add(Conv2D(64,1,activation='relu',use_bias=False)) #24,9
model1.add(BatchNormalization())

model1.add(Dropout(0.1))
#########################################################################
model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), dilation_rate=2,padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False)) #20,13
model1.add(BatchNormalization())
model1.add(Conv2D(64,1,activation='relu',use_bias=False)) #20,13
model1.add(BatchNormalization())

model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), dilation_rate=2,padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False)) #16,17
model1.add(BatchNormalization())
model1.add(Conv2D(128,1,activation='relu',use_bias=False)) #16,17
model1.add(BatchNormalization())
##########################################################################
model1.add(Dropout(0.1))

model1.add(DepthwiseConv2D(kernel_size=3, strides=1, dilation_rate=2,padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False)) #12,21
model1.add(BatchNormalization())
model1.add(Conv2D(128,1,activation='relu',use_bias=False)) #12,21
model1.add(BatchNormalization())

model1.add(DepthwiseConv2D(kernel_size=3, strides=1, dilation_rate=2,padding='valid',depth_multiplier=d_m, activation='relu',use_bias=False)) #8,25
model1.add(BatchNormalization())
model1.add(Conv2D(128,1,activation='relu',use_bias=False)) #8,25
model1.add(BatchNormalization())


#############################################################################

model1.add(DepthwiseConv2D(kernel_size=3, strides=(1, 1), dilation_rate=2,padding='valid',depth_multiplier=d_m, activation=None,use_bias=False)) #4,29
model1.add(Conv2D(256,1,activation=None,use_bias=False)) #4,29

model1.add(AveragePooling2D(4)) #1,32
model1.add(Conv2D(10,1,use_bias=False)) #1,32
model1.add(Flatten())

model1.add(Activation('softmax'))

model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
```


```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #        Receptive Field.
=================================================================
depthwise_conv2d_1 (Depthwis (None, 30, 30, 15)        135             3
_________________________________________________________________
batch_normalization_1 (Batch (None, 30, 30, 15)        60        
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 30, 30, 32)        480             3
_________________________________________________________________
batch_normalization_2 (Batch (None, 30, 30, 32)        128             
_________________________________________________________________
dropout_6 (Dropout)          (None, 30, 30, 32)        0         
_________________________________________________________________
depthwise_conv2d_2 (Depthwis (None, 28, 28, 32)        288              5
_________________________________________________________________
batch_normalization_3 (Batch (None, 28, 28, 32)        128       
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 28, 28, 32)        1024             5
_________________________________________________________________
batch_normalization_4 (Batch (None, 28, 28, 32)        128       
_________________________________________________________________
depthwise_conv2d_3 (Depthwis (None, 24, 24, 32)        288              9
_________________________________________________________________
batch_normalization_5 (Batch (None, 24, 24, 32)        128       
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 24, 24, 64)        2048             9
_________________________________________________________________
batch_normalization_6 (Batch (None, 24, 24, 64)        256       
_________________________________________________________________
dropout_7 (Dropout)          (None, 24, 24, 64)        0         
_________________________________________________________________
depthwise_conv2d_4 (Depthwis (None, 20, 20, 64)        576              13
_________________________________________________________________
batch_normalization_7 (Batch (None, 20, 20, 64)        256       
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 20, 20, 64)        4096             13
_________________________________________________________________ 
batch_normalization_8 (Batch (None, 20, 20, 64)        256       
_________________________________________________________________
depthwise_conv2d_5 (Depthwis (None, 16, 16, 64)        576              17
_________________________________________________________________
batch_normalization_9 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 16, 16, 128)       8192             17
_________________________________________________________________
batch_normalization_10 (Batc (None, 16, 16, 128)       512       
_________________________________________________________________
dropout_8 (Dropout)          (None, 16, 16, 128)       0         
_________________________________________________________________
depthwise_conv2d_6 (Depthwis (None, 12, 12, 128)       1152             21
_________________________________________________________________
batch_normalization_11 (Batc (None, 12, 12, 128)       512       
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 12, 12, 128)       16384            21
_________________________________________________________________
batch_normalization_12 (Batc (None, 12, 12, 128)       512       
_________________________________________________________________
depthwise_conv2d_7 (Depthwis (None, 8, 8, 128)         1152             25
_________________________________________________________________
batch_normalization_13 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 8, 8, 128)         16384            25
_________________________________________________________________
batch_normalization_14 (Batc (None, 8, 8, 128)         512       
_________________________________________________________________
depthwise_conv2d_8 (Depthwis (None, 4, 4, 128)         1152             29
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 4, 4, 256)         32768            29
_________________________________________________________________
average_pooling2d_1 (Average (None, 1, 1, 256)         0                32
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 1, 1, 10)          2560             32
_________________________________________________________________
flatten_2 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_9 (Activation)    (None, 10)                0         
=================================================================
Total params: 93,411
Trainable params: 91,333
Non-trainable params: 2,078
_________________________________________________________________
```


>*Logs:*

```
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 60s 153ms/step - loss: 1.4614 - acc: 0.4709 - val_loss: 2.1885 - val_acc: 0.4970
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 1.1419 - acc: 0.5950 - val_loss: 2.5977 - val_acc: 0.4250
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 1.0036 - acc: 0.6460 - val_loss: 1.2659 - val_acc: 0.6115
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.9174 - acc: 0.6784 - val_loss: 1.0675 - val_acc: 0.6507
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.8651 - acc: 0.6968 - val_loss: 1.4204 - val_acc: 0.5768
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.8224 - acc: 0.7115 - val_loss: 1.2008 - val_acc: 0.6304
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.7890 - acc: 0.7260 - val_loss: 1.0834 - val_acc: 0.6495
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.7587 - acc: 0.7356 - val_loss: 1.0406 - val_acc: 0.6749
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.7354 - acc: 0.7450 - val_loss: 1.0184 - val_acc: 0.6890
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.7189 - acc: 0.7512 - val_loss: 0.7401 - val_acc: 0.7410
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.6969 - acc: 0.7565 - val_loss: 0.9737 - val_acc: 0.7066
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 144ms/step - loss: 0.6859 - acc: 0.7610 - val_loss: 0.7976 - val_acc: 0.7278
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.6690 - acc: 0.7663 - val_loss: 0.8909 - val_acc: 0.7173
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.6516 - acc: 0.7720 - val_loss: 0.7439 - val_acc: 0.7500
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.6383 - acc: 0.7775 - val_loss: 0.8034 - val_acc: 0.7388
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 144ms/step - loss: 0.6380 - acc: 0.7781 - val_loss: 1.0177 - val_acc: 0.6923
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.6227 - acc: 0.7841 - val_loss: 0.7814 - val_acc: 0.7515
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.6117 - acc: 0.7885 - val_loss: 0.8534 - val_acc: 0.7390
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.6066 - acc: 0.7893 - val_loss: 0.8279 - val_acc: 0.7393
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 144ms/step - loss: 0.5986 - acc: 0.7913 - val_loss: 0.7694 - val_acc: 0.7496
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5869 - acc: 0.7963 - val_loss: 0.9332 - val_acc: 0.7097
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5815 - acc: 0.7974 - val_loss: 0.9044 - val_acc: 0.7303
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5702 - acc: 0.7999 - val_loss: 0.8199 - val_acc: 0.7400
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5749 - acc: 0.7989 - val_loss: 0.6660 - val_acc: 0.7776
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5606 - acc: 0.8057 - val_loss: 0.6708 - val_acc: 0.7803
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5564 - acc: 0.8056 - val_loss: 0.6841 - val_acc: 0.7813
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5521 - acc: 0.8064 - val_loss: 0.7576 - val_acc: 0.7663
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5417 - acc: 0.8113 - val_loss: 0.6611 - val_acc: 0.7888
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5421 - acc: 0.8104 - val_loss: 1.0344 - val_acc: 0.6904
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.004999999888241291.
390/390 [==============================] - 56s 143ms/step - loss: 0.5337 - acc: 0.8138 - val_loss: 0.7148 - val_acc: 0.7805
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0037907506.
390/390 [==============================] - 56s 143ms/step - loss: 0.5120 - acc: 0.8218 - val_loss: 0.6956 - val_acc: 0.7801
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0030525031.
390/390 [==============================] - 56s 143ms/step - loss: 0.4829 - acc: 0.8284 - val_loss: 0.6300 - val_acc: 0.7975
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.002554931.
390/390 [==============================] - 56s 143ms/step - loss: 0.4774 - acc: 0.8320 - val_loss: 0.6423 - val_acc: 0.7929
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0021968366.
390/390 [==============================] - 56s 144ms/step - loss: 0.4576 - acc: 0.8403 - val_loss: 0.5757 - val_acc: 0.8142
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0019267823.
390/390 [==============================] - 56s 143ms/step - loss: 0.4468 - acc: 0.8440 - val_loss: 0.5799 - val_acc: 0.8104
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0017158545.
390/390 [==============================] - 56s 144ms/step - loss: 0.4442 - acc: 0.8435 - val_loss: 0.5398 - val_acc: 0.8205
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0015465512.
390/390 [==============================] - 56s 144ms/step - loss: 0.4384 - acc: 0.8451 - val_loss: 0.6038 - val_acc: 0.8045
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0014076577.
390/390 [==============================] - 56s 144ms/step - loss: 0.4295 - acc: 0.8499 - val_loss: 0.5115 - val_acc: 0.8332
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0012916559.
390/390 [==============================] - 56s 144ms/step - loss: 0.4225 - acc: 0.8541 - val_loss: 0.5410 - val_acc: 0.8242
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0011933174.
390/390 [==============================] - 56s 144ms/step - loss: 0.4210 - acc: 0.8514 - val_loss: 0.5546 - val_acc: 0.8187
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0011088933.
390/390 [==============================] - 56s 144ms/step - loss: 0.4118 - acc: 0.8542 - val_loss: 0.5199 - val_acc: 0.8325
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0010356255.
390/390 [==============================] - 56s 143ms/step - loss: 0.4099 - acc: 0.8562 - val_loss: 0.5853 - val_acc: 0.8173
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0009714397.
390/390 [==============================] - 56s 143ms/step - loss: 0.4100 - acc: 0.8570 - val_loss: 0.5185 - val_acc: 0.8332
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0009147457.
390/390 [==============================] - 56s 143ms/step - loss: 0.4020 - acc: 0.8562 - val_loss: 0.5683 - val_acc: 0.8190
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0008643042.
390/390 [==============================] - 56s 143ms/step - loss: 0.4008 - acc: 0.8597 - val_loss: 0.5211 - val_acc: 0.8307
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.000819135.
390/390 [==============================] - 56s 143ms/step - loss: 0.4008 - acc: 0.8588 - val_loss: 0.5382 - val_acc: 0.8261
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0007784524.
390/390 [==============================] - 56s 143ms/step - loss: 0.3986 - acc: 0.8600 - val_loss: 0.5287 - val_acc: 0.8273
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0007416197.
390/390 [==============================] - 56s 143ms/step - loss: 0.3913 - acc: 0.8613 - val_loss: 0.5148 - val_acc: 0.8319
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.000708115.
390/390 [==============================] - 56s 143ms/step - loss: 0.3939 - acc: 0.8603 - val_loss: 0.5037 - val_acc: 0.8352
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0006775068.
390/390 [==============================] - 56s 143ms/step - loss: 0.3914 - acc: 0.8625 - val_loss: 0.5015 - val_acc: 0.8333
Model took 2801.64 seconds to train

Accuracy on test data is: 83.33

Accuracy of 49th epoch on test data is: 83.52
```


