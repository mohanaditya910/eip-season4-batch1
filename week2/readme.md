notebook for submission consideration: final_submission_max_12k_LRS_L1.ipynb

model.evaluate(X_test,Y_test) -----------  [0.05891113919615745, 0.9942].....highest val acc 99.48

strategy:
model structure: VGG inspired...

all the techniques like Batch Normalization,Dropout,Learning rate scheduler (provided in the notebook reference is used) in
the notebooks other than submission one..

in submission notebook l1 and different lrs is used. l1 is choosen because it can make weight to 0 if it feels the weight is not 
worthy. i already feel that there are lot of extra parameters because the state of art was with 11k parameters..



As we are going to use regularizer, i thought of checking out with different Learning Rates and 0.04 gave me the best value. So i stuck with it...

Next part is introducing truncated initializer..which according to the keras documentation performs better on weights and filters..generally nature follows bell curve distribution..so went for this...

then started to play with l1 regularizer value...from previous experience i used 3 values for regularizers 0.00005,0.00001,0.0001(99.38,99.34,99.33 val_acc). 0.00005 is in between the other values and its obtained val_acc is greatest among the 3 l1 values , hence its kind of peaking. so better to stay with that.

for all these values, val_acc,train_acc plots in (epochs and acc) the curves appear to be smooth for the last 7-8 epochs.

now the redifining and better using Learning rate scheduler to explore the local minimas exposed by the regularizer.

for the last 10 epochs,decrement of by 10 (lr*0.1) to the learning rate scheduler already provided is done.

this leads to the exploration of only one minima which may or may not contain the required minima.

hence to give the model a chance to better the exploration, i have taken 3 batches of epochs of size 4 ([7,8,9,10],[12,13,14,15],[17,18,19]) and then applied learning rate as obtained above. but for the left over values i have not decremnted by 10 meaning, the value as it is provided by the previous scheduler is taken. this will make the model to explore 3 neighborhood of minimas giving it a better chance to find the best weights

this has given consistent values..of val_acc over 99.4 in epochs 11 to 20...

this can be observed in logs...

logs:

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.004.
60000/60000 [==============================] - 45s 743us/step - loss: 0.2222 - acc: 0.9461 - val_loss: 0.1287 - val_acc: 0.9749
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0030326005.
60000/60000 [==============================] - 38s 627us/step - loss: 0.1239 - acc: 0.9787 - val_loss: 0.1082 - val_acc: 0.9839
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0024420024.
60000/60000 [==============================] - 38s 628us/step - loss: 0.1110 - acc: 0.9822 - val_loss: 0.1037 - val_acc: 0.9855
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0020439448.
60000/60000 [==============================] - 38s 627us/step - loss: 0.1016 - acc: 0.9849 - val_loss: 0.1004 - val_acc: 0.9861
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0017574692.
60000/60000 [==============================] - 38s 626us/step - loss: 0.0931 - acc: 0.9865 - val_loss: 0.0810 - val_acc: 0.9899
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0015414258.
60000/60000 [==============================] - 38s 626us/step - loss: 0.0856 - acc: 0.9880 - val_loss: 0.0819 - val_acc: 0.9898
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0013726836.
60000/60000 [==============================] - 37s 622us/step - loss: 0.0806 - acc: 0.9893 - val_loss: 0.0852 - val_acc: 0.9886
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0001237241.
60000/60000 [==============================] - 37s 619us/step - loss: 0.0660 - acc: 0.9938 - val_loss: 0.0678 - val_acc: 0.9933
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0001126126.
60000/60000 [==============================] - 38s 626us/step - loss: 0.0604 - acc: 0.9952 - val_loss: 0.0652 - val_acc: 0.9935
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0001033325.
60000/60000 [==============================] - 38s 626us/step - loss: 0.0589 - acc: 0.9952 - val_loss: 0.0638 - val_acc: 0.9945
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0009546539.
60000/60000 [==============================] - 37s 623us/step - loss: 0.0717 - acc: 0.9914 - val_loss: 0.0699 - val_acc: 0.9912
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 8.87115e-05.
60000/60000 [==============================] - 37s 621us/step - loss: 0.0590 - acc: 0.9953 - val_loss: 0.0633 - val_acc: 0.9933
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 8.285e-05.
60000/60000 [==============================] - 37s 619us/step - loss: 0.0555 - acc: 0.9959 - val_loss: 0.0620 - val_acc: 0.9940
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 7.77152e-05.
60000/60000 [==============================] - 37s 623us/step - loss: 0.0537 - acc: 0.9960 - val_loss: 0.0613 - val_acc: 0.9936
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 7.31797e-05.
60000/60000 [==============================] - 37s 618us/step - loss: 0.0531 - acc: 0.9963 - val_loss: 0.0609 - val_acc: 0.9939
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0006914434.
60000/60000 [==============================] - 37s 621us/step - loss: 0.0620 - acc: 0.9932 - val_loss: 0.0643 - val_acc: 0.9925
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 6.55308e-05.
60000/60000 [==============================] - 38s 627us/step - loss: 0.0537 - acc: 0.9957 - val_loss: 0.0603 - val_acc: 0.9948
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 6.22762e-05.
60000/60000 [==============================] - 38s 627us/step - loss: 0.0506 - acc: 0.9967 - val_loss: 0.0599 - val_acc: 0.9942
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 5.93296e-05.
60000/60000 [==============================] - 37s 621us/step - loss: 0.0496 - acc: 0.9969 - val_loss: 0.0594 - val_acc: 0.9945
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 5.66492e-05.
60000/60000 [==============================] - 37s 619us/step - loss: 0.0488 - acc: 0.9971 - val_loss: 0.0589 - val_acc: 0.9942



