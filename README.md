# eip-season4-batch1

Convolution:
Convolution is a process, which results in a single value produced by the weighted sum of the inputs from an interesting region of image/feature maps.

Filters/Kernels:
To perform a convolution operation, weights are required. The structure in which weights are stored with the weights included is called Filter.
            
Epoch:
An epoch is said to be done when the model has gone through all the training samples and updated the necessary weights.

1*1 Convolution:
Structure: 1*1 matrix style, 1 weight in every channel.
This type of kernel is used in networks for organizing the feature maps fthrough channels i.e; combining or seperating the feature maps.
Results in reducing the number of parameters in network.

3*3 Convolution:
Structure: 3*3 matrix style, 9 weight in every channel.
Father of all Kernels because the convolution effect produced by any kernel(kernel_size>3) can be reproduced with lesser number of parameters. Hence, this is the most common kernel used.

Feature Maps:
When convolution is applied through the filters on the image/feature maps, the result obtained is called Feature Maps.

Activation Function:
A function which decides the strength and usefulness of inputs to carry forward the information. Here, it decides which feature value and how much of it should be passed. 

Receptive Field:
The number of features in the input space which influence the interested feature in a feature map can be termed as Receptive Field 
