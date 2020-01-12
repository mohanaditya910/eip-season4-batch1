consider submission file for submission

# Results:

![](images/crop-eip4.png)

# Strategy:

# Data

<img src="images/IMG%20(2902).jpg" width="200">


# Data Augmentation:

As the data provided is very less,Cutout is chosen and implemented because it offers the model to look at the same data in a different way.
For example to make a decision about the gender of a person, face plays a major role. In case of face not shown in the image there are other features like dressing, ornaments etc which can be used to decide.

<img src="images/img-1.png" width="200">
<img src="images/img-2.png" width="200">
<img src="images/img-3.png" width="200">

# Model Architecture:

Model is inspired from RESNET structure and modified with additional skip-connections at the end of very stage to the final layer so that the Model can utilize various receptive field feature maps in making decisions. This is inspired from Path Aggregation Networks.

Finally, instead of Conv2D, Depthwise Conv2D is used as convolutional layer. This type of layers facilitate in reducing the model parameter size upto 9 times. These are used in MOBILENETS.

![](images/arch.png)


 
