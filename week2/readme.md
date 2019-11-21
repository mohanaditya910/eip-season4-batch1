i have written my skeleton model(only Conv2D layers) like a VGG conv block for solving the problem.....using this i tested this out for different configurations of 
number of kernels...i found that having (7,14,21) (10k params)or (8,16,24) (12k params) in each conv block gave me the best results...

then i applied Bacth Normalization,Dropouts for the next step..which gave push to my model...
subsequently applied Learning rate scheduler... which brought to a stage where my validation accuracy is getting stuck in 99.2's.

So,i decided to explore more regularization techniques and found out l1,l2...
I found that l2 curve is pulling down the train accuracy cure down and slightly increasing the validation accuracy curve. The curve is 
becoming very smooth. Anyways, it is bringing my train accuracy down to 99.2 so it is not useful here..

Coming to l1 it has a property to make weights 0 if they are not extracting useful information..so i wanted to eliminate influence of 
any influence from unwanted kernels. So, I concentrated on l1 regularizer.

finding the right l1 value took time but it boosted the val_accuracy to 99.37....

finally data augmentation provided val_accuracy above 99.4....

