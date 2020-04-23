# Convolutional Neural Networks (CNN) *fruit_classification*


Fruit Classification Using Pytorch and Tensor flow



# Convolutional Neural Networks (CNN)

In this lesson we will explore the basics of Convolutional Neural Networks (CNNs )

### List of defaults:


 - `batch_size`: this is how many examples to train on in one batch.
 - `data_dir`: where to store data (check if data exists here, as to not have to download every time).
 - `output_every`: output training accuracy/loss statistics every X generations/epochs.
 - `eval_every`: output test accuracy/loss statistics every X generations/epochs.
 - `image_height`: standardize images to this height.
 - `image_width`: standardize images to this width.
 - `crop_height`: random internal crop before training on image - height.
 - `crop_width`: random internal crop before training on image - width.
 - `num_channels`: number of color channels of image (greyscale = 1, color = 3).
 - `num_targets`: number of different target categories. 
 - `extract_folder`: folder to extract downloaded images to.



<div align="left">
<img src="https://raw.githubusercontent.com/madewithml/images/master/02_Basics/07_Convolutional_Neural_Networks/convolution.gif" width="500">
</div>

* **stride**: amount the filters move from one convolution operation to the next.
* **padding**: values (typically zero) padded to the input, typically to create a volume with whole number dimensions.