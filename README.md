### Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability
This repository contains the implementation of parameter-saliency methods from our paper <a href = "https://arxiv.org/pdf/2108.01335.pdf">Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability </a>. 

Abstract:
Conventional saliency maps highlight input features to which neural network predictions are highly sensitive. We take a different approach to saliency, in which we identify and analyze the network parameters, rather than inputs, which are responsible for erroneous decisions.  We find that samples which cause similar parameters to malfunction are semantically similar. We also show that pruning the most salient parameters for a wrongly classified sample often improves model behavior. Furthermore, fine-tuning a small number of the most salient parameters on a single sample results in error correction on other samples that are misclassified for similar reasons. Based on our parameter saliency method, we also introduce an input-space saliency technique that reveals how image features cause specific network components to malfunction.  Further, we rigorously validate the meaningfulness of our saliency maps on both the dataset and case-study levels.

Getting started
---------------
This repo uses <a href = "https://www.python.org/downloads/">Python 3</a>. To install the requirements, run
```bash
pip install -r requirements.txt
```

Basic Use
---------
The script input_saliency.py computes both the parameter-saliency profile of an image which allows to find misbehaving filters in a neural network responsible for misclassification of a given image. In addition, the script computes the image-space saliency which highlights pixels which drive the high filter saliency values.

To compute the parameter saliency profile for a given image, the script accepts 
* either path to the raw image + image target label
```bash
python3 parameter_and_input_saliency.py --model resnet50 --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg --image_target_label 2
```
* or reference_id -- the index of the given image in ImageNet validation set.
```bash
python3 input_saliency.py --reference_id 107 --k_salient 10
```

here --reference_id specifies the image id from ImageNet validation set

--k_salient specifies the number of top salient filters to use

The resulting plots (input space colormap and filter saliency plot) will be saved to /figures

Demo
-----
The demo raw image is in /raw_images. The results are in /figures.

Project Organization
------------
    ├── README.md
    ├── LICENSE
    ├── requirements.txt 
    ├── utils.py  <- helper functions       
    ├── parameter_and_input_saliency.py  <- main script which computes both input saliency and parameter saliency
    │
    ├── figures <- folder for resulting figures
    ├── helper_objects <- precomputed objects to speed up computation (inference results on ImageNet valset and parameter saliency mean and std for standardization)
    │   ├─ resnet50   
    │   ├─ densenet121
    │   ├─ inception_v3
    │   └── vgg19
    ├── raw_images <- images to use for parameter space saliency computation and for input space saliency visualization
    └── parameter_saliency
        └── saliency_model_backprop  <- script with SaliencyModel class, parameter saliency implementation 
