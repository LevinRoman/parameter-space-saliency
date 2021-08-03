### Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability
This repository contains the implementation of parameter-saliency methods from our paper "Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability".

Getting started
---------
<a href = "https://www.python.org/downloads/">Python 3</a>
```bash
pip install -r requirements.txt
```

Basic Use
---------
python3 input_saliency.py --reference_id 107 --k_salient 10

here --reference_id specifies the image id from ImageNet validation set

--k_salient specifies the number of top salient filters to use

The resulting plots (input space colormap and filter saliency plot) will be saved to /figures


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
