### Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability


Project Organization
------------
    ├── README.md   
    ├── requirements.txt 
    ├── utils.py         
    ├── input_saliency.py  <- main script that computes both input saliency and parameter saliency
    │
    ├── figures <- folder for resulting figures
    ├── saliency
        └── saliency_model_backprop  <- script with SaliencyModel class, parameter saliency implementation 

Basic Use
---------
python3 input_saliency.py --reference_id 107 --k_salient 10

here --reference_id specifies the image id from ImageNet validation set

--k_salient specifies the number of top salient filters to use

The resulting plots (input space colormap and filter saliency plot) will be saved to /figures