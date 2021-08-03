# from comet_ml import Experiment
import yaml
import urllib
import pandas as pd
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import show_heatmap_on_image, test_and_find_incorrectly_classified, transform_raw_image
import cv2
import warnings
import tqdm
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency

parser = argparse.ArgumentParser(description='Input Space Saliency')
parser.add_argument('--model', default='resnet50', type=str, help='name of architecture')
parser.add_argument('--data_to_use', default='ImageNet', type=str, help='which dataset to use (ImageNet or ImageNet_A)')

# Logging
# parser.add_argument('--project_name', default='input_space_saliency', type=str, help='project name for Comet ML')
parser.add_argument('--figure_folder_name', default='input_space_saliency', type=str, help='directory to save figures')

# Modes for the signed saliency model: by default, regular loss on the given example is used.
#All final experiments were done with the following options off
parser.add_argument('--signed', action='store_true', help='Use signed saliency')
parser.add_argument('--logit', action='store_true', help='Use logits to compute parameter saliency')
parser.add_argument('--logit_difference', action='store_true', help='Use logit difference as parameter saliency loss')


#Boosting for input-space saliency
parser.add_argument('--boost_factor', default=100.0, type=float, help='boost factor for salient filters')
parser.add_argument('--k_salient', default=10, type=int, help='num filters to boost')

parser.add_argument('--compare_random', action='store_true',
                    help='whether to boost k random filters for comparison')
# parser.add_argument('--least_salient', action='store_true',
#                     help='whether to boost k least salient filters for comparison to frying most salient')

#Smoothing input space saliency (SmoothGrad-like, should be set to default, off at all times)
parser.add_argument('--noise_iters', default=1, type=int, help='number of noises to average across')
parser.add_argument('--noise_percent', default=0, type=float, help='std of the noises')

#Pick reference image
#Either using an image from raw_images/ folder
parser.add_argument('--image_path', default='raw_images/great_white_shark_mispred_as_killer_whale.jpeg', type=str, help='image id from valset to use')
parser.add_argument('--image_target_label', default=None, type=int, help='image id from valset to use')
#Or using the i-th image from ImageNet validation set, for this ImageNet validation set path must be specified
parser.add_argument('--reference_id', default=None, type=int, help='image id from valset to use') #107 for great white shark

#PATHS
parser.add_argument('--imagenet_val_path', default='/home/rilevin/data/ImageNet/val', type=str, help='ImageNet validation set path')
# parser.add_argument('--testset_stats_path', default='', type=str, help='filter saliency over the testset (where to save)')
# parser.add_argument('--inference_file_path', default='', type=str, help='where to save network inference results')

def save_gradients(grads_to_save, args, experiment, reference_image, inv_transform_test):
    grads_to_save, _ = grads_to_save.max(dim=1)
    grads_to_save = grads_to_save.detach().cpu().numpy().reshape((224, 224))
    grads_to_save = np.abs(grads_to_save)
    # grads_to_save[grads_to_save < 0] = 0.0

    #Percentile thresholding
    grads_to_save[grads_to_save > np.percentile(grads_to_save, 99)] = np.percentile(grads_to_save, 99)
    grads_to_save[grads_to_save < np.percentile(grads_to_save, 90)] = np.percentile(grads_to_save, 90)

    plt.figure()
    plt.imshow(grads_to_save)

    save_path = os.path.join('figures', args.figure_folder_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_name = str(args.reference_id) if args.reference_id is not None else args.image_path.split('/')[-1].split('.')[0]
    save_name += '_' + args.model
    plt.axis('off')
    # plt.savefig(os.path.join(save_path, 'input_space_saliency_{}.pdf'.format(save_name)), bbox_inches='tight')


    grads_to_save = (grads_to_save - np.min(grads_to_save)) / (np.max(grads_to_save) - np.min(grads_to_save))

    #Superimpose gradient heatmap
    reference_image_to_compare = inv_transform_test(reference_image[0].cpu()).permute(1, 2, 0)
    gradients_heatmap = np.ones_like(grads_to_save) - grads_to_save
    gradients_heatmap = cv2.GaussianBlur(gradients_heatmap, (3, 3), 0)

    #Save the heatmap
    heatmap_superimposed = show_heatmap_on_image(reference_image_to_compare.detach().cpu().numpy(), gradients_heatmap)
    plt.imshow(heatmap_superimposed)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'input_saliency_heatmap_{}.png'.format(save_name)), bbox_inches='tight')
    print('Input space saliency saved to {} \n'.format(os.path.join(save_path, 'input_saliency_heatmap_{}.png'.format(save_name))))
    return

def compute_input_space_saliency(reference_inputs, reference_targets, net, args, experiment,
                                 testset_mean_stat=None, testset_std_stat=None, inv_transform_test = None,
                                 readable_labels = None):
    #First, log things
    with torch.no_grad():
        ref_image_to_log = inv_transform_test(reference_inputs[0].detach().cpu()).permute(1, 2, 0)


        reference_outputs = net(reference_inputs)
        _, reference_predicted = reference_outputs.max(1)
        # Log classes:
        print("""\n
        Image target label: {}
        Image target class name: {}
        Image predicted label: {}
        Image predicted class name: {}\n
        """.format(reference_targets[0].item(),
            readable_labels[reference_targets[0].item()],
            reference_predicted[0].item(),
            readable_labels[reference_predicted[0].item()]))

    #Compute filter saliency
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device='cuda', mode='std',
                                          aggregation='filter_wise', signed=args.signed, logit=args.logit,
                                          logit_difference=args.logit_difference)
    reference_inputs, reference_targets = reference_inputs.to(device), reference_targets.to(device)

    grad_samples = []
    #Errors are a fragile concept, we should not perturb too much, we will end up on the object
    for noise_iter in range(args.noise_iters):
        perturbed_inputs = reference_inputs.detach().clone()
        perturbed_inputs = (1-args.noise_percent)*perturbed_inputs + args.noise_percent*torch.randn_like(perturbed_inputs)

        perturbed_outputs = net(perturbed_inputs)
        _, perturbed_predicted = perturbed_outputs.max(1)
        # print(readable_labels[int(perturbed_predicted[0])])

        #Backprop to the input
        perturbed_inputs.requires_grad_()
        #Find the true saliency:
        filter_saliency = filter_saliency_model(
            perturbed_inputs, reference_targets,
            testset_mean_abs_grad=testset_mean_stat,
            testset_std_abs_grad=testset_std_stat).to(device)

        #Find the top-k salient filters
        if args.compare_random:
            sorted_filters = torch.randperm(filter_saliency.size(0)).cpu().numpy()
        else:
            sorted_filters = torch.argsort(filter_saliency, descending=True).cpu().numpy()

        #Boost them:
        filter_saliency_boosted = filter_saliency.detach().clone()
        filter_saliency_boosted[sorted_filters[:args.k_salient]] *= args.boost_factor

        #Form matching loss and take the gradient:
        matching_criterion = torch.nn.CosineSimilarity()
        matching_loss = matching_criterion(filter_saliency[None, :], filter_saliency_boosted[None, :])
        matching_loss.backward()

        grads_to_save = perturbed_inputs.grad.detach().cpu()
        grad_samples.append(grads_to_save)
    #Find averaged gradients (smoothgrad-like)
    grads_to_save = torch.stack(grad_samples).mean(0)

    return grads_to_save, filter_saliency


def sort_filters_layer_wise(filter_profile, layer_to_filter_id, filter_std = None):
    layer_sorted_profile = []
    means = []
    stds = []
    for layer in layer_to_filter_id:
        layer_inds = layer_to_filter_id[layer]
        layer_sorted_profile.append(np.sort(filter_profile[layer_inds])[::-1])
        means.append(np.ones_like(filter_profile[layer_inds])*np.mean(filter_profile[layer_inds]))
        if filter_std is not None:
            stds.append(filter_std[layer_inds][np.argsort(filter_profile[layer_inds])[::-1]])
    layer_sorted_profile = np.concatenate(layer_sorted_profile)
    sal_means = np.concatenate(means)
    if filter_std is not None:
        sal_stds = np.concatenate(stds)
        return layer_sorted_profile, sal_means, sal_stds
    else:
        return layer_sorted_profile, sal_means

if __name__ == '__main__':

    torch.manual_seed(40)
    np.random.seed(40)

    ###########################################################
    ####Define net, testset, precompute testset avg saliency
    ###########################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    experiment = None #Used to be a comet_ml experiment for logging

    model_helpers_root_path = os.path.join('helper_objects', args.model)
    if not os.path.exists(model_helpers_root_path):
        print('No helper objects directory exists for this model, creating one\n')
        os.mkdir(model_helpers_root_path)

    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  ## ImageNet statistics
    ])

    inv_transform_test = transforms.Compose([
        transforms.Normalize(mean=(0., 0., 0.), std=(1 / 0.229, 1 / 0.224, 1 / 0.225)),
        transforms.Normalize(mean=(-0.485, -0.456, -0.406), std=(1., 1., 1.)),
    ])

    # ImageNet validation set
    if args.data_to_use == 'ImageNet':
        images_path = args.imagenet_val_path
    else:
        raise NotImplementedError

    testset = torchvision.datasets.ImageFolder(images_path, transform=transform_test)
    # Downloading imagenet 1000 classes list of readable labels
    label_url = urllib.request.urlopen(
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
    readable_labels = ''
    for f in label_url:
        readable_labels = readable_labels + f.decode("utf-8")
    readable_labels = yaml.load(readable_labels)
    # Model
    print('==> Building model..')

    if args.model == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'vgg19':
        net = torchvision.models.vgg19(pretrained=True)
    elif args.model == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
    elif args.model == 'inception_v3':
        net = torchvision.models.inception_v3(pretrained=True)
    else:
        #Other torchvision models should be inserted here
        raise NotImplementedError

    net = net.to(device)
    net.eval()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    layer_to_filter_id = {}
    ind = 0
    for layer_num, (name, param) in enumerate(net.named_parameters()):
        # print(name, param.shape)
        if len(param.size()) == 4:
            # if 'conv' not in name:
            #     print('Not a conv layer {}: {}'.format(name, layer_num))
            for j in range(param.size()[0]):
                if name not in layer_to_filter_id:
                    layer_to_filter_id[name] = [ind + j]
                else:
                    layer_to_filter_id[name].append(ind + j)

            ind += param.size()[0]


    # print('Unit of interest:', filter_id_to_layer_filter_id[22101])
    total = 0
    for layer in layer_to_filter_id:
        total += len(layer_to_filter_id[layer])
    print('Total filters:', total)
    print('Total layers:', len(layer_to_filter_id))

    #Load inference files
    inference_file = os.path.join(model_helpers_root_path, 'ImageNet_val_inference_results_{:s}.pth'.format(args.model))
    if os.path.isfile(inference_file):
        inf_results = torch.load(inference_file)
        incorrect_id = inf_results['incorrect_id']
        incorrect_predictions = inf_results['incorrect_predictions']
        correct_id = inf_results['correct_id']
    else:
        warnings.warn("Computing inference, check the names of saved inference files if this was not intended")
        incorrect_id, incorrect_predictions, correct_id = test_and_find_incorrectly_classified(net, testset)
        torch.save({'incorrect_id': incorrect_id,
                    'incorrect_predictions': incorrect_predictions,
                    'correct_id': correct_id}, inference_file)

    # if args.logit:
    #     folder = 'logit'
    #     warnings.warn('All final experiments were done with args.logit off')
    # elif args.logit_difference:
    #     folder = 'logit_difference'
    #     warnings.warn('All final experiments were done with args.logit_difference off')
    # else:
    #     folder = 'loss'

    #Load valset stats files
    filter_stats_file = os.path.join(model_helpers_root_path, 'ImageNet_val_saliency_stat_{:s}_filter_wise.pth'.format(args.model))
    if os.path.isfile(filter_stats_file):
        filter_stats = torch.load(filter_stats_file)
        filter_testset_mean_abs_grad = filter_stats['mean']
        filter_testset_std_abs_grad = filter_stats['std']
    else:
        warnings.warn("Computing testset stats, check the names of saved stats files if this was not intended")
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad = find_testset_saliency(net, testset, 'filter_wise', args)
        torch.save({'mean': filter_testset_mean_abs_grad, 'std': filter_testset_std_abs_grad}, filter_stats_file)

    if args.reference_id is None:
        print("""\n
        Using image {}
        and target label {}\n
        """.format(args.image_path, args.image_target_label))
        reference_image = transform_raw_image(args.image_path).unsqueeze(0)
        reference_target = torch.tensor(int(args.image_target_label)).unsqueeze(0)
    else:
        print("""\n
        Using reference_id to select the image for the experiment. 
        Working with {}-th image from ImageNet validation set and its target label.
        If this was intended, please make sure to specify path to ImageNet validation set.
        If using an image from raw_images/ was intended, please do not specify 
        --reference_id and use --image_target_label and --image_path args instead.
        """.format(args.reference_id))
        reference_image, reference_target = testset.__getitem__(args.reference_id)
        reference_target = torch.tensor(reference_target).unsqueeze(0)
        reference_image.unsqueeze_(0)


    grads_to_save, filter_saliency = compute_input_space_saliency(reference_image, reference_target, net, args, experiment, filter_testset_mean_abs_grad, filter_testset_std_abs_grad, inv_transform_test, readable_labels)

    layer_sorted_profile, sal_means = sort_filters_layer_wise(
        filter_saliency.detach().cpu().numpy(), layer_to_filter_id)
    #Save input space saliency:
    save_gradients(grads_to_save, args, experiment, reference_image, inv_transform_test)

    #Plot and save parameter saliency
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    pal = sns.color_palette('colorblind')
    blue_color = pal.as_hex()[0]
    orange_color = pal.as_hex()[1]
    ax.plot(layer_sorted_profile, label='Sorted filter saliency', c=blue_color)  # '0.25')
    ax.legend()
    ax.get_legend().get_frame().set_alpha(0.0)
    ax.set_xlabel('Filter ID')
    ax.set_ylabel('Saliency')
    save_name = str(args.reference_id) if args.reference_id is not None else args.image_path.split('/')[-1].split('.')[0]
    save_name += '_' + args.model
    fig.savefig('figures/filter_saliency_{}.png'.format(save_name))
    print('Filter saliency saved to figures/filter_saliency_{}.png'.format(save_name))
#Run this: python3 input_saliency.py --reference_id 107 --k_salient 10
#Run this: python3 parameter_and_input_saliency.py --image_path raw_images/great_white_shark_mispred_as_killer_whale.jpeg --image_target_label 2
#TODO:
#-rearrange args
#-fix readme
#-give github link in the paper
#-upload preprint to arxiv