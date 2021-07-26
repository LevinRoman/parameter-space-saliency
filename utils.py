'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import random
import csv
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
import numpy as np
import torchvision
import copy
import numpy as np
from PIL import Image, ImageFilter
# import matplotlib.cm as mpl_color_map
from torch.autograd import Variable
import cv2
import torchvision.transforms as transforms
from PIL import Image


def transform_raw_image(image_path):
    # raw_image = cv2.imread(image_path)
    # with open(image_path, 'rb') as f:
    # img = Image.open(image_path)
    raw_image = Image.open(image_path).convert('RGB')
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  ## ImageNet statistics
    ])
    # transformed_image = transform_test(torch.from_numpy(raw_image[..., ::-1].copy()))
    transformed_image = transform_test(raw_image)
    return transformed_image


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def save_output_from_dict(state, save_dir='/checkpoint/', save_name = 'table.csv'):
    out_path = os.path.join(save_dir,save_name)
    print(out_path)

    # Read input information
    args = []
    values = []
    for arg, value in state.items():
        args.append(arg)
        values.append(value)
    
    # Check for file
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    fieldnames = [arg for arg in args]

    # Read or write header
    try:
        with open(out_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except:
        with open(out_path, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()

    # Add row for this experiment 
    with open(out_path, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow({arg: value for (arg, value) in zip(args, values)})
    print('\nResults saved to '+out_path)
    
def get_indices(trainset, num_per_class):
    if num_per_class == -1:
        return [*range(len(trainset))]
    else:
        idx_dict = {} #keys are classes, values are lists of trainset indices
        indices = []
        for idx, (inputs, targets) in enumerate(trainset):
            if targets in idx_dict:
                idx_dict[targets].append(idx)
            else:
                idx_dict[targets] = [idx]
        num_classes = len(idx_dict)
        for key in idx_dict:
            indices.extend(random.sample(idx_dict[key], num_per_class))
        return indices 

def remove_parallel(state_dict):
    ''' state_dict: state_dict of model saved with DataParallel()
        returns state_dict without extra module level ''' 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    return new_state_dict






def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('../results', file_name + '.jpg')
    save_image_pil(gradient, path_to_file)


def save_image_pil(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_image_IN(inputs, target_class, predicted_class, folder_name, figure_name, inv_transform_test):
    true_class = target_class
    image = inv_transform_test(inputs[0])
    print('Saving image...\t True_class:{} | \t Predicted_class:{}'.format(true_class, predicted_class))
    if not os.path.isdir('figures'):
        os.mkdir('figures')
    if not os.path.isdir('figures/{}'.format(folder_name)):
        os.mkdir('figures/{}'.format(folder_name))
    torchvision.utils.save_image(image, './figures/{}/'.format(folder_name) 
                +'image_tc_' + str(true_class) + '_pc_' + str(predicted_class) + '_' + figure_name + '.jpg')


def show_heatmap_on_image(img, mask):
    """both img and mask should be between 0,1"""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam#np.uint8(255 * cam)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_and_find_incorrectly_classified(net, testset, batch_size=128):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    incorrect_id = []
    incorrect_predictions = []

    correct_id = []

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4)
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            correct_output = torch.where(predicted == targets, predicted, torch.tensor(-1, device='cuda'))
            correct_id_batch = (correct_output + torch.tensor(1, device='cuda')).nonzero(as_tuple=True)[0]
            correct_id_batch += batch_idx * batch_size
            correct_id_batch = list(correct_id_batch.detach().cpu().numpy())
            correct_id.extend(correct_id_batch)

            incorrect_id_batch = \
            torch.where(correct_output == -1, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda')).nonzero(
                as_tuple=True)[0]
            incorrect_id_batch += batch_idx * batch_size
            incorrect_id_batch = list(incorrect_id_batch.detach().cpu().numpy())
            # print("incorrect id: ", incorrect_id_batch)
            incorrect_id.extend(incorrect_id_batch)
            incorrect_predictions.extend(list(predicted[~predicted.eq(targets)].detach().cpu().numpy()))
            assert len(incorrect_id) == len(incorrect_predictions)

        assert len(incorrect_id) + len(correct_id) == len(testset)
        # Concatenate the incorrectly classified samples and labels (across batches)
        print(
            "Finished testing: Acc: {:.3f}, total incorrect sample: {:d}".format(100. * len(correct_id) / len(testset),
                                                                                 len(incorrect_id)))
    return incorrect_id, incorrect_predictions, correct_id
if __name__ == '__main__':
    test()