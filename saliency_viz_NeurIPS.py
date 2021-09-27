from comet_ml import Experiment
import yaml
import urllib
import pandas as pd
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import SubsetRandomSampler, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time
# from models import *
from utils import AverageMeter, test_and_find_incorrectly_classified
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from parameter_saliency.saliency_model_backprop import SaliencyModel, find_testset_saliency
import warnings
import tqdm

parser = argparse.ArgumentParser(description='Saliency inversion and boosting')
parser.add_argument('--model', default='resnet50', type=str, help='name of architecture')
parser.add_argument('--data_dir', default='~/data', type=str, help='data directory')
parser.add_argument('--data_to_use', default='ImageNet', type=str, help='which dataset to use (ImageNet or ImageNet_A)')
parser.add_argument('--save_name', default='model.pt', type=str, help='name of table file')
# Logging
parser.add_argument('--log_frequency', default=1000, type=int, help='How frequently to save images (in epochs)')
parser.add_argument('--project_name', default='saliency_viz_NeurIPS', type=str, help='project name for Comet ML')

# Frying parameters:
parser.add_argument('--fry_params', action='store_true', help='whether to fry top salient filters')
parser.add_argument('--k_salient', default=100, type=int, help='num filters to fry')
parser.add_argument('--fry_random', action='store_true',
                    help='whether to fry k random filters for comparison to frying most salient')
parser.add_argument('--least_salient', action='store_true',
                    help='whether to fry k least salient filters for comparison to frying most salient')

# Modes for the signed saliency model
parser.add_argument('--signed', action='store_true', help='Use signed saliency')
parser.add_argument('--logit', action='store_true', help='Use signed saliency')
parser.add_argument('--logit_difference', action='store_true', help='Use signed saliency')

# Filter destruction modes:
parser.add_argument('--rand_perturb_filter', action='store_true', help='Do fixed random perturbation N(0,1)')
parser.add_argument('--rand_perturb_filter_proportional', action='store_true',
                    help='Do random perturbation proportional to parameter size')
parser.add_argument('--zero_out_filter', action='store_true', help='Zero out  filters')
parser.add_argument('--remove_filter_bn', action='store_true',
                    help='Truly remove the filter -- zero out also bn bias so that the filter has no contribution to next layer')




def find_avg_std_saliency(net, testset, aggregation, args, testset_mean_stat, testset_std_stat):
    """find_saliency is a basic saliency method: could be naive, could be averaging across filters, tensors, layers, etc
    Return average magnitude of gradient across samples in the testset and std of that"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # This local testloader should have batch size 1, but we can do more for quick debugging
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)

    # Go through images one by one, cannot go in batches since we need avg abs grad, not just avg grad
    # Incrementally compute the mean and std to not run out of memory
    iter_time = AverageMeter()
    end = time.time()
    for batch_idx, (testset_inputs, testset_targets) in enumerate(testloader):
        testset_inputs, testset_targets = testset_inputs.to(device), testset_targets.to(device)
        testset_outputs = net(testset_inputs)
        _, testset_predicted = testset_outputs.max(1)

        filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device='cuda', mode='std',
                                              aggregation=aggregation, signed=args.signed, logit=args.logit,
                                              logit_difference=args.logit_difference)
        testset_grad = filter_saliency_model(testset_inputs, testset_targets, testset_mean_abs_grad=testset_mean_stat, testset_std_abs_grad=testset_std_stat).detach().to(device)

        if batch_idx == 0:
            # oldM in Welford's method (https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
            testset_mean_abs_grad_prev = torch.zeros_like(testset_grad, dtype=torch.float64)
            testset_mean_abs_grad = testset_grad / float(batch_idx + 1)
            # print(testset_mean_abs_grad)
            testset_std_abs_grad = (testset_grad - testset_mean_abs_grad) * (testset_grad - testset_mean_abs_grad_prev)
        else:
            testset_mean_abs_grad_prev = testset_mean_abs_grad.detach().clone()  # oldM
            testset_mean_abs_grad += (testset_grad - testset_mean_abs_grad) / float(
                batch_idx + 1)  # update M to the current
            # print(testset_mean_abs_grad)
            testset_std_abs_grad += (testset_grad - testset_mean_abs_grad) * (
                    testset_grad - testset_mean_abs_grad_prev)  # update variance
            # print(testset_std_abs_grad)

        iter_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 50 == 0:
            remain_time = (len(testloader) - batch_idx - 1) * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            print("ITer: [{:d}/{:d}]\t iter time: {iter_time.val: .3f}\t remain time: {remain_time}".format(
                batch_idx + 1, len(testloader), iter_time=iter_time, remain_time=remain_time))

    testset_std_abs_grad = testset_std_abs_grad / float(len(testloader) - 1)  # Unbiased estimator of variance
    print('Variance:', testset_std_abs_grad)
    testset_std_abs_grad = torch.sqrt(testset_std_abs_grad)
    print('Std:', testset_std_abs_grad)
    print('Mean:', testset_mean_abs_grad)
    print('Testset_grads_shape:{}'.format(testset_mean_abs_grad.shape))

    return testset_mean_abs_grad, testset_std_abs_grad

def compare_correct_vs_incorrect_predicted_label(net, testset, args, testset_mean_stat=None, testset_std_stat=None):
    """find_saliency is a basic saliency method: could be naive, could be averaging across filters, tensors, layers, etc
    Return average magnitude of gradient across samples in the testset and std of that"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # This local testloader should have batch size 1, but we can do more for quick debugging
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)

    # Go through images one by one, cannot go in batches since we need avg abs grad, not just avg grad
    # Incrementally compute the mean and std to not run out of memory
    iter_time = AverageMeter()
    end = time.time()

    first_correct = True
    first_incorrect = True
    for batch_idx, (testset_inputs, testset_targets) in enumerate(testloader):
        # if batch_idx > 2000:
        #     break
        testset_inputs, testset_targets = testset_inputs.to(device), testset_targets.to(device)
        testset_outputs = net(testset_inputs)
        _, testset_predicted = testset_outputs.max(1)

        filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device='cuda', mode='std',
                                              aggregation='filter_wise', signed=args.signed, logit=args.logit,
                                              logit_difference=args.logit_difference)
        testset_grad = filter_saliency_model(testset_inputs, testset_predicted, testset_mean_abs_grad=testset_mean_stat, testset_std_abs_grad=testset_std_stat).detach().to(device)

        if first_correct:
            if torch.all(testset_predicted == testset_targets):
                #If correctly classified
                # oldM in Welford's method (https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
                testset_mean_abs_grad_prev_correct = torch.zeros_like(testset_grad, dtype=torch.float64)
                testset_mean_abs_grad_correct = testset_grad / float(batch_idx + 1)
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_correct = (testset_grad - testset_mean_abs_grad_correct) * (testset_grad - testset_mean_abs_grad_prev_correct)
                first_correct = False
        else:
            if torch.all(testset_predicted == testset_targets):
                testset_mean_abs_grad_prev_correct = testset_mean_abs_grad_correct.detach().clone()  # oldM
                testset_mean_abs_grad_correct += (testset_grad - testset_mean_abs_grad_correct) / float(
                    batch_idx + 1)  # update M to the current
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_correct += (testset_grad - testset_mean_abs_grad_correct) * (
                        testset_grad - testset_mean_abs_grad_prev_correct)  # update variance
                # print(testset_std_abs_grad)

        if first_incorrect:
            if not torch.all(testset_predicted == testset_targets):
                # If correctly classified
                # oldM in Welford's method (https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
                testset_mean_abs_grad_prev_incorrect = torch.zeros_like(testset_grad, dtype=torch.float64)
                testset_mean_abs_grad_incorrect = testset_grad / float(batch_idx + 1)
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_incorrect = (testset_grad - testset_mean_abs_grad_incorrect) * (
                            testset_grad - testset_mean_abs_grad_prev_incorrect)
                first_incorrect = False
        else:
            if not torch.all(testset_predicted == testset_targets):
                testset_mean_abs_grad_prev_incorrect = testset_mean_abs_grad_incorrect.detach().clone()  # oldM
                testset_mean_abs_grad_incorrect += (testset_grad - testset_mean_abs_grad_incorrect) / float(
                    batch_idx + 1)  # update M to the current
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_incorrect += (testset_grad - testset_mean_abs_grad_incorrect) * (
                        testset_grad - testset_mean_abs_grad_prev_incorrect)  # update variance
                # print(testset_std_abs_grad)

        iter_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 50 == 0:
            remain_time = (len(testloader) - batch_idx - 1) * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            print("ITer: [{:d}/{:d}]\t iter time: {iter_time.val: .3f}\t remain time: {remain_time}".format(
                batch_idx + 1, len(testloader), iter_time=iter_time, remain_time=remain_time))

    testset_std_abs_grad_correct = testset_std_abs_grad_correct / float(len(testloader) - 1)  # Unbiased estimator of variance
    testset_std_abs_grad_incorrect = testset_std_abs_grad_incorrect / float(len(testloader) - 1)  # Unbiased estimator of variance
    # print('Variance:', testset_std_abs_grad)
    # testset_std_abs_grad = torch.sqrt(testset_std_abs_grad)
    # print('Std:', testset_std_abs_grad)
    # print('Mean:', testset_mean_abs_grad)
    # print('Testset_grads_shape:{}'.format(testset_mean_abs_grad.shape))

    return testset_mean_abs_grad_correct, testset_std_abs_grad_correct, testset_mean_abs_grad_incorrect, testset_std_abs_grad_incorrect

def compare_correct_vs_incorrect_target_label(net, testset, args, testset_mean_stat=None, testset_std_stat=None):
    """find_saliency is a basic saliency method: could be naive, could be averaging across filters, tensors, layers, etc
    Return average magnitude of gradient across samples in the testset and std of that"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # This local testloader should have batch size 1, but we can do more for quick debugging
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2)

    # Go through images one by one, cannot go in batches since we need avg abs grad, not just avg grad
    # Incrementally compute the mean and std to not run out of memory
    iter_time = AverageMeter()
    end = time.time()

    first_correct = True
    first_incorrect = True
    for batch_idx, (testset_inputs, testset_targets) in enumerate(testloader):
        # if batch_idx > 2000:
        #     break
        testset_inputs, testset_targets = testset_inputs.to(device), testset_targets.to(device)
        testset_outputs = net(testset_inputs)
        _, testset_predicted = testset_outputs.max(1)

        filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device='cuda', mode='std',
                                              aggregation='filter_wise', signed=args.signed, logit=args.logit,
                                              logit_difference=args.logit_difference)
        testset_grad = filter_saliency_model(testset_inputs, testset_targets, testset_mean_abs_grad=testset_mean_stat, testset_std_abs_grad=testset_std_stat).detach().to(device)

        if first_correct:
            if torch.all(testset_predicted == testset_targets):
                #If correctly classified
                # oldM in Welford's method (https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
                testset_mean_abs_grad_prev_correct = torch.zeros_like(testset_grad, dtype=torch.float64)
                testset_mean_abs_grad_correct = testset_grad / float(batch_idx + 1)
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_correct = (testset_grad - testset_mean_abs_grad_correct) * (testset_grad - testset_mean_abs_grad_prev_correct)
                first_correct = False
        else:
            if torch.all(testset_predicted == testset_targets):
                testset_mean_abs_grad_prev_correct = testset_mean_abs_grad_correct.detach().clone()  # oldM
                testset_mean_abs_grad_correct += (testset_grad - testset_mean_abs_grad_correct) / float(
                    batch_idx + 1)  # update M to the current
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_correct += (testset_grad - testset_mean_abs_grad_correct) * (
                        testset_grad - testset_mean_abs_grad_prev_correct)  # update variance
                # print(testset_std_abs_grad)

        if first_incorrect:
            if not torch.all(testset_predicted == testset_targets):
                # If correctly classified
                # oldM in Welford's method (https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
                testset_mean_abs_grad_prev_incorrect = torch.zeros_like(testset_grad, dtype=torch.float64)
                testset_mean_abs_grad_incorrect = testset_grad / float(batch_idx + 1)
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_incorrect = (testset_grad - testset_mean_abs_grad_incorrect) * (
                            testset_grad - testset_mean_abs_grad_prev_incorrect)
                first_incorrect = False
        else:
            if not torch.all(testset_predicted == testset_targets):
                testset_mean_abs_grad_prev_incorrect = testset_mean_abs_grad_incorrect.detach().clone()  # oldM
                testset_mean_abs_grad_incorrect += (testset_grad - testset_mean_abs_grad_incorrect) / float(
                    batch_idx + 1)  # update M to the current
                # print(testset_mean_abs_grad)
                testset_std_abs_grad_incorrect += (testset_grad - testset_mean_abs_grad_incorrect) * (
                        testset_grad - testset_mean_abs_grad_prev_incorrect)  # update variance
                # print(testset_std_abs_grad)

        iter_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % 50 == 0:
            remain_time = (len(testloader) - batch_idx - 1) * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            print("ITer: [{:d}/{:d}]\t iter time: {iter_time.val: .3f}\t remain time: {remain_time}".format(
                batch_idx + 1, len(testloader), iter_time=iter_time, remain_time=remain_time))

    testset_std_abs_grad_correct = testset_std_abs_grad_correct / float(len(testloader) - 1)  # Unbiased estimator of variance
    testset_std_abs_grad_incorrect = testset_std_abs_grad_incorrect / float(len(testloader) - 1)  # Unbiased estimator of variance
    # print('Variance:', testset_std_abs_grad)
    # testset_std_abs_grad = torch.sqrt(testset_std_abs_grad)
    # print('Std:', testset_std_abs_grad)
    # print('Mean:', testset_mean_abs_grad)
    # print('Testset_grads_shape:{}'.format(testset_mean_abs_grad.shape))

    return testset_mean_abs_grad_correct, testset_std_abs_grad_correct, testset_mean_abs_grad_incorrect, testset_std_abs_grad_incorrect




if __name__ == '__main__':

    torch.manual_seed(40)
    np.random.seed(40)

    ###########################################################
    ####Define net, testset, precompute testset avg saliency
    ###########################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    experiment = Experiment(api_key='P63wSM91MmVDh80ZBZbcylZ8L',
                            project_name=args.project_name)  # "Inverse training: ImageNet {}".format(args.inversion_mode))

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
        images_path = os.path.join('/fs/cml-datasets/ImageNet/ILSVRC2012/', 'val')

    testset = torchvision.datasets.ImageFolder(images_path, transform=transform_test)
    # Downloading imagenet 1000 classes list
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
        #/nfshomes/vcherepa/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
    elif args.model == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
    elif args.model == 'inception_v3':
        net = torchvision.models.inception_v3(pretrained=True)
    else:
        raise NotImplementedError

    filter_id_to_layer_filter_id = {}
    layer_to_filter_id = {}
    ind = 0
    ind_bn = 0
    print(net._modules)
    for layer_num, (name, param) in enumerate(net.named_parameters()):
        print(name, param.shape)
        if len(param.size()) == 4:
            if 'conv' not in name:
                print('ERROR', name, layer_num)
            if args.model == 'inception_v3':
                layers_to_drop_ids = [210, 211, 212, 213, 214, 215, 216, 217]
                if layer_num not in layers_to_drop_ids:
                    for j in range(param.size()[0]):
                        if name not in layer_to_filter_id:
                            layer_to_filter_id[name] = [ind + j]
                        else:
                            layer_to_filter_id[name].append(ind + j)
                        if 'layer1' in name:
                            filter_id_to_layer_filter_id[ind + j] = 'layer1_id{}_param:{}'.format(j, name)
                        if 'layer2' in name:
                            filter_id_to_layer_filter_id[ind + j] = 'layer2_id{}_param:{}'.format(j, name)
                        if 'layer3' in name:
                            filter_id_to_layer_filter_id[ind + j] = 'layer3_id{}_param:{}'.format(j, name)
                        if 'layer4' in name:
                            filter_id_to_layer_filter_id[ind + j] = 'layer4_id{}_param:{}'.format(j, name)
                        # if ind + j not in top_k_salient_filter:
                        #     param.grad[j, :, :, :] = torch.zeros_like(param.grad[j, :, :, :], device='cuda')
                    ind += param.size()[0]
            else:
                for j in range(param.size()[0]):
                    if name not in layer_to_filter_id:
                        layer_to_filter_id[name] = [ind + j]
                    else:
                        layer_to_filter_id[name].append(ind + j)
                    if 'layer1' in name:
                        filter_id_to_layer_filter_id[ind + j] = 'layer1_id{}_param:{}'.format(j, name)
                    if 'layer2' in name:
                        filter_id_to_layer_filter_id[ind + j] = 'layer2_id{}_param:{}'.format(j, name)
                    if 'layer3' in name:
                        filter_id_to_layer_filter_id[ind + j] = 'layer3_id{}_param:{}'.format(j, name)
                    if 'layer4' in name:
                        filter_id_to_layer_filter_id[ind + j] = 'layer4_id{}_param:{}'.format(j, name)
                    # if ind + j not in top_k_salient_filter:
                    #     param.grad[j, :, :, :] = torch.zeros_like(param.grad[j, :, :, :], device='cuda')
                ind += param.size()[0]
        if 'bn' in name:
            if 'bias' in name:
                print('bias detected')
            for j in range(param.size()[0]):
                pass
            print(name)
            ind_bn += param.size()[0]

    # print('Unit of interest:', filter_id_to_layer_filter_id[22101])
    total = 0
    for layer in layer_to_filter_id:
        total += len(layer_to_filter_id[layer])
    print('Total filters:', total)
    print('Total layers:', len(layer_to_filter_id))

    # if args.model == 'inception_v3':
    #     layers_to_drop_ids = [210, 211, 212, 213, 214, 215, 216, 217]
    #     layers_to_drop = len(list(layer_to_filter_id.keys()))#[210:218]
    #     print(layers_to_drop)
    #     layers_to_keep = [layer for layer in layer_to_filter_id if layer not in layers_to_drop]
    #     layer_to_filter_id = {layer: layer_to_filter_id[layer] for layer in layers_to_keep}
    #     print('Total filters:', total)
    #     print('Total layers:', len(layer_to_filter_id))

    net = net.to(device)
    net.eval()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    model_helpers_root_path = os.path.join('helper_objects', args.model)
    if not os.path.exists(model_helpers_root_path):
        print('No helper objects directory exists for this model, creating one\n')
        os.mkdir(model_helpers_root_path)

    inference_file = os.path.join(model_helpers_root_path, 'ImageNet_val_inference_results_{:s}.pth'.format(args.model))
    if os.path.isfile(inference_file):
        inf_results = torch.load(inference_file)
        incorrect_id = inf_results['incorrect_id']
        incorrect_predictions = inf_results['incorrect_predictions']
        correct_id = inf_results['correct_id']
    else:
        incorrect_id, incorrect_predictions, correct_id = test_and_find_incorrectly_classified(net, testset)
        torch.save({'incorrect_id': incorrect_id,
                    'incorrect_predictions': incorrect_predictions,
                    'correct_id': correct_id}, inference_file)


    filter_stats_file = os.path.join(model_helpers_root_path, 'ImageNet_val_saliency_stat_{:s}_filter_wise.pth'.format(args.model))
    if os.path.isfile(filter_stats_file):
        filter_stats = torch.load(filter_stats_file)
        filter_testset_mean_abs_grad = filter_stats['mean']
        filter_testset_std_abs_grad = filter_stats['std']
    else:
        warnings.warn("Computing testset stats, check the names of saved stats files if you did not intend to do this")
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad = find_testset_saliency(net, testset, 'filter_wise',
                                                                                          args)
        torch.save({'mean': filter_testset_mean_abs_grad, 'std': filter_testset_std_abs_grad}, filter_stats_file)

    avg_stdized_file = os.path.join(model_helpers_root_path, 'average_stdized_{:s}_filter_wise.pth'.format(args.model))
    if os.path.isfile(avg_stdized_file):
        avg_stdized = torch.load(avg_stdized_file)
        avg_stdized_filter_wise_mean = avg_stdized['mean']
        avg_stdized_filter_wise_std = avg_stdized['std']
    else:
        warnings.warn('Computing avg stdized, check the names of saved files if you did not intend to do this')
        avg_stdized_filter_wise_mean, avg_stdized_filter_wise_std = find_avg_std_saliency(net, testset, 'filter_wise', args, filter_testset_mean_abs_grad, filter_testset_std_abs_grad)
        torch.save({'mean': avg_stdized_filter_wise_mean, 'std': avg_stdized_filter_wise_std}, avg_stdized_file)

    comparison_file = os.path.join(model_helpers_root_path, 'comparison_target_label_{:s}_filter_wise.pth'.format(args.model))
    if os.path.isfile(comparison_file):
        comparisons = torch.load(comparison_file)
        mean_sal_correct = comparisons['mean_sal_correct']
        std_sal_correct = comparisons['std_sal_correct']
        mean_sal_incorrect = comparisons['mean_sal_incorrect']
        std_sal_incorrect = comparisons['std_sal_incorrect']
    else:
        warnings.warn("Computing comparisons, check the names of saved stats files if you did not intend to do this")
        mean_sal_correct, std_sal_correct, mean_sal_incorrect, std_sal_incorrect = compare_correct_vs_incorrect_target_label(net, testset, args, testset_mean_stat=filter_testset_mean_abs_grad, testset_std_stat=filter_testset_std_abs_grad)
        torch.save({'mean_sal_correct': mean_sal_correct, 'std_sal_correct': std_sal_correct, 'mean_sal_incorrect': mean_sal_incorrect, 'std_sal_incorrect': std_sal_incorrect}, comparison_file)

    comparison_file_predicted = os.path.join(model_helpers_root_path, 'comparison_predicted_label_{:s}_filter_wise.pth'.format(args.model))
    if os.path.isfile(comparison_file_predicted):
        comparisons_predicted = torch.load(comparison_file_predicted)
        mean_sal_correct_predicted = comparisons_predicted['mean_sal_correct']
        std_sal_correct_predicted = comparisons_predicted['std_sal_correct']
        mean_sal_incorrect_predicted = comparisons_predicted['mean_sal_incorrect']
        std_sal_incorrect_predicted = comparisons_predicted['std_sal_incorrect']
    else:
        warnings.warn("Computing comparisons, check the names of saved stats files if you did not intend to do this")
        mean_sal_correct_predicted, std_sal_correct_predicted, mean_sal_incorrect_predicted, std_sal_incorrect_predicted = compare_correct_vs_incorrect_predicted_label(
            net, testset, args, testset_mean_stat=filter_testset_mean_abs_grad,
            testset_std_stat=filter_testset_std_abs_grad)
        torch.save({'mean_sal_correct': mean_sal_correct_predicted, 'std_sal_correct': std_sal_correct_predicted,
                    'mean_sal_incorrect': mean_sal_incorrect_predicted, 'std_sal_incorrect': std_sal_incorrect_predicted}, comparison_file_predicted)

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


    # layer_sorted_profile, sal_means = sort_filters_layer_wise(mean_sal_correct.detach().cpu().numpy(), layer_to_filter_id)
    layer_sorted_profile, sal_means, naive_sorted_stds = sort_filters_layer_wise(filter_testset_mean_abs_grad.detach().cpu().numpy(), layer_to_filter_id,
                                                                                 filter_testset_std_abs_grad.detach().cpu().numpy())
    layer_sorted_profile_std, sal_means_std = sort_filters_layer_wise(avg_stdized_filter_wise_mean.detach().cpu().numpy(),
                                                              layer_to_filter_id)
    correct_sorted, correct_means = sort_filters_layer_wise(mean_sal_correct.detach().cpu().numpy(), layer_to_filter_id)
    # correct_sorted = mean_sal_correct.detach().cpu().numpy()
    incorrect_sorted, incorrect_means = sort_filters_layer_wise(mean_sal_incorrect.detach().cpu().numpy(), layer_to_filter_id)
    # incorrect_sorted = mean_sal_incorrect.detach().cpu().numpy()/np.linalg.norm(incorrect_sorted)

    correct_sorted_predicted, correct_means_predicted, correct_stds_predicted = sort_filters_layer_wise(mean_sal_correct_predicted.detach().cpu().numpy(), layer_to_filter_id,
                                                                                std_sal_correct_predicted.detach().cpu().numpy())
    # correct_sorted_predicted = mean_sal_correct_predicted.detach().cpu().numpy()
    incorrect_sorted_predicted, incorrect_means_predicted, incorrect_stds_predicted = sort_filters_layer_wise(mean_sal_incorrect_predicted.detach().cpu().numpy(),
                                                                layer_to_filter_id, std_sal_incorrect_predicted.detach().cpu().numpy())
    # incorrect_sorted_predicted = mean_sal_incorrect_predicted.detach().cpu().numpy()
    # incorrect_sorted_predicted = incorrect_sorted_predicted/np.linalg.norm(incorrect_sorted_predicted)

    mean_sal_incorrect_predicted.unsqueeze_(0)
    mean_sal_incorrect.unsqueeze_(0)
    print('Cos distance:', 1 - torch.nn.CosineSimilarity()(mean_sal_incorrect, mean_sal_incorrect_predicted))
    # layer_sorted_profile = avg_stdized_filter_wise_mean.detach().cpu().numpy()# filter_testset_mean_abs_grad.detach().cpu().numpy()#mean_sal_correct.detach().cpu().numpy()
    # Seaborn settings:
    sns.set(font_scale=1.3)
    sns.set_palette('colorblind')
    sns.set_style("ticks")  # , {'grid.linestyle': '--'})

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    #Naive vs std
    fig, ax = plt.subplots(2, 1, figsize=(15, 7))
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    pal = sns.color_palette('colorblind')
    blue_color = pal.as_hex()[0]
    orange_color = pal.as_hex()[1]
    ax[0].plot(layer_sorted_profile, label = 'Average gradient magnitudes', c = blue_color)#'0.25')
    # ci = 1.96*naive_sorted_stds / np.sqrt(50000)
    # x=np.arange(len(naive_sorted_stds))
    # ax[0].fill_between(x, (layer_sorted_profile - ci), (layer_sorted_profile + ci), color='b', alpha=.3)
    # ax[0].plot(sal_means, ls = '--', c='0.45', linewidth = 0.5)
    ax[0].legend()
    ax[0].get_legend().get_frame().set_alpha(0.0)
    # ax[0].set_xlabel('Filter ID')
    ax[0].set_ylabel('Saliency')

    ax[1].plot(layer_sorted_profile_std, label='Average standardized saliency profile', c = orange_color)
    # ax[1].plot(sal_means_std, ls='--', c='0.45', linewidth = 0.5, label = 'Average of a layer')
    ax[1].legend()
    ax[1].get_legend().get_frame().set_alpha(0.0)
    ax[1].set_xlabel('Filter ID')
    ax[1].set_ylabel('Saliency')
    fig.savefig('naive_vs_std_{}.pdf'.format(args.model), bbox_inches='tight')

    #Incorrect vs correct
    fig, ax = plt.subplots(2, 1, figsize=(15, 5))
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)

    pal = sns.color_palette('colorblind')
    blue_color = pal.as_hex()[0]
    orange_color = pal.as_hex()[1]
    ax[0].plot(correct_sorted, label='Average saliency profile of correctly classified samples', c = blue_color)
    ax[0].legend()
    ax[0].get_legend().get_frame().set_alpha(0.0)
    ax[0].set_xlabel('Filter ID')
    ax[0].set_ylabel('Saliency')

    ax[1].plot(incorrect_sorted, label='Average saliency profile of incorrectly classified samples', c=orange_color)
    # ax[1].plot(sal_means_std, ls='--', c='0.45', linewidth = 0.5, label = 'Average of a layer')
    ax[1].legend()
    ax[1].get_legend().get_frame().set_alpha(0.0)
    ax[1].set_xlabel('Filter ID')
    ax[1].set_ylabel('Saliency')
    fig.savefig('correct_vs_incorrect_target_{}.pdf'.format(args.model), bbox_inches='tight')

    #Incorrect vs correct with predicted label:
    fig, ax = plt.subplots(2, 1, figsize=(15, 5))
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)

    pal = sns.color_palette('colorblind')
    blue_color = pal.as_hex()[0]
    orange_color = pal.as_hex()[1]
    ax[0].plot(correct_sorted_predicted, label='Average saliency profile of correctly classified samples', c=blue_color)
    ax[0].legend()
    ax[0].get_legend().get_frame().set_alpha(0.0)
    ax[0].set_xlabel('Filter ID')
    ax[0].set_ylabel('Saliency')

    ax[1].plot(incorrect_sorted_predicted, label='Average saliency profile of incorrectly classified samples', c=orange_color)
    # ax[1].plot(sal_means_std, ls='--', c='0.45', linewidth = 0.5, label = 'Average of a layer')
    ax[1].legend()
    ax[1].get_legend().get_frame().set_alpha(0.0)
    ax[1].set_xlabel('Filter ID')
    ax[1].set_ylabel('Saliency')
    fig.savefig('correct_vs_incorrect_predicted_{}.pdf'.format(args.model), bbox_inches='tight')
    # ax[2].plot(correct_means, ls='--')
    # ax[2].legend()
    # ax[3].plot(incorrect_sorted, label='Incorrect', c='g')
    # # ax[3].plot(incorrect_means, ls='--')
    # ax[3].legend()
    # ax[4].plot(correct_sorted_predicted, label='Correct', c='g')
    # ax[4].plot(correct_sorted_predicted+correct_stds_predicted, label='Correct', c='g', ls = '--')
    # ax[4].plot(correct_sorted_predicted-correct_stds_predicted, label='Correct', c='g', ls = '--')
    # ax[4].plot(correct_means_predicted, ls='--')
    # ax[4].legend()
    # ax[4].plot(incorrect_sorted_predicted, label='Incorrect', c='r')
    # ax[4].plot(incorrect_means_predicted, ls='--')
    # ax[4].plot(incorrect_sorted_predicted+incorrect_stds_predicted, label='Incorrect', c='r', ls='--')
    # ax[4].plot(incorrect_sorted_predicted-incorrect_stds_predicted, label='Incorrect', c='r', ls='--')
    # ax[4].legend()
    # ax[0].axvline(layer_to_filter_iplt.legend()d['layer4.0.downsample.0.weight'][0])
    # ax[0].axvline(layer_to_filter_id['layer4.0.downsample.0.weight'][-1])
    # ax[0].axvline(layer_to_filter_id['layer3.0.downsample.0.weight'][0])
    # ax[0].axvline(layer_to_filter_id['layer3.0.downsample.0.weight'][-1])
    # ax[0].axvline(layer_to_filter_id['layer2.0.downsample.0.weight'][0])
    # ax[0].axvline(layer_to_filter_id['layer2.0.downsample.0.weight'][-1])
    # ax[0].axvline(layer_to_filter_id['layer1.0.downsample.0.weight'][0])
    # ax[0].axvline(layer_to_filter_id['layer1.0.downsample.0.weight'][-1])
    # plt.legend()
    # ax[1].plot(mean_sal_incorrect.detach().cpu().numpy(), label = 'Incorrect', c = 'r')
    # plt.legend()
    # ax[2].plot(mean_sal_correct.detach().cpu().numpy() - mean_sal_incorrect.detach().cpu().numpy(), label = 'Difference')
    # plt.legend()
    # ax[0].set_title('Correct mean saliency with predicted label')
    # ax[1].set_title('Incorrect mean saliency with predicted label')
    # ax[2].set_title('Difference')

    # for layer in layer_to_filter_id:
    #     print('\n Layer {}, filter id from {} to {} \n'.format(layer, layer_to_filter_id[layer][0], layer_to_filter_id[layer][-1]))


    # def weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    # net.apply(weights_init)
#Run this:  python pred_label_comparison.py --project_name correct_vs_incorrect_pred_label