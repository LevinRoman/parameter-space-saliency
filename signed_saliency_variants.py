"""Frying experiments are here
Note the order of the target_to_subtract and the reference_targets in the loss regime of SaliencyModel """
#import code from clean repo
from comet_ml import Experiment
import yaml
import urllib
import pandas as pd
import multiprocessing
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
from models import *
from utils import AverageMeter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import warnings
import tqdm
from saliency.saliency_model_backprop import SaliencyModel

parser = argparse.ArgumentParser(description='Saliency inversion and boosting')
parser.add_argument('--model', default='resnet50', type=str, help='name of architecture')
parser.add_argument('--data_dir', default='~/data', type=str, help='data directory')
parser.add_argument('--data_to_use', default='ImageNet', type=str, help='which dataset to use (ImageNet or ImageNet_A)')
parser.add_argument('--save_name', default='model.pt', type=str, help='name of table file')
# Logging
parser.add_argument('--log_frequency', default=1000, type=int, help='How frequently to save images (in epochs)')
parser.add_argument('--project_name', default='frying_experiments', type=str, help='project name for Comet ML')

# Frying parameters:
parser.add_argument('--fry_params', action='store_true', help='whether to fry top salient filters')
parser.add_argument('--k_salient', default=100, type=int, help='num filters to fry')
parser.add_argument('--fry_random', action='store_true', help='whether to fry k random filters for comparison to frying most salient')
parser.add_argument('--least_salient', action='store_true', help='whether to fry k least salient filters for comparison to frying most salient')

#Modes for the signed saliency model
parser.add_argument('--signed', action='store_true', help='Use signed saliency')
parser.add_argument('--logit', action='store_true', help='Use signed saliency')
parser.add_argument('--logit_difference', action='store_true', help='Use signed saliency')

#Filter destruction modes:
parser.add_argument('--rand_perturb_filter', action='store_true', help='Do fixed random perturbation N(0,1)')
parser.add_argument('--rand_perturb_filter_proportional', action='store_true', help='Do random perturbation proportional to parameter size')
parser.add_argument('--zero_out_filter', action='store_true', help='Zero out  filters')
parser.add_argument('--remove_filter_bn', action='store_true', help='Truly remove the filter -- zero out also bn bias so that the filter has no contribution to next layer')

#Chunk running:
parser.add_argument('--chunk', default=0, type=int, help='number of the chunk to run')
parser.add_argument('--chunksize', default=1500, type=int, help='chunk size to run')
parser.add_argument('--log_freq', default=20, type=int, help='chunk size to run')


def find_testset_saliency(net, testset, aggregation, args):
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

        filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device='cuda', mode='naive',
                                              aggregation=aggregation, signed = args.signed, logit = args.logit, logit_difference = args.logit_difference)
        testset_grad = filter_saliency_model(testset_inputs, testset_targets).detach().to(device)

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

# class SaliencyModel(nn.Module):
#     def __init__(self, net, criterion, device='cuda', mode='std', aggregation='filter_wise', signed = False, logit = False, logit_difference = False):
#         super(SaliencyModel, self).__init__()
#         self.net = net
#         self.criterion = criterion
#         self.device = device
#         self.mode = mode
#         self.aggregation = aggregation
#         self.signed = signed
#         self.logit = logit
#         self.logit_difference = logit_difference
#
#     def forward(self, inputs, targets, target_to_subtract = None, testset_mean_abs_grad=None, testset_std_abs_grad=None):
#         """inputs should be a random noise image, targets should be the target label, note that this should be a batch of size 1
#         target_to"""
#         # Turn on gradient for the input image
#         inputs.requires_grad_()
#
#         # Set the net in eval mode
#         self.net.eval()
#
#         inputs, targets, target_to_subtract = inputs.to(self.device), targets.to(self.device), target_to_subtract.to(self.device)
#
#         self.net.zero_grad()
#         outputs = self.net(inputs)
#
#         if self.logit:
#             loss = outputs[0][int(targets[0])]
#         elif self.logit_difference:
#             loss = outputs[0][int(targets[0])] - outputs[0][int(target_to_subtract[0])]
#         else:
#             loss = self.criterion(outputs, target_to_subtract)# targets)
#             #TODO handle this better!
#
#         # with torch.enable_grad():
#
#         gradients = autograd.grad(loss, self.net.parameters(), create_graph=True)
#
#         filter_grads = []
#         for i in range(len(gradients)):  # Filter-wise aggregation
#             # print(gradients[i].size())
#
#             if self.aggregation == 'filter_wise':
#                 if len(gradients[i].size()) == 4:  # If conv layer
#                     filter_grads.append(gradients[i].mean(-1).mean(-1).mean(-1))
#             if self.aggregation == 'parameter_wise':
#                 filter_grads.append(gradients[i].view(-1))
#             if self.aggregation == 'tensor_wise':
#                 raise NotImplementedError
#
#         if not self.signed:
#             naive_saliency = torch.abs(torch.cat(filter_grads))
#         else:
#             naive_saliency = torch.cat(filter_grads)
#         if self.mode == 'naive':
#             return naive_saliency
#         if self.mode == 'std':
#             testset_std_abs_grad[testset_std_abs_grad <= 1e-14] = 1  # This should fix nans in the resulting saliency
#             std_saliency = (naive_saliency - testset_mean_abs_grad.to(self.device)) / testset_std_abs_grad.to(
#                 self.device)
#             return std_saliency
#         if self.mode == 'norm':
#             testset_mean_abs_grad[testset_mean_abs_grad <= 1e-14] = 1
#             norm_saliency = naive_saliency / testset_mean_abs_grad.to(self.device)
#             return norm_saliency

# def find_top5_outputs(net, image, readable_labels, experiment, inv_transform_test, name):
#     outputs = torch.nn.functional.softmax((net(image))).cpu()
#     scores, top5_id = torch.topk(outputs, 5)
#     top5 = [readable_labels[i] for i in np.squeeze(top5_id.cpu().numpy().astype('int'))]
#
#     image_intermediate = inv_transform_test(image[0].detach().cpu()).permute(1, 2, 0)
#     experiment.log_image(image_intermediate, 'Image {}'.format(name))
#     experiment.log_other('Top 5 classes {}'.format(name), top5)
#     experiment.log_other('Confidence score {}'.format(name), scores.to('cpu'))
#     return top5, scores


def track_top_and_true(net, image, target_label, readable_labels):  # experiment, inv_transform_test, name):
    outputs = torch.nn.functional.softmax(net(image)).cpu()
    scores_top, top_id = torch.topk(outputs, 1)
    top1 = readable_labels[int(top_id)]  # for i in np.squeeze(top_id.cpu().numpy().astype('int'))]

    score_target = outputs[0][target_label]
    # image_intermediate = inv_transform_test(image[0].detach().cpu()).permute(1, 2, 0)
    # experiment.log_image(image_intermediate, 'Image {}'.format(name))
    # experiment.log_other('Top 5 classes {}'.format(name), top5)
    # experiment.log_other('Confidence score {}'.format(name), scores.to('cpu'))
    return top1, scores_top, score_target


def track_missed_and_true(net, image, target_label, missed_label):  # experiment, inv_transform_test, name):
    with torch.no_grad():
        outputs = torch.nn.functional.softmax(net(image))
        _, predicted = outputs.max(1)
        score_missed = outputs[0][missed_label]
        score_target = outputs[0][target_label]
        flipped = predicted == target_label
    return score_missed, score_target, flipped


if __name__ == '__main__':

    torch.manual_seed(40)
    np.random.seed(40)

    ###########################################################
    ####Define net, testset, precompute testset avg saliency
    ###########################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    experiment = Experiment(api_key='P63wSM91MmVDh80ZBZbcylZ8L',
                            project_name=args.project_name)

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
    if args.data_to_use == 'ImageNet_A':
        images_path = os.path.join('/cmlscratch/vcherepa/rilevin/ImageNetA/imagenet_a/', 'imagenet-a')
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
        filter_id_to_layer_filter_id = {}
        ind = 0
        ind_bn = 0
        print(net._modules)
        for name, param in net.named_parameters():
            print(name, param.shape)
            if len(param.size()) == 4:
                for j in range(param.size()[0]):
                    if 'layer1' in name:
                        filter_id_to_layer_filter_id[ind+j] = 'layer1_id{}_param:{}'.format(j, name)
                    if 'layer2' in name:
                        filter_id_to_layer_filter_id[ind+j] = 'layer2_id{}_param:{}'.format(j, name)
                    if 'layer3' in name:
                        filter_id_to_layer_filter_id[ind+j] = 'layer3_id{}_param:{}'.format(j, name)
                    if 'layer4' in name:
                        filter_id_to_layer_filter_id[ind+j] = 'layer4_id{}_param:{}'.format(j, name)
                    # if ind + j not in top_k_salient_filter:
                    #     param.grad[j, :, :, :] = torch.zeros_like(param.grad[j, :, :, :], device='cuda')
                ind += param.size()[0]
            if 'bn' in name:
                if 'bias' in name:
                    print('bias detected')
                for j  in range(param.size()[0]):
                    pass
                print(name)
                ind_bn += param.size()[0]

        print('Unit of interest:', filter_id_to_layer_filter_id[17969])
    else:
        raise NotImplementedError

    net = net.to(device)
    net.eval()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    inference_file = '/cmlscratch/vcherepa/rilevin/ImageNet_experiments/ImageNet_val_inference_results_{:s}.pth'.format(args.model)
    if os.path.isfile(inference_file):
        inf_results = torch.load(inference_file)
        incorrect_id = inf_results['incorrect_id']
        incorrect_predictions = inf_results['incorrect_predictions']
        correct_id = inf_results['correct_id']
    else:
        incorrect_id, incorrect_predictions, correct_id = test_and_find_incorrecty_classified(net, testset)
        torch.save({'incorrect_id': incorrect_id,
                    'incorrect_predictions': incorrect_predictions,
                    'correct_id': correct_id}, inference_file)

    if args.logit:
        folder = 'logit'
    elif args.logit_difference:
        folder = 'logit_difference'
    else:
        folder = 'loss'

    filter_stats_file = '/cmlscratch/vcherepa/rilevin/ImageNet_experiments/ImageNet_val_saliency_stat_{:s}_filter_wise.pth'.format(args.model)
    #filter_stats_file_cpu = '/cmlscratch/vcherepa/rilevin/ImageNet_experiments/signed/{:s}/ImageNet_val_saliency_stat_{:s}_filter_wise_cpu.pth'.format(folder, args.model)
    if os.path.isfile(filter_stats_file):
        filter_stats = torch.load(filter_stats_file)
        filter_testset_mean_abs_grad = filter_stats['mean']
        filter_testset_std_abs_grad = filter_stats['std']
    else:
        warnings.warn("Computing testset stats, check the names of saved stats files if you did not intend to do this")
        filter_testset_mean_abs_grad, filter_testset_std_abs_grad = find_testset_saliency(net, testset, 'filter_wise', args)
        torch.save({'mean': filter_testset_mean_abs_grad, 'std': filter_testset_std_abs_grad}, filter_stats_file)

    filter_testset_mean_abs_grad, filter_testset_std_abs_grad = filter_testset_mean_abs_grad.to(device), filter_testset_std_abs_grad.to(device)

    #frying
    def run_image(args, net, reference_id, missed_conf, true_conf, flipped_to_true_label):
        reference_image, reference_target = testset.__getitem__(reference_id)
        reference_target = torch.tensor(reference_target).unsqueeze(0)
        reference_image.unsqueeze_(0)

        reference_image, reference_target = reference_image.to(device), reference_target.to(device)
        # top5_before, scores_before = find_top5_outputs(net, image, readable_labels, experiment,
        #                                                inv_transform_test,
        #                                                'before frying')
        outputs = torch.nn.functional.softmax(net(reference_image))
        _, missed_label = outputs.max(1)

        score_missed_initial = outputs[0][missed_label]
        score_target_initial = outputs[0][reference_target]
        # print('\n Memory allocated in run: {} \n'.format(torch.cuda.memory_allocated(0)))

        # reference_saliency = saliency_model(reference_image, reference_target, testset_mean_abs_grad, testset_std_abs_grad)
        # reference_saliency = reference_saliency.detach().to(device)
        # We standardize using the signed saliency and then use absolute values of that
        filter_saliency_model = SaliencyModel(net, nn.CrossEntropyLoss(), device=device, mode='std',
                                              aggregation='filter_wise', signed=args.signed, logit=args.logit, logit_difference=args.logit_difference)

        #We try to decrease the missed label logit and increase the correct label logit
        filter_saliency = filter_saliency_model(reference_image, reference_target, None, filter_testset_mean_abs_grad,
                                                filter_testset_std_abs_grad)
        filter_saliency = torch.abs(filter_saliency).detach().cpu()


        sorted_filters = list(torch.argsort(filter_saliency, descending=True).cpu().numpy())

        if args.least_salient:
            sorted_filters = list(torch.argsort(filter_saliency, descending=False).cpu().numpy())
        if args.fry_random:
            random_filters_idx = list(torch.randperm(filter_saliency.size(0)).cpu().numpy())
        # top_labels = []

        # net.eval()
        # net.zero_grad()

        #need conf score of incorrect lbl
        # loss = nn.CrossEntropyLoss()(outputs, reference_target.to(device))
        # if args.logit:
        #     loss = outputs[0][int(missed_label[0])]
        # elif args.logit_difference:
        #     raise NotImplementedError
        # #     loss = outputs[0][int(targets[0])] - outputs[0][int(target_to_subtract[0])]
        # else:
        #     loss = nn.CrossEntropyLoss()(outputs, reference_target.to(device))
        # loss.backward()
        # print('\n Memory allocated in run: {} \n'.format(torch.cuda.memory_allocated(0)))

        for k in tqdm.tqdm(range(args.k_salient)):
            if args.fry_random:
                # print('Frying random')
                top_k_salient_filter = random_filters_idx[:k]
            else:
                top_k_salient_filter = sorted_filters[:k]
            # Fry the top salient params
            # print(top_k_salient_filter)
            ind = 0
            ind_bn = 0
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if len(param.size()) == 4:
                        for j in range(param.size()[0]):
                            if ind + j in top_k_salient_filter:
                                # print('Fried filter {}'.format(ind+j))
                                #only zero out the ones that point in the same direction
                                #with their gradient taking the sign into account

                                # if (param[j, :, :, :]*param.grad[j, :, :, :]).detach().cpu().sum() > 0:
                                if args.zero_out_filter:
                                    param[j, :, :, :].copy_(torch.zeros_like(param[j, :, :, :]))
                                if args.rand_perturb_filter:
                                    param[j, :, :, :].copy_(param[j, :, :, :] + torch.randn(param[j, :, :, :].size()))#.to(device))
                                if args.rand_perturb_filter_proportional:
                                    noise = torch.randn(param[j, :, :, :].size())
                                    noise = noise/torch.norm(noise)
                                    param[j, :, :, :].copy_(param[j, :, :, :] + torch.norm(param[j, :, :, :])*noise)#.to(device))
                        ind += param.size()[0]
                    else:
                        param.grad = None  # TODO: handle this better
                    if args.remove_filter_bn:
                        if 'bn' in name:
                            if 'bias' in name:
                                for j in range(param.size()[0]):
                                    if ind_bn + j in top_k_salient_filter:
                                        # if param[j] > 0:
                                        #     print(name, ind+j, param[j])
                                        param[j].copy_(torch.zeros_like(param[j]))
                                ind_bn += param.size()[0]

            # top5_after_frying, scores_after_frying = find_top5_outputs(net, image, readable_labels, experiment, inv_transform_test,
            #                                                'after frying')

            # top1, score_top, score_target = track_top_and_true(net, image, reference_target, readable_labels)
            score_missed, score_target, flipped = track_missed_and_true(net, reference_image, reference_target, missed_label)
            # top_labels.append(top1)

            # if k == 0:
            #     initial_score_missed = score_missed.clone()
            #     initial_score_target = score_target.clone()
            #
            # if k > 5:
            #     if top_labels[-1] != top_labels[-2]:
            #         print('Flipped to {} at iteration {}!'.format(top_labels[-1], k))
            # experiment.log_metric("Missed_label_score", score_missed.item(), step=k)
            # experiment.log_metric("Top_score", score_top.item(), step=k)
            # experiment.log_metric("Target_score", score_target.item(), step=k)
            missed_conf.append([reference_id, k, score_missed.cpu().item() - score_missed_initial.cpu().item()])
            true_conf.append([reference_id, k, score_target.cpu().item() - score_target_initial.cpu().item()])
            flipped_to_true_label.append([reference_id, k, flipped.cpu().item()])
        # experiment.log_other('Top_labels', top_labels)
        del reference_image, outputs, reference_target, filter_saliency_model, filter_saliency, missed_label, score_missed, score_target
        torch.cuda.empty_cache()

        return missed_conf, true_conf, flipped_to_true_label


    # missed_confidence_curves = []
    # true_confidence_curves = []

    missed_conf = []
    true_conf = []
    flipped_to_true_label = []
    sample_incorrect_id = np.random.choice(incorrect_id, size=20)
    chunksize = args.chunksize

    #Check files
    last_processed_id_missed = last_processed_id_true = last_processed_id_flipped = -1
    last_processed_id = -1
    last_processed_num = -1

    if args.fry_random:
        mode_name = 'random'
    elif args.least_salient:
        mode_name = 'least_salient'
    else:
        mode_name = 'most_salient'

    saving_path = '/cmlscratch/vcherepa/rilevin/NeurIPS/pruning/'
    if os.path.exists(os.path.join(saving_path, 'missed_conf_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk))):
        print('Continue from preemption')
        missed_conf = pd.read_csv(os.path.join(saving_path, 'missed_conf_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk)), index_col=0)
        missed_conf = np.array(missed_conf)
        missed_conf = missed_conf.tolist()
        last_processed_id_missed = missed_conf[-1][0]
    if os.path.exists(os.path.join(saving_path, 'true_conf_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk))):
        print('Continue from preemption')
        true_conf = pd.read_csv(os.path.join(saving_path, 'true_conf_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk)), index_col=0)
        true_conf = np.array(true_conf)
        true_conf = true_conf.tolist()
        last_processed_id_true = true_conf[-1][0]
    if os.path.exists(os.path.join(saving_path, 'flipped_to_true_label_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk))):
        print('Continue from preemption')
        flipped_to_true_label = pd.read_csv(os.path.join(saving_path, 'flipped_to_true_label_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk)), index_col=0)
        flipped_to_true_label = np.array(flipped_to_true_label)
        flipped_to_true_label = flipped_to_true_label.tolist()
        last_processed_id_flipped = flipped_to_true_label[-1][0]

    if last_processed_id_missed == last_processed_id_true == last_processed_id_flipped != -1:
        last_processed_id = last_processed_id_missed
        last_processed_num = incorrect_id[chunksize * args.chunk:chunksize * (args.chunk + 1)].index(last_processed_id)
    if not last_processed_id_missed == last_processed_id_true == last_processed_id_flipped:
        raise ValueError('Something is wrong: the last processed id is different')

    for num, reference_id in tqdm.tqdm(enumerate(incorrect_id[chunksize*args.chunk:chunksize*(args.chunk+1)][last_processed_num+1:])):
        # print('\n Memory allocated: {} \n'.format(torch.cuda.memory_allocated(0)))
        # print('\n Memory reserved: {} \n'.format(torch.cuda.memory_reserved(0)))
        net = torchvision.models.resnet50(pretrained=True)
        net = net.to(device)
        net.eval()
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
        missed_conf, true_conf, flipped_to_true_label = run_image(args, net, reference_id, missed_conf, true_conf, flipped_to_true_label)

        if (num > 0) and (num%args.log_freq == 0):
            missed_conf_intermediate = pd.DataFrame(np.array(missed_conf), columns=['reference_id', 'k', 'conf_score'])
            missed_conf_intermediate.to_csv(os.path.join(saving_path, 'missed_conf_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk)))

            true_conf_intermediate = pd.DataFrame(np.array(true_conf), columns=['reference_id', 'k', 'conf_score'])
            true_conf_intermediate.to_csv(os.path.join(saving_path, 'true_conf_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk)))

            flipped_to_true_label_intermediate = pd.DataFrame(np.array(flipped_to_true_label).astype('float'), columns=['reference_id', 'k', 'conf_score'])
            flipped_to_true_label_intermediate.to_csv(os.path.join(saving_path, 'flipped_to_true_label_intermediate_{}_chunk_{}.csv'.format(mode_name, args.chunk)))
    #     # del net
        # torch.cuda.empty_cache()
    if args.fry_random:
        mode_name = 'random'
    elif args.least_salient:
        mode_name = 'least_salient'
    else:
        mode_name = 'most_salient'
    # fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    saving_path = '/cmlscratch/vcherepa/rilevin/NeurIPS/pruning/'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    missed_conf = pd.DataFrame(np.array(missed_conf), columns=['reference_id', 'k', 'conf_score'])
    missed_conf.to_csv(os.path.join(saving_path, 'missed_conf_{}_chunk_{}.csv'.format(mode_name, args.chunk)))
    print('\n Len missed_conf:{}'.format(missed_conf.shape[0]))
    true_conf = pd.DataFrame(np.array(true_conf), columns=['reference_id', 'k', 'conf_score'])
    true_conf.to_csv(os.path.join(saving_path, 'true_conf_{}_chunk_{}.csv'.format(mode_name, args.chunk)))
    print('\n Len true_conf:{}'.format(true_conf.shape[0]))
    flipped_to_true_label = pd.DataFrame(np.array(flipped_to_true_label).astype('float'), columns=['reference_id', 'k', 'flipped'])
    flipped_to_true_label.to_csv(os.path.join(saving_path, 'flipped_{}_chunk_{}.csv'.format(mode_name, args.chunk)))
    print('\n Len flipped_to_true_label:{}'.format(flipped_to_true_label.shape[0]))
    # sns.lineplot(data=missed_conf, x='k', y='conf_score', ax=ax[0], ci=68)
    # ax[0].set_title('Predicted Missed Label')
    # sns.lineplot(data=true_conf, x='k', y='conf_score', ax=ax[1], ci=68)
    # ax[1].set_title('True Label')
    # sns.lineplot(data=flipped_to_true_label, x='k', y='flipped', ax=ax[2], ci=68)
    # ax[2].set_title('Flipped to True Label')
    # if args.fry_random:
    #     fig.savefig('conf_scores_random_neurips.pdf')
    # elif args.least_salient:
    #     fig.savefig('conf_scores_least_salient_neurips.pdf')
    # else:
    #     fig.savefig('conf_scores_most_salient_neurips.pdf')
    #
