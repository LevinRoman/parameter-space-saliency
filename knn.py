import os
import argparse
from pathlib import Path

import random
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, AutoConfig

from data import load_file, filter_samples, apply_template, batchify
from utils import sort_grads
from inference import get_dataset_stats, test_and_find_incorrect_prediction, sample_saliency_curves


CACHE_DIR = './cache'

def closest_k_saliency(reference_saliency_list, sample_saliency_list, k=10, lst=[], cosine=True):
    # Initialize list of indices to output
    idx_lst = []

    # Iterate through all the input indices
    for i in lst:
        if cosine:
            dist_vec = F.cosine_similarity(reference_saliency_list[i].repeat(sample_saliency_list.size()[0], 1), sample_saliency_list, dim=1)
            dist_vec = torch.abs(dist_vec - 1.0)
            dist_vec = dist_vec.cpu()
        else:
            dist_vec = F.pairwise_distance(reference_saliency_list[i].repeat(sample_saliency_list.size()[0], 1), sample_saliency_list)
        # print("distance vec size: ", dist_vec.size())
        # Use distance matrix to find the top k smallest distance elements away
        distances, closest_idx = torch.topk(dist_vec, k, largest=False)
        # print("closest index for sample{:d}: ".format(i), closest_idx)
        idx_lst.append([distances, closest_idx])
    
    return idx_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency curve comparison')
    parser.add_argument('--bert_model_name',
                        default='bert-base-cased', type=str)
    parser.add_argument('--dataset_name', default='/cmlscratch/manlis/data/lama/Squad/test.jsonl', type=str)
    parser.add_argument('--max_seq_length', default=1024, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--aggr', default="column-wise", type=str,
                        choices=['naive', 'column-wise'])
    # knn
    parser.add_argument('--k_closest', default=5, type=int, help='number of closest saliency curves')
    parser.add_argument('--chosen_samples', default="[]", 
                            type=str, help='list of samples_ids to run k_closest on')
    parser.add_argument('--num_samples', default=5, type=int, help='number of chosen samples')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--partial', default=None, 
                            type=int, help='index in the saliency profile to cut it in half')
    args = parser.parse_args()

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # set up the model
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model_name, cache_dir=CACHE_DIR)

    config = AutoConfig.from_pretrained(
        args.bert_model_name, cache_dir=CACHE_DIR)
    model = AutoModelForMaskedLM.from_pretrained(
        args.bert_model_name,
        config=config,
        cache_dir=CACHE_DIR,
    )
    model = model.to(args.device)
    # model.resize_token_embeddings(len(tokenizer))
    # for name, param in model.named_parameters():
    #     print("{}: {}".format(name, param.size()))
    # raise NotImplementedError()

    # load data TODO: add an outer loop for relations
    data = load_file(args.dataset_name)
    print(len(data))

    template = ""  # TODO: add a lookup dict for relation-template pairs
    all_samples, ret_msg = filter_samples(
        model, tokenizer, data, args.max_seq_length, template
    )
    # print(ret_msg)
    print(len(all_samples))

    if template != "":
        all_samples = apply_template(all_samples, template)

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1

    # get inference results
    inference_file = './{:s}_{:s}_inference_results_{:s}.pth'.format(
        Path(os.path.dirname(args.dataset_name)).stem, Path(args.dataset_name).stem, args.bert_model_name)
    if os.path.isfile(inference_file):
        inf_results = torch.load(inference_file)
        incorrect_id = inf_results['incorrect_id']
        incorrect_predictions = inf_results['incorrect_predictions']
        correct_id = inf_results['correct_id']
    else:
        incorrect_id, incorrect_predictions, correct_id = test_and_find_incorrect_prediction(
            all_samples, model, tokenizer, args)
        torch.save({'incorrect_id': incorrect_id,
                    'incorrect_predictions': incorrect_predictions,
                    'correct_id': correct_id}, inference_file)
    print("correct prediction: {}/{} (P@1 = {:.3f})".format(len(correct_id), len(correct_id)+len(incorrect_id), len(correct_id)/(len(correct_id)+len(incorrect_id))))
    if args.chosen_samples == "[]":
        random.seed(args.seed)
        np.random.seed(args.seed)
        args.chosen_samples = list(np.random.choice(incorrect_id, args.num_samples)) 
    else:
        string_list = args.chosen_samples
        lst = string_list.strip('][').split(',') 
        for i in range(len(lst)):
            lst[i] = int(lst[i])
        args.chosen_samples = lst
    print("slected sample ids: ", args.chosen_samples)


    # mean and std of the testset
    stat_file = './{:s}_{:s}_saliency_stat_{:s}.pth'.format(
        Path(os.path.dirname(args.dataset_name)).stem, args.aggr, args.bert_model_name)
    if os.path.isfile(stat_file):
        stats = torch.load(stat_file)
        testset_mean = stats['testset_mean']
        testset_std = stats['testset_std']
    else:
        testset_mean, testset_std = get_dataset_stats(
            all_samples, model, tokenizer, args)
        torch.save({'testset_mean': testset_mean,
                    'testset_std': testset_std}, stat_file)

    # saliency curves
    saliency_file = './{:s}_{:s}_{:s}_saliency_profile_{:s}.pth'.format(Path(os.path.dirname(args.dataset_name)).stem, Path(args.dataset_name).stem, args.aggr, args.bert_model_name)
    if os.path.isfile(saliency_file):
        saliency_curves = torch.load(saliency_file)
    else:
        saliency_curves = sample_saliency_curves(all_samples, model, tokenizer, testset_mean, testset_std, args)
        torch.save(saliency_curves, saliency_file)

    print("total saliency profile size: ", saliency_curves.size())
    if args.partial is not None:
        saliency_curves = saliency_curves[:, args.partial:]
        print("using partial saliency profile, size: ", saliency_curves.size())

    # sample_short = saliency_curves[0:100, :].numpy()
    # sample_med = saliency_curves[1500:1600, :].numpy()
    # sample_long = saliency_curves[3500:3600, :].numpy()
    # print("sample shape", sample_long.shape)
    # ind = np.arange(0, saliency_curves.size(1))
    # fig, ax = plt.subplots()
    # print("before plot")
    # ax.plot(ind[::50], np.average(sample_short, axis=0)[::50])
    # ax.plot(ind[::50], np.average(sample_med, axis=0)[::50])
    # ax.plot(ind[::50], np.average(sample_long, axis=0)[::50])
    # print("before save")
    # fig.savefig('./test.png')
    # raise NotImplementedError()

    closest_idx_lst = closest_k_saliency(saliency_curves, saliency_curves, args.k_closest, lst=args.chosen_samples)


    fp = open('./{:s}_{:s}_log_results_seed{:d}_{:s}.txt'.format(
        Path(os.path.dirname(args.dataset_name)).stem, args.aggr, args.seed, args.bert_model_name), 'w')
    print(closest_idx_lst)
    samples_batches, sentences_batches, label_batches = batchify(all_samples, 1)
    for ind, res in enumerate(closest_idx_lst):
        print("reference sample: {}".format(label_batches[args.chosen_samples[ind]]))
        fp.write("reference sample: {}\n".format(label_batches[args.chosen_samples[ind]]))
        nns = list(res[-1].numpy())
        for i, id in enumerate(nns):
            print("{} closest: {}".format(i, label_batches[id]), end='\t')
            fp.write("{} closest: {} \t".format(i, label_batches[id]))
            if id in correct_id:
                print("correctly predicted: {}".format(tokenizer.decode([incorrect_predictions[id]])))
                fp.write("correctly predicted: {}\n".format(tokenizer.decode([incorrect_predictions[id]])))
            elif id in incorrect_id:
                print("incorrectly predicted as: {}".format(tokenizer.decode([incorrect_predictions[id]])))
                fp.write("incorrectly predicted as: {}\n".format(tokenizer.decode([incorrect_predictions[id]])))
    fp.close()



