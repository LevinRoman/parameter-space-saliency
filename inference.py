import os
import argparse
import time

import torch
# torch.set_printoptions(threshold=10_000)
import transformers
from transformers import BertForMaskedLM, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, AutoConfig
# from datasets import load_dataset

from data import load_file, filter_samples, apply_template, batchify
from utils import sort_grads, AverageMeter

CACHE_DIR = './cache'

def test_and_find_incorrect_prediction(all_samples, model, tokenizer, args):
    incorrect_id = []
    incorrect_predictions = []
    correct_id = []

    samples_batches, sentences_batches, label_batches = batchify(all_samples, 100)
    for i in range(len(samples_batches)):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        gt_b = label_batches[i]

        inputs_b = tokenizer(sentences_b, padding=True, return_tensors='pt')
        labels_b = tokenizer(gt_b, padding=True, return_tensors='pt')["input_ids"]
        labels_b[inputs_b["input_ids"] != tokenizer.mask_token_id] = -100 # only calculate loss on masked tokens

        _, mask_inds = torch.where(inputs_b["input_ids"]==tokenizer.mask_token_id)
        mask_inds = mask_inds.to(args.device)
        with torch.no_grad():
            try:
                outputs = model(**inputs_b.to(args.device), labels=labels_b.to(args.device))
            except:
                print(inputs_b)
                print(labels_b)

        logits = outputs.logits
        _, pred = logits.max(dim=2)

        mask_pred = torch.gather(pred, 1, mask_inds.unsqueeze(1))
        mask_gt = torch.gather(labels_b.to(args.device), 1, mask_inds.unsqueeze(1))
        correct = mask_pred.eq(mask_gt)

        id_f, _ = torch.where(correct==False)
        id_t, _ = torch.where(correct==True)
        incorrect_id.extend(list(id_f.cpu().numpy() + i * 100))
        incorrect_predictions.extend(list(mask_pred.view(-1).cpu().numpy()))
        correct_id.extend(list(id_t.cpu().numpy() + i * 100))

    return incorrect_id, incorrect_predictions, correct_id

        
def get_dataset_stats(all_samples, model, tokenizer, args):
    ### run inference
    samples_batches, sentences_batches, label_batches = batchify(all_samples, 1)

    iter_time = AverageMeter()
    end = time.time()
    for i in range(len(samples_batches)):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        gt_b = label_batches[i]

        # zero-grad
        for param in model.parameters():
            param.grad = None

        inputs_b = tokenizer(sentences_b, return_tensors='pt')
        labels_b = tokenizer(gt_b, return_tensors='pt')["input_ids"]
        labels_b[inputs_b["input_ids"] != tokenizer.mask_token_id] = -100 # only calculate loss on masked tokens

        outputs = model(**inputs_b.to(args.device), labels=labels_b.to(args.device))
        loss = outputs.loss
        logits = outputs.logits

        # compute gradient profile
        loss.backward()
        saliency_profile = sort_grads(model, args.aggr)
        
        if i == 0:
            print(saliency_profile.shape)
            # oldM in Welford's method (https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
            testset_mean_sal_prev = torch.zeros_like(saliency_profile, dtype=torch.float64)
            testset_mean_sal = saliency_profile / float(i+1)
            # print(testset_mean_abs_grad)
            testset_std_sal = (saliency_profile - testset_mean_sal) * (saliency_profile - testset_mean_sal_prev)
            # print(naive_saliency.shape)
        else:
            testset_mean_sal_prev = testset_mean_sal.detach().clone()  # oldM
            testset_mean_sal += (saliency_profile - testset_mean_sal) / float(i+1)  # update M to the current
            # print(testset_mean_abs_grad)
            testset_std_sal += (saliency_profile - testset_mean_sal) * (saliency_profile - testset_mean_sal_prev)  # update variance

        iter_time.update(time.time() - end)
        end = time.time()
        if (i)%50 == 0:
            remain_time = (len(samples_batches) - i - 1) * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            print("Iter: [{:d}/{:d}]\t iter time: {iter_time.val: .3f}\t remain time: {remain_time}".format(
                            i+1, len(samples_batches), iter_time=iter_time, remain_time=remain_time))

    testset_std_sal = testset_std_sal / float(len(samples_batches) - 1)  # Unbiased estimator of variance
    print('Variance:', testset_std_sal)
    testset_std_sal = torch.sqrt(testset_std_sal)
    print('Std:', testset_std_sal)
    print('Mean:', testset_mean_sal)
    print('Testset_grads_shape:{}'.format(testset_mean_sal.shape))

    return testset_mean_sal, testset_std_sal

def sample_saliency_curves(all_samples, model, tokenizer, testset_mean, testset_std, args):
    saliency_curves = []
    ### run inference
    samples_batches, sentences_batches, label_batches = batchify(all_samples, args.batch_size)
    
    iter_time = AverageMeter()
    end = time.time()
    for i in range(len(samples_batches)):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        gt_b = label_batches[i]

        # zero-grad
        for param in model.parameters():
            param.grad = None

        inputs_b = tokenizer(sentences_b, return_tensors='pt')
        labels_b = tokenizer(gt_b, return_tensors='pt')["input_ids"]
        labels_b[inputs_b["input_ids"] != tokenizer.mask_token_id] = -100 # only calculate loss on masked tokens

        outputs = model(**inputs_b.to(args.device), labels=labels_b.to(args.device))
        loss = outputs.loss
        logits = outputs.logits

        # compute gradient profile
        loss.backward()
        saliency_profile = sort_grads(model, args.aggr)


        saliency_curves.append((saliency_profile - testset_mean) / testset_std)

        iter_time.update(time.time() - end)
        end = time.time()
        if (i+1)%50 == 0:
            remain_time = (len(samples_batches) - i - 1) * iter_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            print("Iter: [{:d}/{:d}]\t iter time: {iter_time.val: .3f}\t remain time: {remain_time}".format(
                            i+1, len(samples_batches), iter_time=iter_time, remain_time=remain_time))

    return torch.stack(saliency_curves)


def debug_padding(all_samples, model, tokenizer, len_samples, pad, args):
    saliency_curves = []
    ### run inference
    samples_batches, sentences_batches, label_batches = batchify(all_samples, 1)
    for i in range(len_samples):
        samples_b = samples_batches[i]
        sentences_b = sentences_batches[i]
        gt_b = label_batches[i]

        # zero-grad
        for param in model.parameters():
            param.grad = None

        inputs_b = tokenizer(sentences_b, padding='max_length' if pad==True else False, max_length=10, return_tensors='pt')
        labels_b = tokenizer(gt_b, padding='max_length' if pad==True else False, max_length=10, return_tensors='pt')["input_ids"]
        labels_b[inputs_b["input_ids"] != tokenizer.mask_token_id] = -100 # only calculate loss on masked tokens
        print('inputs:', inputs_b)
        print('labels_b:', labels_b)


        outputs = model(**inputs_b.to(args.device), labels=labels_b.to(args.device))
        loss = outputs.loss
        logits = outputs.logits

        # compute gradient profile
        loss.backward()
        saliency_profile = sort_grads(model, args.aggr)

        saliency_curves.append(saliency_profile)

    return torch.stack(saliency_curves)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Saliency curve comparison')
    parser.add_argument('--bert_model_name', default='bert-base-cased', type=str)
    parser.add_argument('--dataset_name', default='/cmlscratch/manlis/data/lama/Squad/test.jsonl', type=str)
    parser.add_argument('--max_seq_length', default=1024, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--aggr', default="column-wise", type=str, choices=['naive', 'column-wise'])
    args = parser.parse_args()

    ### set up the model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, cache_dir=CACHE_DIR)

    config = AutoConfig.from_pretrained(args.bert_model_name, cache_dir=CACHE_DIR)
    model =  AutoModelForMaskedLM.from_pretrained(
            args.bert_model_name,
            config=config,
            cache_dir=CACHE_DIR,
        )
    model.resize_token_embeddings(len(tokenizer))

    ### load data TODO: add an outer loop for relations
    data = load_file(args.dataset_name)
    print(len(data))

    template = ""  # TODO: add a lookup dict for relation-template pairs
    all_samples, ret_msg = filter_samples(
        model, tokenizer, data, args.max_seq_length, template
    )
    print(ret_msg)
    print(len(all_samples))

    if template != "":
        all_samples = apply_template(all_samples, template)

    # create uuid if not present
    i = 0
    for sample in all_samples:
        if "uuid" not in sample:
            sample["uuid"] = i
        i += 1


    ### unit tests

    # incorrect_id, incorrect_pred, correct_id = test_and_find_incorrect_prediction(all_samples, model, tokenizer, args)
    # print("incorrect: ", len(incorrect_id))

    # testset_mean_sal, testset_std_sal = get_dataset_stats(all_samples, model, tokenizer, args)
    # entries = torch.where(testset_std_sal<1e-14, torch.tensor(1.0), torch.tensor(0.0))
    # print(torch.sum(entries))

    non_padded_curves = debug_padding(all_samples, model, tokenizer, 1, False, args)
    padded_curves = debug_padding(all_samples, model, tokenizer, 1, True, args)
    assert torch.is_nonzero(non_padded_curves.sum())
    assert torch.equal(non_padded_curves, padded_curves)
