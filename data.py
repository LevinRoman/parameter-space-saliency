import json


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

# no need to filter out labels that are not in the vocab, because only one vocab.txt is used
def filter_samples(model, tokenizer, samples, max_sentence_length, template):
    msg = ""
    new_samples = []
    samples_exluded = 0
    for sample in samples:
        excluded = False
        if "obj_label" in sample or "sub_label" in sample:
            # obj_label_ids = tokenizer(sample["obj_label"])

            # if obj_label_ids:
            #     recostructed_word = " ".join(
            #         [model.vocab[x] for x in obj_label_ids]
            #     ).strip()
            # else:
            #     recostructed_word = None

            excluded = False
            if not template or len(template) == 0:
                masked_sentences = sample["masked_sentences"]
                text = " ".join(masked_sentences)
                if len(text.split()) > max_sentence_length:
                    msg += "\tEXCLUDED for exeeding max sentence length: {}\n".format(
                        masked_sentences
                    )
                    samples_exluded += 1
                    excluded = True
                elif text.count("[MASK]") > 1:
                    msg += "\tEXCLUDED for having more than one mask: {}\n".format(
                        masked_sentences
                    )
                    samples_exluded += 1
                    excluded = True
                elif sample['obj_label'] not in tokenizer.get_vocab():
                    msg += "\tEXCLUDED object label {} not in vocab subset\n".format(sample['obj_label'])
                    samples_exluded+=1
                    excluded = True

            if excluded:
                pass

            elif "judgments" in sample:
                # only for Google-RE
                num_no = 0
                num_yes = 0
                for x in sample["judgments"]:
                    if x["judgment"] == "yes":
                        num_yes += 1
                    else:
                        num_no += 1
                if num_no > num_yes:
                    # SKIP NEGATIVE EVIDENCE
                    pass
                else:
                    new_samples.append(sample)
            else:
                new_samples.append(sample)
        else:
            msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                sample
            )
            samples_exluded += 1
    msg += "samples exluded  : {}\n".format(samples_exluded)
    return new_samples, msg

def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]

def apply_template(all_samples, template):
    facts = []
    for sample in all_samples:
        sub = sample["sub_label"]
        obj = sample["obj_label"]
        if (sub, obj) not in facts:
            facts.append((sub, obj))
    local_msg = "distinct template facts: {}".format(len(facts))
    print(local_msg)
    all_samples = []
    for fact in facts:
        (sub, obj) = fact
        sample = {}
        sample["sub_label"] = sub
        sample["obj_label"] = obj
        # sobstitute all sentences with a standard template
        sample["masked_sentences"] = parse_template(
            template.strip(), sample["sub_label"].strip(), "[MASK]"
        )
        all_samples.append(sample)
    return all_samples

def batchify(data, batch_size):
    list_samples_batches = []
    list_sentences_batches = []
    list_label_batches = []
    current_samples_batch = []
    current_sentences_batches = []
    current_label_batches = []
    c = 0

    # sort to group togheter sentences with similar length
    for sample in sorted(
        data, key=lambda k: len(" ".join(k["masked_sentences"]).split())
    ):
        masked_sentence = sample["masked_sentences"][0]
        current_samples_batch.append(sample)
        current_sentences_batches.append(masked_sentence)
        current_label_batches.append(masked_sentence.replace("[MASK]", " {} ".format(sample["obj_label"])))
        c += 1
        if c >= batch_size:
            list_samples_batches.append(current_samples_batch)
            list_sentences_batches.append(current_sentences_batches)
            list_label_batches.append(current_label_batches)
            current_samples_batch = []
            current_sentences_batches = []
            current_label_batches = []
            c = 0

    # last batch
    if current_samples_batch and len(current_samples_batch) > 0:
        list_samples_batches.append(current_samples_batch)
        list_sentences_batches.append(current_sentences_batches)
        list_label_batches.append(current_label_batches)

    return list_samples_batches, list_sentences_batches, list_label_batches