import torch


def sort_grads(model, method):
    grads = []
    # print(model.named_parameters)
    for param in model.named_parameters():
        name = param[0]
        content = param[1]
        grad = content.grad.detach()
        # print(name)
        if method == 'column-wise':
            if "attention.self" in name:
                if "Norm" not in name:
                    # print(name)
                    if len(grad.size()) == 1:
                        grads.append(grad.detach().abs().view(-1).cpu())
                    elif len(grad.size()) == 2:
                        grads.append(grad.detach().abs().mean(-1).view(-1).cpu())
        elif method == 'naive':
            grads.append(grad.detach().abs().view(-1).cpu())
    grads = torch.cat(grads)
    return grads


def truncate_dataset(filename):
    nf_1 = open(filename.replace("test.jsonl", "test_1.jsonl"), "w")
    nf_2 = open(filename.replace("test.jsonl", "test_2.jsonl"), "w")
    nf_3 = open(filename.replace("test.jsonl", "test_3.jsonl"), "w")
    nf_4 = open(filename.replace("test.jsonl", "test_4.jsonl"), "w")
    nf_5 = open(filename.replace("test.jsonl", "test_5.jsonl"), "w")

    with open(filename, "r") as f:
        lines = f.readlines()
        nf_1.writelines(lines[0::5])
        nf_2.writelines(lines[1::5])
        nf_3.writelines(lines[2::5])
        nf_4.writelines(lines[3::5])
        nf_5.writelines(lines[4::5])

    nf_1.close()
    nf_2.close()
    nf_3.close()
    nf_4.close()
    nf_5.close()

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

if __name__=='__main__':
    truncate_dataset("/cmlscratch/manlis/data/LAMA/ConceptNet/test.jsonl")
            
    

