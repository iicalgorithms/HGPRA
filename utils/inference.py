import time
import torch
import logging
import torch.nn.functional as F
from copy import deepcopy
from utils import init_parameters
from metrics import Eval_Metrics


def train_epoch(net, data, train_idx, optimizer, epoch, gradients=None, return_emb=False):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    outs, emb_x, emb_edge = net(data, return_emb=return_emb)
    outs = F.log_softmax(outs, dim=1)
    loss = F.nll_loss(outs[train_idx], data.y[train_idx])
    loss.backward()

    if gradients is not None:
        gradients.append([p.grad.detach().cpu().clone() for p in net.parameters() if p.grad is not None])

    optimizer.step()
    logging.info(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return (loss.item(), emb_x, emb_edge) if return_emb else loss.item()


@torch.no_grad()
def infer_epoch(net, data, idx, test=False, return_emb=False):
    net.eval()
    outs, emb_x, emb_edge = net(data, return_emb=return_emb)
    outs = F.log_softmax(outs, dim=1)
    res = Eval_Metrics(data.y[idx].cpu().numpy(), outs[idx].cpu().numpy())
    return (res, emb_x, emb_edge) if return_emb else res


def train_on_condensed_eval_whole(net, sub_data, data, val_idx, test_idx, eval_optimizer, eval_epochs, early_stop=50):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sub_data, data = sub_data.to(device), data.to(device)

    net = net.to(device)
    net.reset_parameters()

    best_state = None
    best_val_acc = 0
    best_val = None
    patience = 0
    for epoch in range(eval_epochs):
        # train
        epoch_loss = train_epoch(net, sub_data, sub_data.train_idx, eval_optimizer, epoch)
        with torch.no_grad():
            val_res = infer_epoch(net, data, val_idx)
        if val_res.acc > best_val_acc:
            logging.info(f"epoch: {epoch}, train loss: {epoch_loss:.4f}, update best: {val_res.acc:.5f}")
            best_val_acc = val_res.acc
            best_val = val_res
            best_state = deepcopy(net.state_dict())
        else:
            patience += 1
            if early_stop and patience >= early_stop:
                break
    net.load_state_dict(best_state)
    test_res = infer_epoch(net, data, test_idx)
    return best_val, test_res


def train_whole(net, X, G, labels, train_idx, val_idx, optimizer, num_epochs, early_stop=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    X, labels = X.to(device), labels.to(device)
    G = G.to(device)

    net = net.to(device)
    init_parameters(net)

    best_state = None
    best_val = 0
    patience = 0
    for epoch in range(num_epochs):
        # train
        epoch_loss = train_epoch(net, X, G, labels, train_idx, optimizer, epoch)
        with torch.no_grad():
            val_res = infer_epoch(net, X, G, labels, val_idx)
        if val_res.acc > best_val:
            logging.info(f"epoch: {epoch}, train loss: {epoch_loss:.4f}, update best: {val_res.acc:.5f}")
            best_val = val_res.acc
            best_state = deepcopy(net.state_dict())
        else:
            patience += 1
            if early_stop and patience >= early_stop:
                break
    
    net.load_state_dict(best_state)
    return net
