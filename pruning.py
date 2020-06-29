import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_params(net, bias=False):
    """Gets the parameters of the network."""
    params = []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            params.append(m.weight)
        if bias:
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    params.append(m.bias)
    return params


def mask_net(net, mask, bias=False):
    """Masks the network inplace."""
    for p, m in zip(get_params(net, bias), mask):
        p.data.mul_(m.float())


def get_mask_iteratively(method, net, prunings, loader=None, n=float('inf'),
                         reg=0.0, previous_mask=None):
    """Iteratively prune using pruning ratios specified in prunings.

    E.g. [0.2, 0.4, 0.6, 0.8] to prune 80 % of the network in 4 iterations.

    """
    for pruning in prunings:
        previous_mask = get_mask(method, net, pruning, loader, n, reg=reg,
                                 previous_mask=previous_mask)
    return previous_mask


def get_mask(method, net, pruning, loader=None, n=float('inf'), reg=0.0,
             previous_mask=None):
    """Returns the maks computed on net by method."""
    # Prepare network
    net = copy.deepcopy(net)
    net.eval()
    net.zero_grad()
    if previous_mask is not None:
        mask_net(net, previous_mask)
    params = get_params(net)
    if previous_mask is None:
        previous_mask = [torch.ones_like(p) for p in params]
    if method == 'MP':
        grads = [torch.zeros_like(p) for p in params]
        diags = [torch.ones_like(p) for p in params]
    elif method == 'LM':
        grads = GradComputer().compute(net, loader, n=n)
        diags = [torch.zeros_like(p) for p in params]
    elif method in 'QM':
        grads, diags = DiagonalGGNComputer().compute(net, loader, n=n)
    elif method == 'OBD':
        grads = [torch.zeros_like(p) for p in params]
        _, diags = DiagonalGGNComputer().compute(net, loader, n=n)
    else:
        raise NotImplementedError

    ranks = compute_ranks(params, diags, grads, previous_mask, reg=reg)
    masks = global_diagonal_pruning(ranks, pruning)
    return masks


def compute_ranks(params, diags, grads, masks, reg=0.0):
    """Computes the ranks, using diag and optional gradient term."""
    ranks = []
    for p, d, g, m in zip(params, diags, grads, masks):
        r = (g * (-p) + 0.5 * d * p ** 2).abs_() + 0.5 * reg * p ** 2
        r[m == 0] = -float('inf')
        ranks.append(r)
    return ranks


def global_diagonal_pruning(ranks, pruning):
    """Compute pruning using pre-computed ranks."""
    rank = torch.cat([r.view(-1) for r in ranks])
    _, idx = rank.topk(int(round(rank.numel() * pruning)), largest=False)
    mask = torch.ones_like(rank)
    mask[idx] = 0
    mask = torch.split(mask, [r.numel() for r in ranks])
    mask = [m.view(*r.shape) for m, r in zip(mask, ranks)]
    return mask


class GradComputer():
    """Computes gradients."""

    def compute(self, net, loader, device='cuda', n=float('inf')):
        params = get_params(net, False)
        gradients = [torch.zeros_like(p) for p in params]
        criterion = torch.nn.CrossEntropyLoss()
        counter = 0
        # Processing loop
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(net(inputs), targets)
            loss.backward()
            for g, p in zip(gradients, params):
                g += p.grad.detach()
            net.zero_grad()
            counter += inputs.size(0)
            if counter >= n:
                break
        gradients = [g / float(i + 1) for g in gradients]
        return gradients


class DiagonalGGNComputer():
    """Computes gradients and diagonal of GGN."""

    def _save_input(self, mod, i):
        if isinstance(mod, nn.Linear):
            self.state[mod] = i[0] ** 2
        elif isinstance(mod, nn.Conv2d):
            x = F.unfold(i[0], mod.kernel_size, mod.dilation, mod.padding,
                         mod.stride)
            x = x.permute(0, 2, 1).contiguous()
            self.state[mod] = x
        else:
            raise NotImplementedError

    def _compute_grads(self, mod, grad_input, grad_output):
        gy = grad_output[0] * grad_output[0].size(0)
        x = self.state[mod]
        # Getting gradients on the eventual biases
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            if mod.bias is not None:
                if gy.dim() == 4:
                    gb = gy.sum(3).sum(2)
                else:
                    gb = gy
                gb = gb.pow(2).mean(0)
                if mod.bias in self.diags:
                    self.diags[mod.bias] += gb
                else:
                    self.diags[mod.bias] = gb
        # Now for the weights
        if isinstance(mod, nn.Linear):
            gw = torch.mm((gy ** 2).t(), x) / x.size(0)
        elif isinstance(mod, nn.Conv2d):
            gw = torch.bmm(gy.flatten(2), x)
            gw = gw.view(gy.size(0), *mod.weight.size())
            gw = gw.pow_(2).mean(0)
        if mod.weight in self.diags:
            self.diags[mod.weight] += gw
        else:
            self.diags[mod.weight] = gw

    def hook_net(self, net):
        self.diags = {}
        self.state = {}
        handles = []
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                handle = m.register_forward_pre_hook(self._save_input)
                handles.append(handle)
                handle = m.register_backward_hook(self._compute_grads)
                handles.append(handle)
        return handles

    def unhook_net(self, handles):
        del self.state
        del self.diags
        for handle in handles:
            handle.remove()

    def compute(self, net, loader, device='cuda', n=float('inf')):
        params = get_params(net, False)
        gradients = [torch.zeros_like(p) for p in params]
        diags_ggn = [torch.zeros_like(p) for p in params]
        handles = self.hook_net(net)
        counter = 0
        # Processing loop
        for i, (inputs, targets) in enumerate(loader):
            net.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            log_probs = F.log_softmax(outs, dim=1)

            # Classical backprop to get the gradients
            loss = F.nll_loss(log_probs, targets)
            loss.backward(retain_graph=True)
            for g, p in zip(gradients, params):
                g += p.grad.detach()
            self.diags = {}  # clean buffer

            # Backprops for the GGN
            probs = F.softmax(outs, dim=1).detach_()
            for j in range(probs.size(-1)):
                (log_probs[:, j] * probs[:, j].sqrt()).mean().backward(retain_graph=True)
                net.zero_grad()
            for d, p in zip(diags_ggn, params):
                d += self.diags[p].detach()
            self.state = {}
            self.diags = {}
            counter += inputs.size(0)
            if counter >= n:
                break
        gradients = [g / float(i + 1) for g in gradients]
        diags_ggn = [d / float(i + 1) for d in diags_ggn]
        self.diags = {}
        self.unhook_net(handles)
        return gradients, diags_ggn
