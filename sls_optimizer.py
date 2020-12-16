import torch
import time
import contextlib
import numpy as np
import copy


@contextlib.contextmanager
def rng_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(gpu_rng_state, device)

def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1, init_step_size=None):
    if reset_option == 0:
        pass
    elif reset_option == 1:
        step_size = step_size * gamma**(1. / n_batches_per_epoch)
    elif reset_option == 2:
        step_size = init_step_size
    return step_size

def find_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm

class Sls(torch.optim.Optimizer):
    """ docstring
    """
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 beta_f=2.0,
                 reset_option=1,
                 eta_max=10,
                 bound_step_size=True,
                 line_search_fn="armijo"):
        defaults = dict(n_batches_per_epoch=n_batches_per_epoch, 
                        init_step_size=init_step_size,
                        c=c,
                        beta_b=beta_b, 
                        gamma=gamma, 
                        beta_f=beta_f, 
                        reset_option=reset_option,
                        eta_max=eta_max,
                        bound_step_size=bound_step_size,
                        line_search_fn=line_search_fn)
        super().__init__(params, defaults)
        self.state['step'] = 0
        self.state['step_size'] = init_step_size
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        
    def step(self, closure):
        seed = time.time()
        def closure_deterministic():
            with rng_torch(int(seed)):
                return closure()

        batch_step_size = self.state['step_size']
        loss = closure_deterministic()
        loss.backward()

        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        for group in self.param_groups:
            params = group['params']
            params_current = copy.deepcopy(params)
            grad_current = [p.grad for p in params]
            grad_norm = find_grad_norm(grad_current)
            step_size = reset_step(step_size=batch_step_size,
                                   n_batches_per_epoch=group['n_batches_per_epoch'],
                                   gamma=group['gamma'],
                                   reset_option=group['reset_option'],
                                   init_step_size=group['init_step_size'])
            
            with torch.no_grad():
                if grad_norm >= 1e-8:
                    found = 0
                    step_size_old = step_size

                    for _ in range(100):
                        for p_next, p_current, g_current in zip(params, params_current, grad_current):
                            p_next.data = p_current - step_size * g_current
                        loss_next = closure_deterministic()
                        self.state['n_forwards'] += 1

                        if group['line_search_fn'] == "armijo":
                            break_condition = loss_next - (loss - (step_size) * group['c'] * grad_norm * 2)
                            if (break_condition <= 0):
                                found = 1
                            else:
                                step_size = step_size * group['beta_b']
                            if found == 1:
                                break
                        
                    if found == 0:
                        for p_next, p_current, g_current in zip(params, params_current, grad_current):
                            p_next.data = p_current - 1e-6 * g_current
        
            self.state['step_size'] = step_size
            self.state['step'] += 1
        
        return loss