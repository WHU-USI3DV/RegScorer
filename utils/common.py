import os
import os.path as osp
import pickle


def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def get_print_format(value):
    if isinstance(value, int):
        return 'd'
    if isinstance(value, str):
        return 's'
    if value == 0:
        return '.3f'
    if value < 1e-6:
        return '.3e'
    if value < 1e-3:
        return '.6f'
    return '.3f'


def get_format_strings(kv_pairs):
    r"""Get format string for a list of key-value pairs."""
    log_strings = []
    for key, value in kv_pairs:
        fmt = get_print_format(value)
        format_string = '{}: {:' + fmt + '}'
        log_strings.append(format_string.format(key, value))
    return log_strings


def get_log_string(result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, timer=None):
    log_strings = []
    if epoch is not None:
        epoch_string = f'Epoch: {epoch}'
        if max_epoch is not None:
            epoch_string += f'/{max_epoch}'
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f'iter: {iteration}'
        if max_iteration is not None:
            iter_string += f'/{max_iteration}'
        if epoch is None:
            iter_string = iter_string.capitalize()
        log_strings.append(iter_string)
    if 'metadata' in result_dict:
        log_strings += result_dict['metadata']
    for key, value in result_dict.items():
        if key != 'metadata':
            format_string = '{}: {:' + get_print_format(value) + '}'
            log_strings.append(format_string.format(key, value))
    if lr is not None:
        log_strings.append('lr: {:.3e}'.format(lr))
    if timer is not None:
        log_strings.append(timer.tostring())
    message = ', '.join(log_strings)
    return message

class ExpDecayLR():
    def __init__(self,lr_init, lr_decay_rate, decay_step):
        self.lr_init=lr_init 
        self.decay_step=decay_step
        self.decay_rate=lr_decay_rate

    def __call__(self, step, *args, **kwargs):
        return self.lr_init*(self.decay_rate**(step//self.decay_step))
    
def reset_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        # print(param_group)
        # lr_before = param_group['lr']
        param_group['lr'] = lr
    # print('changing learning rate {:5f} to {:.5f}'.format(lr_before,lr))
    return lr