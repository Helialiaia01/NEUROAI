"""Trial-aware CEBRA adapters. Neural padding and label dynamics are separate."""
import types
import numpy as np
import torch


def validate_trials(trial_ids, trial_length, time_ids=None):
    ids = np.asarray(trial_ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or trial_length < 1:
        raise ValueError('Expected nonempty 1D trial IDs and positive trial length')
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    if len(np.unique(ids[starts])) != len(starts) or np.any(ends-starts != trial_length):
        raise ValueError('Each trial must be one contiguous block of trial_length samples')
    if time_ids is not None and not np.array_equal(time_ids, np.tile(np.arange(trial_length), len(starts))):
        raise ValueError('Time IDs must run from zero to trial_length-1 in each trial')
    return ids, starts, ends


def install_trial_safe_expander(dataset, trial_ids, trial_length):
    ids, starts, ends = validate_trials(trial_ids, trial_length)
    device = dataset.neural.device
    lower = torch.as_tensor(np.repeat(starts, ends-starts), device=device)
    upper = torch.as_tensor(np.repeat(ends-1, ends-starts), device=device)
    left, right = int(dataset.offset.left), int(dataset.offset.right)
    if trial_length < left + right:
        raise ValueError('Trial shorter than the encoder receptive field')
    delta = torch.arange(-left, right, device=device)

    def expand_index(self, index):
        centers = torch.as_tensor(index, dtype=torch.long, device=device).reshape(-1)
        if torch.any(centers < 0) or torch.any(centers >= len(ids)):
            raise IndexError('Sample outside trial data')
        # Replicate edge samples, preserving the sampled centre/label index.
        return (centers[:, None] + delta).maximum(lower[centers, None]).minimum(upper[centers, None])

    dataset.expand_index = types.MethodType(expand_index, dataset)


def install_trial_safe_differences(loader, trial_ids):
    distribution = getattr(loader, 'distribution', None)
    if not hasattr(distribution, 'time_difference'):
        return 0
    if not hasattr(distribution, 'time_delta'):
        raise TypeError('Unsupported continuous distribution: missing time_delta')
    lag = int(distribution.time_delta)
    ids = np.asarray(trial_ids)
    if lag < 1 or lag >= len(ids):
        raise ValueError('Invalid time-difference offset')
    valid = np.flatnonzero(ids[lag:] == ids[:-lag]) + lag
    if not len(valid):
        raise ValueError('No within-trial label transitions at configured time offset')
    data = distribution.data
    index = torch.as_tensor(valid, device=data.device)
    distribution.time_difference = data[index] - data[index-lag]
    return len(valid)
