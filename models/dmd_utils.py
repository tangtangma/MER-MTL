"""
DMD Utility Functions
From DMD original trains package.

Functions:
- to_numpy: Convert tensor/Variable to numpy
- squeeze/unsqueeze: Array shape manipulation
- is_due: Determine if action should be performed at current epoch/step
- softmax: Softmax with temperature
- distance_metric: Imitation learning distance metrics (cosine/l2/l1/kl)
- min_cosine: Minimum cosine distance for distillation
- get_segments: Split input into overlapping segments
- get_stats: Calculate accuracy
- get_stats_detection: Calculate accuracy and average precisions for detection
- info/warn/err: Colored console output
"""
import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
import torch.utils.data


def to_numpy(array):
    """Convert tensor/Variable to numpy array."""
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.autograd.Variable):
        array = array.data
    if array.is_cuda:
        array = array.cpu()
    return array.numpy()


def squeeze(array):
    """Remove unnecessary dimensions."""
    if not isinstance(array, list) or len(array) > 1:
        return array
    else:
        # len(array) == 1:
        return array[0]


def unsqueeze(array):
    """Add dimension if not already a list."""
    if isinstance(array, list):
        return array
    else:
        return [array]


def is_due(*args):
    """
    Determines whether to perform an action or not, depending on the epoch.
    Used for logging, saving, learning rate decay, etc.
    
    Args:
        *args: 
            epoch, due_at (due at epoch due_at)
            epoch, num_epochs, due_every (due every due_every epochs)
            step, due_every (due every due_every steps)
    
    Returns:
        due: boolean: perform action or not
    """
    if len(args) == 2 and isinstance(args[1], list):
        epoch, due_at = args
        due = epoch in due_at
    elif len(args) == 3:
        epoch, num_epochs, due_every = args
        due = (due_every >= 0) and (epoch % due_every == 0 or epoch == num_epochs)
    else:
        step, due_every = args
        due = (due_every > 0) and (step % due_every == 0)
    return due


def softmax(w, t=1.0, axis=None):
    """Softmax function with temperature."""
    w = np.array(w) / t
    e = np.exp(w - np.amax(w, axis=axis, keepdims=True))
    dist = e / np.sum(e, axis=axis, keepdims=True)
    return dist


def min_cosine(student, teacher, option='cosine', weights=None):
    """
    Minimum cosine distance for distillation.
    
    Used in Hetero GD (cross-modal distillation) where we want to maximize
    the minimum similarity between representations.
    """
    cosine = torch.nn.CosineEmbeddingLoss()
    dists = cosine(student, teacher.detach(), torch.tensor([-1]).cuda() if student.is_cuda else torch.tensor([-1]))
    if weights is None:
        dist = dists.mean()
    else:
        dist = (dists * weights).mean()
    return dist


def distance_metric(student, teacher, option='cosine', weights=None):
    """
    Distance metric to calculate the imitation loss.
    
    Used in Homo GD (same-modal distillation) for both logits and representations.
    
    Args:
        student: batch_size x n_classes (or batch_size x hidden_size for repr)
        teacher: batch_size x n_classes (or batch_size x hidden_size for repr)
        option: one of [cosine, l2, l1, kl]
        weights: batch_size or float
    
    Returns:
        The computed distance metric.
    """
    if option == 'cosine':
        dists = 1 - F.cosine_similarity(student, teacher.detach(), dim=1)
    elif option == 'l2':
        dists = (student - teacher.detach()).pow(2).sum(1)
    elif option == 'l1':
        dists = torch.abs(student - teacher.detach()).sum(1)
    elif option == 'kl':
        # KL divergence with temperature
        T = 8  # averaged for each minibatch
        dist = F.kl_div(
            F.log_softmax(student / T),
            F.softmax(teacher.detach() / T)
        ) * (T * T)
        return dist
    else:
        raise NotImplementedError(f"Unknown distance metric option: {option}")
    
    if weights is None:
        dist = dists.mean()
    else:
        dist = (dists * weights).mean()
    return dist


def get_segments(input, timestep):
    """
    Split entire input into segments of length timestep.
    
    Used during inference for handling variable-length inputs.
    
    Args:
        input: 1 x total_length x n_frames x ...
        timestep: the timestamp
    
    Returns:
        input: concatenated video segments
        start_indices: indices of the segments
    """
    assert input.size(0) == 1, 'Test time, batch_size must be 1'
    input.squeeze_(dim=0)
    
    # Find overlapping segments
    length = input.size()[0]
    step = timestep // 2
    num_segments = (length - timestep) // step + 1
    
    start_indices = (np.arange(num_segments) * step).tolist()
    if length % step > 0:
        start_indices.append(length - timestep)
    
    # Get the segments
    segments = []
    for s in start_indices:
        segment = input[s:(s + timestep)].unsqueeze(0)
        segments.append(segment)
    
    input = torch.cat(segments, dim=0)
    return input, start_indices


def get_stats(logit, label):
    """Calculate the accuracy."""
    logit = to_numpy(logit)
    label = to_numpy(label)
    pred = np.argmax(logit, 1)
    acc = np.sum(pred == label) / label.shape[0]
    return acc, pred, label


def get_stats_detection(logit, label, n_classes=52):
    """
    Calculate the accuracy and average precisions for detection tasks.
    
    Args:
        logit: model predictions (batch_size, n_classes)
        label: ground truth labels (batch_size,)
        n_classes: total number of classes (including background)
    
    Returns:
        ap: mean average precision
        acc: overall accuracy
        acc_bg: background class accuracy
        acc_action: action class accuracy
        ratio_bg: ratio of background samples
        pred: predicted labels
        label: ground truth labels
    """
    logit = to_numpy(logit)
    label = to_numpy(label)
    scores = softmax(logit, axis=1)
    pred = np.argmax(logit, 1)
    length = label.shape[0]
    
    acc = np.sum(pred == label) / length
    
    # Background class metrics
    keep_bg = label == 0
    acc_bg = np.sum(pred[keep_bg] == label[keep_bg]) / label[keep_bg].shape[0]
    ratio_bg = np.sum(keep_bg) / length
    
    # Action class metrics
    keep_action = label != 0
    acc_action = np.sum(pred[keep_action] == label[keep_action]) / label[keep_action].shape[0]
    
    # Average precision
    y_true = np.zeros((len(label), n_classes))
    y_true[np.arange(len(label)), label] = 1
    aps = average_precision_score(y_true, scores, average=None)
    aps = list(filter(lambda x: not np.isnan(x), aps))
    ap = np.mean(aps)
    
    return ap, acc, acc_bg, acc_action, ratio_bg, pred, label


def info(text):
    """Print info message in blue."""
    print('\033[94m' + text + '\033[0m')


def warn(text):
    """Print warning message in yellow."""
    print('\033[93m' + text + '\033[0m')


def err(text):
    """Print error message in red."""
    print('\033[91m' + text + '\033[0m')


if __name__ == '__main__':
    """Test utility functions."""
    print("Testing DMD Utils...")
    
    # Test to_numpy
    tensor = torch.randn(3, 4)
    arr = to_numpy(tensor)
    print(f"to_numpy: shape={arr.shape}")
    
    # Test softmax
    w = [1.0, 2.0, 3.0]
    s = softmax(w, t=1.0)
    print(f"softmax: {s}")
    
    # Test distance_metric
    student = torch.randn(8, 5)
    teacher = torch.randn(8, 5)
    dist = distance_metric(student, teacher, 'cosine')
    print(f"distance_metric (cosine): {dist.item():.4f}")
    
    # Test get_stats
    logit = torch.randn(10, 3)
    label = torch.randint(0, 3, (10,))
    acc, pred, label = get_stats(logit, label)
    print(f"get_stats: acc={acc:.4f}")
    
    print("\n✓ All utils tests passed!")
