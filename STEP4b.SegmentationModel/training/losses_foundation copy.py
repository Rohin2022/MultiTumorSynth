import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import HungarianAlgorithm as HA
import sys
import os
import yaml
import nibabel as nib
import math


def dilate_volume(volume, kernel_size, full_pass_radius=3):
    # ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # for small kernels, just do one pass
    if kernel_size <= (2*full_pass_radius+1):
        return dilate_volume_conv(volume, kernel_size)

    # compute how many "radius‑3" (kernel=7) passes we need
    # radius = (kernel_size‑1)//2  (an integer number of voxels)
    radius = (kernel_size - 1) // 2

    num_full = radius // full_pass_radius  # integer division
    remainder = radius % full_pass_radius  # 0, 1, or 2

    # apply all full radius‑3 passes
    for _ in range(num_full):
        volume = dilate_volume_conv(volume, 2*full_pass_radius + 1)

    # handle the leftover radius if any (1→kernel=3, 2→kernel=5)
    if remainder > 0:
        volume = dilate_volume_conv(volume, 2*remainder + 1)

    return volume



def dilate_volume_conv(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        kernel_size+=1



    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    #we can use a maxpool or a ball convolution to dilate the volume. Maxpool should be faster, but it uses a cube kernel, while the ball kernel is more accurate.
    #dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)
    ball_kernel = create_ball_kernel(kernel_size).type_as(volume).unsqueeze(0).unsqueeze(0).repeat(volume.shape[1],1, 1, 1, 1)

    # Calculate padding such that the output size is the same as the input size.
    kernel_size = ball_kernel.shape[-1]
    padding = kernel_size // 2

    dilated = F.conv3d(volume, ball_kernel, padding=padding, groups=volume.shape[1])
    #binarize
    dilated = (dilated > 0).float()

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated

def dilate_volume_maxpool(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    kernel_size = max(1,int(kernel_size/(2**(0.5))))#compensates for the fact that maxpool is not a round kernel
    if kernel_size%2==0:
        kernel_size+=1

    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number for proper alignment.")

    # Calculate padding such that the output size is the same as the input size.
    padding = kernel_size // 2


    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated

counter = 0

def get_known_voxels(y: torch.Tensor, unk_voxels: torch.Tensor, dilation=5,sanity=True, classes = None):
    """
    We cannot calculate the BCE loss for voxels we do not know the ground-truth for.
    This function will output a per-voxel masks showing the known voxels. You can use it to mask the loss (or the output and label).
    Args:
        y (torch.Tensor): Tensor of shape (B, C, H, W, D).
        unk_voxels (torch.Tensor): Tensor of shape (B, C, H, W, D) indicating the regions that have tumors not annotated per voxel for each class. I.e., in this tensor, 1 represents voxels we do not know the per-voxel ground-truth. 
        Zero representas voxels we do know the per-voxel ground-truth.
        dilation (int): Size of the cubic structuring element for dilation. Default is 5.
    """
    unk_voxels=unk_voxels.float()
    assert torch.equal(unk_voxels.bool().float(),unk_voxels), 'unk_voxels must be binary'

    if dilation>0:
        #dilate unk voxels: adds a margin around the unknown voxels
        unk_voxels = dilate_volume(unk_voxels, dilation)

    #print("unk_voxels unique values:", torch.unique(unk_voxels), flush=True)
    #print("unk_voxels sum:", unk_voxels.sum(), flush=True)
    one = torch.ones(unk_voxels.shape).type_as(unk_voxels)
    known_voxels = one-unk_voxels
    known_voxels = known_voxels.type_as(y).float()
    assert torch.equal(known_voxels + unk_voxels,one)

    #print('Sum of known voxels:',known_voxels.sum())
    #print('Sum of unknown voxels:',unk_voxels.sum())
    #print('Sum of all voxels:',one.sum(),'matches?',torch.equal(known_voxels + unk_voxels,one))

    if sanity:
        global counter
        if counter<10:
            debug_save_labels(y,str(counter)+'_y',label_names=classes) 
            debug_save_labels(known_voxels,str(counter)+'_known_voxels',label_names=classes)
            debug_save_labels(unk_voxels,str(counter)+'_unk_voxels',label_names=classes)
            print('Saved to '+ str(counter)+'_known_voxels')
            counter+=1



    #print number of channels with unknown voxels
    #num_unknown_channels = unk_voxels.float().sum(dim=(-1,-2,-3))>0
    #num_unknown_channels = num_unknown_channels.float().sum(-1)
    #num_unknown_channels = num_unknown_channels.mean(0)
    #print("---------Number of channels with unknown voxels: ", num_unknown_channels, flush=True, file=sys.stderr)
    #print("Number of known voxels: ", known_voxels.sum(), flush=True, file=sys.stderr)

    #with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
    #    classes = yaml.load(f, Loader=yaml.SafeLoader)

    return known_voxels



def huber_with_tolerance(x: torch.Tensor,
                                 y: torch.Tensor,
                                 tolerance: float,
                                 delta: float = 1.0,
                                 reduction: str = 'none'):
    """
    Huber-with-Tolerance using PyTorch's built-in F.huber_loss. 
    Loss is zero for |x - y| <= tolerance, and standard Huber beyond that.
    Args:
        x (Tensor): Predicted values.
        y (Tensor): Target values.
        tolerance (float): Half-width of the 'dead zone' around y.
        delta (float): Huber 'transition point' between L2 and L1. Default=1.0.
        reduction (str): Same as PyTorch's huber_loss (e.g. 'none', 'mean', 'sum').
    Returns:
        Tensor: The Huber-with-Tolerance loss. Shape depends on `reduction`.
    """
    # 1) Create a zero region inside [y - tolerance, y + tolerance].
    diff = (x - y).abs() - tolerance
    # 2) Clamp negative values to zero => effectively remove small errors
    diff = torch.clamp(diff, min=0.0)
    # 3) Apply PyTorch's Huber to 'diff' vs 0
    #    => HuberLoss( diff, 0 ), with delta controlling the L2-to-L1 transition
    return F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction=reduction)


def plot_huber_with_tolerance(huber_fn=huber_with_tolerance, tolerance=0.1, x_min=0.0, x_max=3, num_points=100):
    """
    Plots the huber_fn loss for x from x_min to x_max against y=1
    with a specified tolerance.
    
    :param huber_fn:   A function huber_fn(x, y, tolerance) -> loss tensor
    :param tolerance:  The tolerance margin around y=1
    :param x_min:      The left boundary of the x range
    :param x_max:      The right boundary of the x range
    :param num_points: How many points to sample in [x_min, x_max]
    """
    import matplotlib.pyplot as plt
    # Prepare x-values
    x_vals = np.linspace(x_min, x_max, num_points)
    y_val  = torch.tensor(1.0)  # y=1, as per your requirement
    
    # Evaluate the loss at each point
    losses = []
    for x in x_vals:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        # Our function expects x, y to have the same shape, so let's reshape if needed
        loss = huber_fn(x_tensor, y_val, tolerance)
        # If the function returns a 1D or 0D tensor, convert to float
        losses.append(loss.item() if loss.dim() == 0 else loss.mean().item())
    
    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(x_vals, losses, label=f'HWT (tolerance={tolerance})')
    plt.title('Huber-with-Tolerance Loss')
    plt.xlabel('x')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)  # losses should be >= 0
    #bound x-axis to be between x_min and x_max
    plt.xlim(x_min, x_max)
    plt.grid(True)
    plt.legend()
    plt.show()
    

import torch
import torch.nn.functional as F
import numpy as np

def l1_with_tolerance(x: torch.Tensor,
                      y: torch.Tensor,
                      tolerance: float,
                      reduction: str = 'none'):
    """
    L1-with-Tolerance:
      1) "Dead zone" of zero loss for |x - y| <= tolerance
      2) Standard L1 (|x - y|) beyond that, shifted by `tolerance`.

    Mathematically:  loss = max(|x - y| - tolerance, 0)

    Args:
        x (Tensor):       Predicted values.
        y (Tensor):       Target values.
        tolerance (float):Half-width of the 'dead zone' around y.
        reduction (str):  'none', 'mean', or 'sum'.

    Returns:
        Tensor: The L1-with-Tolerance loss. Shape depends on `reduction`.
    """
    # 1) Create a zero region inside [y - tolerance, y + tolerance].
    diff = torch.abs(x - y) - tolerance
    # 2) Clamp negative values to zero => effectively remove small errors
    diff = torch.clamp(diff, min=0.0)

    # 3) Apply a reduction
    if reduction == 'none':
        return diff
    elif reduction == 'mean':
        return diff.mean()
    elif reduction == 'sum':
        return diff.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    


def plot_l1_with_tolerance(l1_fn=l1_with_tolerance, tolerance=0.1, x_min=0.0, x_max=3.0, num_points=100):
    """
    Result: this loss made all tumor detections zero. I suspect if has too little penaliziton of the zero solution.
    Plots the L1-with-Tolerance loss for x from x_min to x_max against y=1
    with a specified tolerance.

    :param l1_fn:      The function l1_fn(x, y, tolerance) -> loss tensor
    :param tolerance:  The tolerance margin around y=1
    :param x_min:      The left boundary of the x range
    :param x_max:      The right boundary of the x range
    :param num_points: How many points to sample in [x_min, x_max]
    """
    import matplotlib.pyplot as plt

    # Prepare x-values
    x_vals = np.linspace(x_min, x_max, num_points)
    y_val  = torch.tensor(1.0)  # Fix y=1

    # Evaluate the loss at each point
    losses = []
    for x in x_vals:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        loss = l1_fn(x_tensor, y_val, tolerance)
        # Convert tensor -> float (handles scalar or 1D)
        losses.append(loss.item() if loss.dim() == 0 else loss.mean().item())

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(x_vals, losses, label=f'L1 (tol={tolerance})')
    plt.title('L1-with-Tolerance Loss')
    plt.xlabel('x')
    plt.ylabel('Loss')
    plt.xlim(x_min, x_max)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.show()


def get_lesion_channels(out, classes, assertion = False):
    #merge lesion channels if they are in the same organ. Outputs will have only lesion channels, removes organ channels.
    assert out.shape[1] == len(classes)
    #print('Shapes here: ', out.shape, chosen_segment_mask.shape, flush=True, file=sys.stderr)

    lesion_out = {}
    

    for i,clss in enumerate(classes,0):
        #print('Class is:',clss,'Mask sum is:',chosen_segment_mask[:,i].sum())
        if 'lesion' in clss:
            name = clss[:clss.index('_lesion')+len('_lesion')].replace('pancreatic','pancreas')
            if name not in lesion_out:
                lesion_out[name] = []
            lesion_out[name].append(out[:,i])

    for key in lesion_out.keys():#this combines multi-channel outputs into a single channel
        lesion_out[key] = torch.stack(lesion_out[key],dim=0).max(dim=0).values

    #from dicts to tensor
    kys=list(lesion_out.keys())
    lesion_out = torch.stack([lesion_out[key] for key in kys],dim=1).type_as(out)
    
    if assertion:
        for i in range(lesion_out.shape[0]):
            # For sample i, lo has shape (num_lesion_channels, ...spatial dimensions...)
            lo = lesion_out[i]
            # Sum over all dimensions except the channel, regardless of the number of spatial dims.
            lo_sum = lo.sum(dim=(-1,-2,-3))
            # Create a boolean mask for channels with any nonzero value.
            active_mask = lo_sum > 0
            active_count = active_mask.sum().item()
            if active_count > 1:  # If more than one lesion channel is active
                # Prepare the names of the lesion channels that are active.
                active_names = [kys[j] for j in range(len(kys)) if active_mask[j]]
                raise ValueError(
                    f"Error: For sample index {i}, more than one lesion channel has active elements. "
                    f"Active lesion channels: {active_names}"
                    f"lo.sum(dim=(-1,-2,-3)): {lo.sum(dim=(-1,-2,-3))}"
                )

    return lesion_out

def volume_loss_basic(out,chosen_segment_mask,tumor_volumes, 
                      labels,unk_voxels,
                      classes='/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      dilation=31, tolerance=0.1,loss_function='selective_volume_reduction_huber',n='huber',
                      sigmoid=True, class_weights=None):
    """
    Computes the basic tumor volume loss. This loss compares the total predicted tumor volume inside a subsegment with the tumor volume from the report.
    The loss is based on a relative huber loss with a margin of 0.1.

    Args:
        out (torch.Tensor): The predicted tumor volume.
        chosen_segment_mask (torch.Tensor): The mask indicating the chosen subsegment, the subsegment mask should allign with the lesion channels in the out tensor. 
        For example, if there is a lesion in the pancreas head, the pancreas lesion channel in chosen_segment_mask should show the pancreas head.
        tumor_volumes (torch.Tensor): The tumor volumes from the report.
        classes: list of class names.
        dilation: increases the tumor subsegment margins. This can compensate for errors in the sub-segment segmentation.
    """
    print_loss=True
    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    
    if class_weights is not None:
        assert len(class_weights.shape) == 5
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert class_weights.shape[0] == out.shape[0], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        #repeat class weights to match the batch size of out
        class_weights = class_weights.repeat(1, 1, out.shape[2], out.shape[3], out.shape[4]) # B,C,H,W,D
        
    #activation
    if sigmoid:
        out = torch.sigmoid(out)

    #dilate the chosen segment mask
    chosen_segment_mask = dilate_volume(chosen_segment_mask,dilation)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,dilation)

    #remove from this loss any channel with a tumor that is annotated per-voxel
    per_voxel_positives = (labels.sum((-1,-2,-3),keepdim=True)>0).float()#B,C, which elements we have a tumor annotated per voxel
    #labels = labels * (1-per_voxel_positives)
    out = out * (1-per_voxel_positives)
    
    #voxels we are sure have no tumor:
    negative_voxels = 1 - ((labels + unk_voxels + chosen_segment_mask) > 0).float() #B,C

    #get only the channels with lesions
    out = get_lesion_channels(out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes,assertion=False)
    negative_voxels = get_lesion_channels(negative_voxels, classes)
    labels = get_lesion_channels(labels, classes)
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)
        class_weights = class_weights.mean(dim=(-1,-2,-3)) #reduce to B,C, this will be used to weight the loss for each channel.
    

    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape
    assert out.shape == negative_voxels.shape
    out_in_subsegment = out * chosen_segment_mask
    out_in_negative_voxels = out * negative_voxels

    #we have 1 report volume per batch item, but to what class does it refer to? we can use chosen_segment_mask to figure that out
    report_volume = tumor_volumes.sum(-1) # shape B, we sum the multiple tumors we can have
    report_volume = report_volume.unsqueeze(-1).repeat(1,chosen_segment_mask.shape[1])#B,3
    gate=(chosen_segment_mask.sum(dim=(-1,-2,-3))>0).float()#B,3, one in the lesion channel the report volume refers to, 0 otherwise
    #assert gate.shape[-1]==3
    report_volume = report_volume * gate #B,C, only non-zero for lesion we care about in each CT patch

    if 'huber_entropy' in loss_function:
        loss=huber_entropy(out_in_subsegment,report_volume,tolerance)
        #shape of loss should be B,C
        if class_weights is not None:
            #apply class weights to the loss
            loss = loss * class_weights
        loss = loss.mean() #mean over batch and channels, this should be B,C
        loss={'huber_entropy_loss':loss}
        #print('Using the huber entropy loss')
        return loss
    elif 'dice' in loss_function or 'dce_vol' in loss_function:
        loss=dice_based_volume_loss(out_in_subsegment,report_volume,tolerance=tolerance,E=500,cross_entropy=('entropy' in loss_function))
        #shape of loss should be B,C
        if class_weights is not None:
            #apply class weights to the loss
            loss = loss * class_weights
        loss = loss.mean()
        loss={'dice_volume_loss':loss}
        #print('Using dice volume loss')
        assert not torch.isnan(loss['dice_volume_loss']).any(), 'loss is nan'
        return loss
    else:
        raise ValueError('Deprecated loss function')
        #foreground volume loss: makes the tumor volume inside the subsegment NOT SURPASS the tumor volume from the report, uses sum and divides by report_volume, not a very strong penalization
        if 'l1' in loss_function:
            n='l1'
        elif 'l2' in loss_function:
            n='l2'
        elif 'huber' in loss_function:
            n='huber'
        if 'selective' in loss_function:
            volume_reduction_loss = volume_reduction_loss_selective(out_in_subsegment, report_volume, tolerance, n=n)
        else:
            volume_reduction_loss = ln_with_tolerance_right_side(out_in_subsegment, report_volume, tolerance, n=n)
        if print_loss:
            print('Volume in report:', report_volume, 'volume in subsegment:', out_in_subsegment.sum((-1,-2,-3)), 'volume reduction loss:', volume_reduction_loss.item())
            print()

        
        concentrate=1
        if 'concentrate' in loss_function:
            concentrate=5 #we divide the voxels beyond the top N by 5, creating a harder transition and concentrating more of the loss on the top N voxels    
        #volume_expansion_loss: this uses loss cross entropy to create a strong penalization of false negatives, and it uses GWRP to only boost the TOP N most confident voxels, where N is the number of tumor voxels in the report
        volume_expansion_loss = GWRP_expansion_loss(out_in_subsegment, report_volume, concentrate=concentrate)
        if print_loss:
            print('Volume in report:', report_volume, 'volume in subsegment:', out_in_subsegment.sum((-1,-2,-3)), 'volume expansion loss:', volume_expansion_loss.item())
            print()

        if 'no_bkg' not in loss_function:
            #background_detection loss: penalizes the model for predicting any tumor in the voxels we know have no tumor, uses GWRP and cross entropy for a strong penalizaiton
            background_minimization_loss = GWRP_background_loss(out_in_negative_voxels)
            if print_loss:
                print('Volume in background:', out_in_negative_voxels.sum((-1,-2,-3))[0], 'background minimization loss:', background_minimization_loss.item())
                print()
            print('No background loss')
        else:
            background_minimization_loss = torch.tensor(0).type_as(out)

        loss = volume_reduction_loss + volume_expansion_loss + background_minimization_loss
        loss={'volume_reduction_loss':volume_reduction_loss, 
              'volume_expansion_loss':volume_expansion_loss, 
              'background_minimization_loss':background_minimization_loss}

    if print_loss:
        print(f'Loss is:',loss)

    return loss

def dice_based_volume_loss(x,y,tolerance=0.1,E=500,cross_entropy=False):
    #assert no negative values
    assert torch.min(y).item()>=0
    assert torch.min(x).item()>=0

    #assert no nan
    assert not torch.isnan(x).any(), 'Output is nan'
    assert not torch.isnan(y).any(), 'label is nan'

    #tolerance: return 0 if x/y is within 1+/- tolerance
    if len(x.shape)==5:
        x = x.sum((-1,-2,-3))
    assert len(x.shape)==2, f'shape of x is: {x.shape}'

    predicted_volume = x
    target_volume = y

    assert predicted_volume.shape == target_volume.shape

    loss=torch.abs(predicted_volume-target_volume)/(predicted_volume+target_volume+E)
    #E allows this to work when the ground-truth is zero.

    #subtract the loss at tolerance, for continuity
    v=(1-tolerance)*target_volume
    loss_at_tolerance=torch.abs(v-target_volume)/(v+target_volume+E)

    loss=loss-loss_at_tolerance

    #clamp at zero
    loss=torch.clamp(loss,min=0,max=1)

    if cross_entropy:
        #print('Using cross-entropy')
        loss = -torch.log(torch.ones(loss.shape).type_as(loss)-loss+1e-5)
    else:
        #print('Using dice volume without cross-entropy')
        pass

    return loss

def plot_dice_based_volume_loss(y_value=1000, tolerance=0.1, E=500, num_points=100, x_min=0, x_max=10000,
                                cross_entropy=False):
    """
    Plots the loss for a fixed ground truth volume (y_value) as the predicted volume (x) varies.
    
    y_value : float
        The fixed ground truth volume.
    tolerance : float
        Tolerance percentage (default 0.1 means ±10%).
    E : float
        Offset constant in the denominator.
    num_points : int
        Number of points to sample for predicted volumes.
    x_min, x_max : float
        The range of predicted volumes to consider. If x_max is None, it defaults to 2*y_value.
    """
    import matplotlib.pyplot as plt
    if x_max is None:
        x_max = 2 * y_value  # Default range if not provided

    # Create a series of predicted volume values
    x_values = torch.linspace(x_min, x_max, num_points)
    
    # Create a dummy tensor "x" of shape (num_points, 1, 1, 1)
    # so that summing over the last three dims gives the predicted volume
    x_tensor = x_values.view(num_points, 1, 1, 1)
    
    # Create a target tensor "y" with the same predicted volume for each sample
    y_tensor = torch.full((num_points,), y_value)
    
    # Compute the individual loss values
    loss = dice_based_volume_loss(x_tensor, y_tensor, tolerance=tolerance, E=E, cross_entropy=cross_entropy)
    
    # Plot the loss as a function of the predicted volume
    plt.figure(figsize=(8, 6))
    plt.plot(x_values.numpy(), loss.numpy(), label='Dice-Based Volume Loss')
    plt.xlabel("Predicted Volume (x)")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Predicted Volume for Ground Truth y = {y_value}")
    plt.legend()
    plt.grid(True)
    plt.show()


def ln_with_tolerance_right_side(x: torch.Tensor,
                                 y: torch.Tensor,
                                 tolerance: float,
                                 delta: float = 1.0,
                                 n='huber'):
    """
    Huber-with-Tolerance using PyTorch's built-in F.huber_loss. 
    This loss is one-sided: it only penalizes cases where x > y + tolerance.
    Args:
        x (Tensor): Predicted values.
        y (Tensor): Target values.
        tolerance (float): Half-width of the 'dead zone' around y.
        delta (float): Huber 'transition point' between L2 and L1. Default=1.0.
    Returns:
        Tensor: The Huber-with-Tolerance loss. Shape depends on `reduction`.
    """
    #reduce in case x was not already reduced
    assert len(x.shape)==5 or len(x.shape)==2, 'x must have 5 or 2 dimensions, 2 means we already summed the spatial dimensions'
    assert len(y.shape)==2
    if len(x.shape)==5:
        x = x.sum((-1,-2,-3)) 
    
    #as we will normalize, we cannot have places where y is zero and x is not. This should already have been ensured by gating, since this is only a loss for organ segments with tumors!
    assert torch.allclose(x[y == 0], torch.zeros_like(x[y == 0]))

    #normalize
    x = x / (y + 1e-5)
    y = y / (y + 1e-5)
    diff = x - (y + tolerance)
    diff = torch.clamp(diff, min=0.0)

    if n=='huber':
        loss = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction='none')
    elif n=='l2':
        loss = F.mse_loss(diff, torch.zeros_like(diff), reduction='none')
    elif n=='l1':
        loss = F.l1_loss(diff, torch.zeros_like(diff), reduction='none')
    else:
        raise ValueError('loss not supported')

    assert len(loss.shape)==2, 'loss shape should be B,C'

    loss = loss.mean()

    return loss

def volume_reduction_loss_selective(x,y,tolerance=0.1,n='huber',k=1.5):
    #In this loss, we want to reduce the tumor volume inside its sub-segment, to match the volume in the report.
    #however, we do not want to penalize the N voxels with the highest values. N is the volume of the tumor in the report.
    #the other voxels we will penalize.
    #we use the opposite of GWRP, without averaging, to sum the output voxels (in the subseg) that are not the top N ones.
    #We normalize this sum according to the report volume and we use a huber or l2 loss to make it be lower than the tolerance.
    #the loss is only applied if the volume in the output is greater than the report volume with the tolerance.

    B,C,H,W,D=x.shape

    pooled_x = GlobalWeightedRankPooling(x,N=y,c=0.3,inverse=True) #we ignore the top N voxels
    
    #we want to penalize only channels where the predicted tumor volume is greater than the report volume plus the tolerance
    summed_x = x.sum((-1,-2,-3))
    assert pooled_x.shape==summed_x.shape, 'shapes should be equal'
    assert len(pooled_x.shape)==2 and pooled_x.shape==y.shape, 'x shape should be B,C and match y'
    excessive_volume = (summed_x > (y * (1+tolerance)))#B,C
    #we also do not want to penalize cases where the report volume is zero here!
    pooled_x = pooled_x[excessive_volume & (y>0)]
    if pooled_x.numel() == 0:
        return torch.tensor(0).type_as(x)

    #we want to normalize the loss by the report volume
    y = y[excessive_volume & (y>0)]
    pooled_x = pooled_x / y

    #tolerance: we have a target of tolerance, not zero. Thus, we subtract the tolerance and clamp. Then our target becomes 0
    pooled_x = torch.clamp(pooled_x - tolerance, min=0.0)

    if n=='huber':
        loss = F.huber_loss(pooled_x/k, torch.zeros_like(pooled_x), delta=1.0, reduction='none')
    elif n=='l2':
        loss = F.mse_loss(pooled_x/k, torch.zeros_like(pooled_x), reduction='none')
    elif n=='l1':
        loss = F.l1_loss(pooled_x/k, torch.zeros_like(pooled_x), reduction='none')
    else:
        raise ValueError('loss not supported')
    
    loss = loss.mean()
    
    return loss


def GWRP_expansion_loss(x,y, concentrate=1, eps=1e-5):
    #this loss uses cross entropy and gwrp to enforce the model to detect a tumor of the correct size.

    #this is an expansion loss, we do not want this loss to penalize elements where y==0 then. These elements should have been already removed by gating.
    x_sum=x.sum((-1,-2,-3))
    assert torch.allclose(x_sum[y == 0], torch.zeros_like(x_sum[y == 0]))

    x=GlobalWeightedRankPooling(x,N=y,c=0.75,concentrate=concentrate)
    #we apply gwrp to x, returning a B,C tensor. The spatial dimenions are summed attributing 90% of the summation weight to the top y voxels, and 10% to the rest
    assert len(x.shape)==2, 'x shape should be B,C'

    #now we get only the x items where y is not 0
    x=x[y>0]

    #if empty, return zero
    if x.numel() == 0:
        return torch.tensor(0).type_as(x)

    #now we do cross entropy with target 1
    loss = -torch.log(x+eps)

    loss=loss.mean()

    return loss


def GWRP_background_loss(x, eps=1e-4, decision_th=30):
    #this loss uses cross entropy and gwrp to enforce no false positive tumor detection in the background. 
    #it focuses on the top "decision_th" voxels, strongly penalizing them to become 0 with cross entropy.
    #we use 30 as the decision th, since this is n acceptable decision threshold to convert a segmentation output into a binary classification label.

    #apply gwrp
    x = GlobalWeightedRankPooling(x,N=decision_th,c=0.9)

    #apply cross entropy with target 0
    loss = -torch.log((1-x)+eps)

    assert len(loss.shape)==2, 'loss shape should be B,C'

    loss = loss.mean()

    return loss


def GlobalWeightedRankPooling(x, N=1000, c=0.75, inverse=False, concentrate=1, return_weights=False,hard_cutoff=False):
    """
    Performs Global Weighted Rank Pooling (GWRP). The weights decay exponentially so that
    the top N voxels receive c% of the total weight.
    Ps: the raw weight at voxel N will be 1-c. 
    So, the inverse weight will be c.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        N (int or torch.Tensor): Number of top voxels to concentrate. If an integer, a scalar
                                 value is used; if a tensor of shape (B, C), each (B,C) pair 
                                 uses its own N.
        c (float): Fraction (e.g. 0.9 for 90%) of the total weight to be concentrated in the top N voxels.
    
    Returns:
        torch.Tensor: The pooled tensor of shape (B, C).
    """
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0).unsqueeze(0)
        reduce=True
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    B, C, H, W, D = x.shape
    L = H * W * D  # total number of voxels per (B, C)
    
    # Sort the spatial elements in descending order.
    x_sorted, sort_indices = torch.sort(x.view(B, C, L), dim=-1, descending=True)
    
    # Compute the decay factor d.
    # If N is a scalar, convert it to a tensor of shape (B, C) with that constant.
    if not torch.is_tensor(N):
        N_tensor = torch.full((B, C), N, dtype=torch.float32, device=x.device)
    else:
        N_tensor = N.to(x.device).float()
    # Ensure N is at least 1.
    N_tensor = torch.clamp(N_tensor, min=1)
    
    # Compute d elementwise: d = (1-c)^(1/N).
    d = (1 - c) ** (1.0 / N_tensor)  # shape (B, C)
    # Reshape d to (B, C, 1) so it can broadcast.
    d = d.unsqueeze(-1)
    
    # Create an index tensor of shape (1, 1, L).
    indices = torch.arange(L, dtype=torch.float32, device=x.device).view(1, 1, L)
    
    # Compute weights: each weight is d^(i), broadcasting over (B, C).
    weights_raw = d ** indices  # shape (B, C, L)
    weights = weights_raw / weights_raw.sum(dim=-1, keepdim=True)  # normalize to sum to 1

    #assert that, for a random B,C element, the sum of the first N weights is equal to c
    #rand_b=torch.randint(0,B,(1,))
    #rand_c=torch.randint(0,C,(1,))
    #summed = weights[rand_b, rand_c, :int(N_tensor[rand_b, rand_c].item())].sum()  
    #assert abs(summed.item() - c) < 0.2

    if inverse:
        # For the inverse case we want to ignore the top N voxels.
        # Create a mask that is 0 for indices < N and 1 for indices >= N.
        mask_inv = (indices >= N_tensor.unsqueeze(-1)).float()  # shape (B, C, L)
        # Use the complementary weights for the background: here we use (1 - weights_raw)
        weights = mask_inv * (1 - weights_raw)
        # Note: We do not normalize these weights to sum to 1 because the goal here is to measure
        # the background (i.e. the voxels outside the top N).
    elif concentrate!=1:
        assert concentrate>1, 'concentrate must be greater than 1'
        # Create two masks: one for the top N voxels and one for the rest.
        mask_top = (indices < N_tensor.unsqueeze(-1)).float()      # 1 for indices < N, 0 otherwise
        mask_rest = (indices >= N_tensor.unsqueeze(-1)).float()     # 1 for indices >= N, 0 otherwise
        # Leave top N voxels unchanged and scale the rest by (1/concentrate)
        new_weights = mask_top * weights + mask_rest * (weights / concentrate)
        # Renormalize the weights so they sum to 1.
        weights = new_weights / new_weights.sum(dim=-1, keepdim=True)

    if return_weights:
        if hard_cutoff:
            #make all weights after N zero and re-normalize
            mask_top = (indices < N_tensor.unsqueeze(-1)).float()
            weights = mask_top * weights
            weights = weights / weights.sum(dim=-1, keepdim=True)
        # We need to return the weights reorganized into the original spatial order.
        # sort_indices tells us, for each (B, C, i), which voxel in the unsorted order that value came from.
        # Compute the inverse permutation.
        inverse_indices = sort_indices.argsort(dim=-1)
        # unsort the weights so that they align with the original order.
        weights_unsorted = weights.gather(dim=-1, index=inverse_indices)
        # Reshape to original spatial dimensions.
        weights_unsorted = weights_unsorted.view(B, C, H, W, D)
        if reduce:
            weights_unsorted = weights_unsorted.squeeze(0).squeeze(0)
        return weights_unsorted
    
    # Compute weighted sum and normalize by the sum of weights.
    pooled = (x_sorted * weights).sum(dim=-1)

    return pooled



def DiceLossMultiClass(preds, targets, known_voxels, alpha = 0.5, beta=0.5, size_average=True, reduce=True, sigmoid=True, class_weights=None):

    if len(preds.shape)==3:
        preds=preds.unsqueeze(0).unsqueeze(0)
    if len(targets.shape)==3:
        targets=targets.unsqueeze(0).unsqueeze(0)
    if len(known_voxels.shape)==3:
        known_voxels=known_voxels.unsqueeze(0).unsqueeze(0)

    if len(preds.shape)==4:
        preds=preds.unsqueeze(0)
        targets=targets.unsqueeze(0)
        known_voxels=known_voxels.unsqueeze(0)

    assert len(preds.shape)==5
    assert (preds.shape == targets.shape) and (targets.shape == known_voxels.shape), f"Shapes do not match, pred, target and unk are: {preds.shape}, {targets.shape}, {known_voxels.shape}"

    N = preds.size(0)
    C = preds.size(1)
    
    if sigmoid:
        P = torch.sigmoid(preds)
    else:
        P = preds

    P = P * known_voxels
    targets = targets * known_voxels

    smooth = 1e-5

    class_mask = targets

    ones = torch.ones(P.shape).to(P.device)
    P_ = ones - P 
    class_mask_ = ones - class_mask

    TP = P * class_mask
    FP = P * class_mask_
    FN = P_ * class_mask

    alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
    alpha = alpha.unsqueeze(0).repeat(N, 1) # repeat for each batch item, now alpha is B,C

    alpha = torch.clamp(alpha, min=0.2, max=0.8) 
    #print('alpha:', alpha)
    beta = 1 - alpha
    num = torch.sum(TP, dim=(-1,-2,-3)).float()
    den = num + alpha * torch.sum(FP, dim=(-1,-2,-3)).float() + beta * torch.sum(FN, dim=(-1,-2,-3)).float()

    dice = num / (den + smooth)
    loss = 1 - dice
    if class_weights is not None:
        class_weights = class_weights.mean(dim=(-1,-2,-3))
        while len(class_weights.shape) < len(loss.shape):
            class_weights = class_weights.unsqueeze(0)
        assert class_weights.shape == loss.shape, f'Class weights shape {class_weights.shape} does not match the shape of dice loss {loss.shape}'
        # Apply class weights
        loss = loss * class_weights
    
    if not reduce:
        return loss

    if size_average:
        assert len(loss.shape) == 2, f'Loss should be 2D after reduction, but got {loss.shape}.'
        loss = loss.mean()  # Average over the batch size

    return loss

counter2=100


class MultiTaskLossWrapper(nn.Module):
    """
    Learnable loss weighting for multiple loss components.
    For loss components L_i, we learn parameters s_i and compute:
    
        WeightedLoss = sum_i [ 0.5 * exp(-s_i) * L_i + s_i ]
    """
    def __init__(self, num_losses):
        super(MultiTaskLossWrapper, self).__init__()
        # Initialize log variance parameters to 0 (i.e. initial weight exp(0)=1)
        self.s = nn.Parameter(torch.zeros(num_losses),requires_grad=True)
        
    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss = total_loss + 0.5 * torch.exp(-self.s[i]) * loss + self.s[i]
        #print the loss weights
        print('Loss weights:', [torch.exp(-s).item() for s in self.s])
        return total_loss
    

def classification_loss(cls_out, label, unk_voxels, args, chosen_segment_mask, classes, class_weights=None):
    #calculate classification loss
    
        
    if args.epai_stage_2:
        lesion_idx = [i for i, class_name in enumerate(classes) if (('background' in class_name) or ('pdac' in class_name) or ('pnet' in class_name) or ('cyst' in class_name))]
        lesion_labels = label[:, lesion_idx].float()
        if chosen_segment_mask is not None:
            lesion_labels += chosen_segment_mask[:, lesion_idx].float()
        #print('Lesion labels shape:', lesion_labels.shape)
        #class should be the class of the center voxel
        lesion_labels = lesion_labels[:, :, lesion_labels.shape[2]//2, lesion_labels.shape[3]//2, lesion_labels.shape[4]//2]
        #assert single label
        assert len(lesion_labels.shape)==2, f'Lesion labels shape is: {lesion_labels.shape}'
        assert lesion_labels.sum(dim=1).max()<=1, f'Lesion labels should be single label, but got {lesion_labels.sum(dim=1).max()}'
        #print('Lesion labels:', lesion_labels)
        target_idx = lesion_labels.argmax(dim=1).long() # (B,)
        #print('Lesion label:', target_idx)
        #print('cls_out out:', cls_out.shape)
        #background class? no cyst? lesion? what!?
    else:
        lesion_idx = [i for i, class_name in enumerate(classes) if ('lesion' in class_name)]
        #print(f'Classification loss for classes: {[classes[i] for i in lesion_idx]}')
        lesion_labels = label[:, lesion_idx].float()
        #multi-class
        if chosen_segment_mask is not None:
            lesion_labels += chosen_segment_mask[:, lesion_idx].float()
        lesion_labels = (lesion_labels.sum(dim=(-1,-2,-3))>0).float()
    #now check chosen_segment_mask
    assert len(cls_out.shape)==2 and cls_out.shape[0]==label.shape[0], f'Classification output shape is: {cls_out.shape}, label shape is: {label.shape}'
    if args.epai_stage_2:
        #softmax
        #for i in range(target_idx.shape[0]):
        #    print('Target idx:', target_idx[i], 'cls_out:', cls_out[i],
        #           'class:', classes[target_idx[i]])
        cls_loss   = F.cross_entropy(cls_out, target_idx,reduction='none')
    else:
        #sigmoid
        cls_loss = F.binary_cross_entropy_with_logits(cls_out, lesion_labels, reduction='none', weight=class_weights)
        #print(f'Labels: {lesion_labels[0]}')
        #print(f'cls_out: {cls_out[0]}')
        #print(f'cls_loss: {cls_loss[0]}')
    #if channels with unknown voxels are present and their label is 0, remove them from the loss (multiply by 0)
    if unk_voxels is not None:  
        unk_labels = (unk_voxels[:, lesion_idx].sum(dim=(-1,-2,-3))>0).float()
        #where unk_labels is 1 and label is 0:
        known_labels = (1-unk_labels)+lesion_labels
        known_labels = (known_labels>0).float()
        cls_loss = cls_loss * known_labels
    cls_loss = cls_loss.mean()
    #print('Classification loss:', cls_loss)
    return cls_loss

def calculate_loss(result, label, unk_voxels, args, matcher,chosen_segment_mask,tumor_volumes_report,tumor_diameters,
                   classes,loss_wrapper=None,input_tensor=None, class_weights=None):
    global counter2
    #print('Unk voxels:', unk_voxels)
    
    
    if args.epai_stage_2 and class_weights is not None:
        raise ValueError(
            "Per‑sample/voxel class‑weight matrices are not supported when "
            "epai_stage_2=True (single‑label CrossEntropy). "
            "The standard cross-entropy loss with softmax already favors the positive class"
        )
    
    if args.epai_stage_2:
        result, y_class, y_class_2 = result
    if args.classification_branch and not args.epai_stage_2:
        result, y_class = result
    
    if chosen_segment_mask is not None and chosen_segment_mask.sum()>0:
        if unk_voxels.sum()==0:
            raise ValueError('unk_voxels should not be all zeros if chosen_segment_mask is not all zeros')
    
    #raise ValueError(f'Number of classes in classes: {len(classes)}. Number of classes in label: {label.shape[1]}. Number of classes in result: {result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]}')
    assert len(classes) == label.shape[1], f'Number of classes in classes: {len(classes)} does not match the number of channels in label: {label.shape[1]}'
    assert len(classes) == (result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]), \
    f'Number of classes in result: {(result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1])} does not match the number of channels in label: {label.shape[1]}'
    
    if class_weights is not None and torch.equal(class_weights, torch.ones_like(class_weights)):
        class_weights = None
        
    if class_weights is not None:
        class_weights = class_weights.to(label.device) #make sure class weights are the same size as cls_out
        assert  class_weights.shape[0] == label.shape[0], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert class_weights.shape[1] == label.shape[1], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert len(class_weights.shape) == 2, f'Class weights should be 2D, but got {class_weights.shape}'
        

    if args.classification_branch and not args.epai_stage_2:
        cls_loss = classification_loss(y_class, label, unk_voxels, args, chosen_segment_mask, classes, class_weights)
    if args.epai_stage_2:
        cls_loss_1 = classification_loss(y_class, label, unk_voxels, args, chosen_segment_mask, classes, class_weights)
        cls_loss_2 = classification_loss(y_class_2, label, unk_voxels, args, chosen_segment_mask, classes, class_weights)
        cls_loss = cls_loss_1 + cls_loss_2
        
        
        #print('Average out:', result[-1].float().mean(dim=(-1,-2,-3)))
        #print('Average label:', label.float().mean(dim=(-1,-2,-3)))
        

    loss = 0
    loss_report = 0
    loss_segmentation = 0
    if class_weights is not None:
        class_weights = class_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        assert len(class_weights.shape)==len(label.shape), f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        
    if isinstance(result, tuple) or isinstance(result, list):
        # if use deep supervision, add all loss together---outpput: [final output, hidden output]
        if unk_voxels is not None:
            known_voxels = get_known_voxels(label,unk_voxels,classes=classes)#this will remove (substitute by 0) any channels we are unsure if about the label
            assert torch.equal((known_voxels*label).float().sum(),label.float().sum()), f'The unknown region should not cover channels where our label is different from 0---knwon. we got{(known_voxels*label).float().sum()} and {label.float().sum()}'
            known_voxels_original = known_voxels.clone()
            #print('Assertion successful')
        else:
            known_voxels=torch.ones(label.shape).type_as(label)
        for j in range(len(result)):
            r,l = result[j],label
            if j==0 and args.multi_ch_tumor:
                #hungarian algorithm---run on the final output, use same indices on hidden layer outputs
                out_ids, label_ids = matcher(r,l)
            if args.multi_ch_tumor:
                #shuffle accorind to hungarian algo output
                r=r[out_ids]
                l=l[label_ids]
                known_voxels = known_voxels_original[label_ids]
                unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
                chosen_segment_mask = chosen_segment_mask[label_ids]
                assert r.shape == known_voxels.shape, f'Label mismatch, known voxels is: {known_voxels.shape}, r is: {r.shape}'
                assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
            
            #assert no nan in output
            assert not torch.isnan(r).any(), 'Output is nan'
            if args.report_volume_loss_basic > 0:
                if ('ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss) and not (j!=0 and 'last' in args.loss):
                    #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                    #print('Using the ball loss')
                    loss_r = ball_loss (out=r, labels=l, unk_voxels=unk_voxels, chosen_segment_mask=chosen_segment_mask, 
                                        tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                        apply_dice_loss=('dice' in args.loss), input_tensor=input_tensor,
                                        sigmoid=(not (args.cls_gate and j==0)),
                                        standard_ce=args.stardard_ce_ball, class_weights = class_weights,
                                        single_class= args.epai_stage_2)
                else:
                    loss_r = volume_loss_basic(r, chosen_segment_mask, tumor_volumes_report, l, unk_voxels, classes, loss_function=args.loss,
                                               sigmoid=(not (args.cls_gate and j==0)), class_weights = class_weights)
                    #print('Using the volume loss')
            else:
                loss_r = torch.tensor(0).type_as(r)

            if (not (args.cls_gate and j==0)):
                #print('Using class weights:', class_weights)
                if args.epai_stage_2:
                    #softmax
                    # voxel‑wise single‑label CE (no per‑sample weights)
                    target_idx = l.argmax(dim=1).long()      # (B,H,W,D)
                    loss_seg   = F.cross_entropy(r, target_idx, reduction='none').unsqueeze(1)
                else:
                    loss_seg = F.binary_cross_entropy_with_logits(r, l.float(), reduction='none', weight=class_weights)
                    #print('Using BCE with logits in seg loss')
            else:
                #softmax already applied
                if args.epai_stage_2:
                    raise ValueError('cls_gate not implemented for epai_stage_2')
                #assert l is in the range 0-1
                assert (l>=0).all() and (l<=1).all(), f'Label is not in the range 0-1, its min is: {l.min()}, its max is: {l.max()}'
                assert (r>=0).all() and (r<=1).all(), f'Output is not in the range 0-1, its min is: {r.min()}, its max is: {r.max()}'
                #print('Using class weights:', class_weights)
                loss_seg = F.binary_cross_entropy(r, l.float(), reduction='none', weight=class_weights)

            if not args.epai_stage_2:
                assert loss_seg.shape == known_voxels.shape, f'Loss shape {loss_seg.shape} does not match known voxels shape {known_voxels.shape}'
            else:
                assert len(loss_seg.shape) == len(known_voxels.shape), f'Loss shape {loss_seg.shape} does not match known voxels shape {known_voxels.shape}'
            if counter2<5 and j==0:
                label_names = classes
                if (not (args.cls_gate and j==0)):
                    debug_save_labels(torch.sigmoid(r),str(counter),out_dir='SanityOutputs',label_names=label_names)
                else:
                    debug_save_labels(r,str(counter),out_dir='SanityOutputs',label_names=label_names)
                debug_save_labels(l.float(),str(counter),out_dir='SanityLabelsBeforeLoss',label_names=label_names)
                if args.epai_stage_2:
                    debug_save_labels(loss_seg.repeat(1,4,1,1,1),str(counter),out_dir='SanityLossBCE',label_names=label_names)
                else:
                    debug_save_labels(loss_seg,str(counter),out_dir='SanityLossBCE',label_names=label_names)
                    debug_save_labels(loss_seg*known_voxels,str(counter),out_dir='SanityLossBCEAfterKnownVoxels',label_names=label_names)
                counter2+=1
            loss_seg = loss_seg * known_voxels
            loss_seg = loss_seg.mean() + DiceLossMultiClass(r, l, known_voxels, sigmoid=(not (args.cls_gate and j==0)),class_weights=class_weights)
            loss_segmentation = loss_segmentation + args.aux_weight[j] * args.seg_loss * loss_seg

            if not isinstance(loss_r, dict):
                loss_report = loss_report + args.aux_weight[j] * args.report_volume_loss_basic * loss_r
            else:
                if isinstance(loss_report,int):
                    loss_report = {}
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                else:#dict
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        if key not in list(loss_report.keys()):
                            loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                        else:
                            loss_report[key] = loss_report[key] + args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
    else:
        #raise ValueError('Result is not a tuple or list, you should be using deep supervision')
        if unk_voxels is not None:
            known_voxels = get_known_voxels(label,unk_voxels,classes=classes)#this will remove (substitute by 0) any channels we are unsure if about the label
            assert torch.equal((known_voxels*label).float().sum(),label.float().sum()), 'The unknown region should not cover channels where our label is different from 0---knwon'
        else:
            known_voxels=torch.ones(label.shape).type_as(label)

        if args.multi_ch_tumor:
            out_ids, label_ids = matcher(result,label)
            result=result[out_ids]
            label=label[label_ids]
            assert result.shape == known_voxels.shape
            known_voxels = known_voxels[out_ids]
            unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
            chosen_segment_mask = chosen_segment_mask[label_ids]
            assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
        
        #assert no nan in output
        assert not torch.isnan(result).any(), 'Output is nan'
        if args.report_volume_loss_basic > 0:
            if 'ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss:
                #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                loss_r = ball_loss (out=result, labels=label, unk_voxels=unk_voxels, chosen_segment_mask=chosen_segment_mask, 
                                    tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                    apply_dice_loss=('dice' in args.loss),sigmoid=(not args.cls_gate),
                                    standard_ce=args.stardard_ce_ball,class_weights=class_weights,
                                    single_class= args.epai_stage_2)
            else:
                loss_r = volume_loss_basic(result,chosen_segment_mask,tumor_volumes_report, 
                                           label, unk_voxels, classes, loss_function=args.loss,
                                           sigmoid=(not args.cls_gate), class_weights=class_weights)
        else:
            loss_r = torch.tensor(0).type_as(result)

        if not args.cls_gate:
            if args.epai_stage_2:
                #softmax
                target_idx = label.argmax(dim=1).long()
                loss_seg   = F.cross_entropy(result, target_idx, reduction='none') #use BCE with logits for the segmentation loss
            else:
                loss_seg = F.binary_cross_entropy_with_logits(result, label.float(), reduction='none', weight=class_weights) #use BCE with logits for the segmentation loss
        else:
            if args.epai_stage_2:
                raise ValueError('cls_gate not implemented for epai_stage_2')
            #assert l is in the range 0-1
            assert (label>=0).all() and (label<=1).all(), f'Label is not in the range 0-1, its min is: {label.min()}, its max is: {label.max()}'
            assert (result>=0).all() and (result<=1).all(), f'Output is not in the range 0-1, its min is: {result.min()}, its max is: {result.max()}'
            loss_seg = F.binary_cross_entropy(result, label.float(), reduction='none', weight=class_weights) #use BCE for the segmentation loss when cls_gate is used, this will be 0/1 for the binary classification case

        assert loss_seg.shape == known_voxels.shape

        loss_seg = loss_seg * known_voxels
        loss_seg = loss_seg.mean() + DiceLossMultiClass(result, label, known_voxels, sigmoid=(not args.cls_gate),class_weights=class_weights)
        loss_segmentation = loss_segmentation + args.seg_loss * loss_seg
        if not isinstance(loss_r, dict):
            loss_report = loss_report + args.report_volume_loss_basic * loss_r
        else:
            if isinstance(loss_report,int):
                loss_report = {}
            for key in loss_r.keys():
                if key == 'ball_loss_bce':
                    weight = args.ball_bce_weight
                    #print(f'Using the ball bce weight: {weight}')
                elif key == 'ball_loss_dice':
                    weight = args.ball_dice_weight
                    #print(f'Using the ball dice weight: {weight}')
                else:
                    weight = 1
                loss_report[key] = args.report_volume_loss_basic * weight * loss_r[key]
                
    loss={'segmentation':loss_segmentation}
    if isinstance(loss_report,dict):
        for key in loss_report.keys():
            loss[key] = loss_report[key]
    else:
        loss['report'] = loss_report

    #print('Loss report is:', loss_report)
    
    if args.classification_branch:
        loss['classification'] = cls_loss

    if loss_wrapper is None:
        loss_overall = 0
        for key in loss.keys():
            #print('loss key:', key)
            loss_overall = loss_overall + loss[key]
    else:
        #create a list of losses from the dict
        losses = [loss[key] for key in sorted(loss.keys())]
        print('Losses sent to the wrapper:',  sorted(loss.keys()))
        loss_overall = loss_wrapper(losses)
        #assert weihts are 1
        #assert args.seg_loss==1 and args.report_volume_loss_basic==1, 'We should not be weighting the losses, as this would mean we are not using the wrapper'
        #check if wrapper requires grad
        for l in loss_wrapper.parameters():
            assert l.requires_grad, 'Loss wrapper parameters should require grad'

    loss['overall']=loss_overall

    #check if loss_overall requires grad
    assert loss_overall.requires_grad, 'Loss overall should require grad'

    return loss

def debug_save_labels(labels: torch.Tensor,
                      name='',
                      label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      out_dir: str = "./LossChecking",
                      batch_idx = 0):
    """
    Saves each channel of the specified batch index in `labels` as a .nii.gz file.
    
    Args:
        labels (torch.Tensor): A tensor of shape (B, C, H, W, D).
        label_names_yaml (str): Path to a YAML file containing a list of label names.
                                The list will be sorted alphabetically and used
                                to name the channels.
        out_dir (str): Output directory to save the .nii.gz files. Defaults to "LossSanity".
        batch_idx (int): Which batch element to save. Defaults to 0.
    """
    import nibabel as nib
    # 1. Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    #raise ValueError(f'Label names is: {label_names}')
    
    # 2. Load and sort label names
    if not isinstance(label_names, list):
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)  # e.g. ["liver", "kidney", "pancreas", ...]
        
    label_names_sorted = sorted(label_names)  # sort alphabetically
    
    # 3. Basic shape check
    if len(labels.shape)==4:
        labels = labels.unsqueeze(0)

    if labels.shape[1]!=len(label_names_sorted):
        raise ValueError(f"Number of channels in labels ({labels.shape[1]}) does not match the number of label names ({len(label_names_sorted)}). Labels loaded from: {label_names}. ")
        label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/list/label_names.yaml'
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)
        label_names_sorted = sorted(label_names)
    
    assert len(labels.shape) == 5
    B, C, H, W, D = labels.shape
    assert batch_idx < B, f"batch_idx={batch_idx} is out of range for B={B}."
    if C != len(label_names_sorted):
        label_names_sorted = [str(i) for i in list(range(C))]
    
    # 4. Extract just the batch element we want
    #    This will have shape (C, H, W, D).
    label_slice = labels[batch_idx]
    
    # 5. Loop over channels, save each one as a nii.gz
    for c in range(C):
        # Move channel c to CPU numpy for saving
        channel_data = label_slice[c].detach().cpu().numpy()
        
        # Build a simple identity affine; if you have real metadata, replace it
        affine = np.eye(4, dtype=np.float32)
        
        # Convert to float32 (or int16, float64, etc.)
        channel_data = channel_data.astype(np.float32)
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(channel_data, affine)
        
        # Derive a filename from the label name
        channel_label_name = label_names_sorted[c]
        out_path = os.path.join(out_dir, f"{name}_{channel_label_name}.nii.gz")

        #print(f'Saving: {out_path}, its sum is {channel_data.sum()}')
        
        # Save
        nib.save(nifti_img, out_path)
        
    print(f"Saved to {out_path}")


def huber_entropy(x: torch.Tensor,
                  y: torch.Tensor,
                  tolerance: float,
                  k: float = 1.5,
                  reduction: str = 'mean',
                  eps: float = 1e-3):
    """
    Result: better than L1, in terms of avoiding false negatives. However, this loss caused too many FP! We need more penalization of FP cases.
    Compute a piecewise loss (named huber_entropy) where the target is normalized to 1.
    
    Steps:
      1. Normalize: x_norm = x / (y + 1e-4). For y > 0, the ideal value is 1.
      2. If x_norm < (1 - tolerance):  
             Use a BCE-like loss with a soft target of (1 - tolerance):
             L = -[ (1-tolerance)*log(x_norm + eps) + (tolerance)*log(1 - x_norm + eps) ]
      3. If x_norm > (1 + tolerance):  
             Compute z = (x_norm - (1+tolerance)) / k, and then apply a scaled Huber loss on z with delta=1:
             L = { 0.5 * z^2,    if |z| < 1  
                 { |z| - 0.5,    otherwise.
      4. Otherwise, if x_norm is in [1-tolerance, 1+tolerance]: loss = 0.
      
    Args:
      x (torch.Tensor): Predicted values.
      y (torch.Tensor): Target values.
      tolerance (float): Tolerance defining the deadzone around 1.
      k (float): Scaling factor for the Huber branch (default 1). The higher, the less agressive the loss is for tumors predicted to be larger than they actually are.
      reduction (str): Reduction mode: 'none', 'mean', or 'sum'.
      eps (float): Small constant to avoid log(0). This controls the strength of your gradient at volume=0. This grad will be -1/eps
    
    Returns:
      torch.Tensor: The reduced loss.
    """
    # Normalize so that target becomes 1
    if len(x.shape) == 5:
        x = x.sum(dim=(-3,-2,-1))
    assert len(x.shape) == 2, f"x.shape={x.shape} is not supported"
    
    x_norm = x / (y + 1e-4)
    
    # Initialize loss tensor with zeros
    loss = torch.zeros_like(x_norm)
    
    # Region 1: x_norm < (1 - tolerance): BCE-like loss with soft target (1-tolerance)
    mask_low = x_norm < (1 - tolerance)
    if mask_low.any():
        ce_loss = - (torch.log(x_norm[mask_low]/(1 - tolerance) + eps))
        loss[mask_low] = ce_loss

    # Region 2: x_norm > (1 + tolerance): Scaled Huber loss on z = (x_norm - (1+tolerance))/k
    mask_high = x_norm > (1 + tolerance)
    if mask_high.any():
        z = (x_norm[mask_high] - (1 + tolerance)) / k
        abs_z = torch.abs(z)
        huber_loss = torch.where(abs_z < 1, 0.5 * z**2, abs_z - 0.5)
        loss[mask_high] = huber_loss

    # Region 3: In-between: loss remains 0.
    
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction option: {reduction}")

def ln_entropy(x,y,tolerance=0.1,
               k=1.5,reduction = 'none',
               eps = 1e-3, n=2):
    """
    Compute a piecewise loss (named l2_entropy) where the target is normalized to 1.
    
    Steps:
      1. Normalize: x_norm = x / (y + 1e-4). For y > 0, the ideal value is 1.
      2. If x_norm < (1 - tolerance):  
             Use a BCE-like loss with a soft target of (1 - tolerance):
             L = -[ (1-tolerance)*log(x_norm + eps) + (tolerance)*log(1 - x_norm + eps) ]
      3. If x_norm > (1 + tolerance):  
             Compute z = (x_norm - (1+tolerance)) / k, and then apply a scaled L2 loss on z
      4. Otherwise, if x_norm is in [1-tolerance, 1+tolerance]: loss = 0.
      
    Args:
      x (torch.Tensor): Predicted values.
      y (torch.Tensor): Target values.
      tolerance (float): Tolerance defining the deadzone around 1.
      k (float): Scaling factor for the Huber branch (default 1). The higher, the less agressive the loss is for tumors predicted to be larger than they actually are.
      reduction (str): Reduction mode: 'none', 'mean', or 'sum'.
      eps (float): Small constant to avoid log(0). This controls the strength of your gradient at volume=0. This grad will be -1/eps
      negative_norm: float: Value to avoid division by zero when normalizing y. Default is 1000. The lower this number, the more we penalize false positives.
    
    Returns:
      torch.Tensor: The reduced loss.
    """
    # Normalize so that target becomes 1
    denom = torch.where(y > 0, y, torch.ones_like(y)*negative_norm)
    x_norm = x / denom
    y_norm = torch.where(y > 0, y/y, torch.zeros_like(y))
    #check for nan
    assert not torch.isnan(x_norm).any(), 'x_norm is nan'
    assert not torch.isnan(y_norm).any(), 'y_norm is nan'
    #normalization: we avoid dividion by zero. If y is zero, we divide by negative_norm. negative_norm=1000 should be a common normalization value for positives. 100 should be pretty strong penalization of FP. 
    
    # Initialize loss tensor with zeros
    loss = torch.zeros_like(x_norm)
    
    # Region 1: x_norm < (1 - tolerance): BCE-like loss with soft target (1-tolerance)
    #negatives: y_norm will be 0, x_norm cannot be negative, so we go to region 2. 
    mask_low = x_norm < (1 - tolerance)*y_norm
    if mask_low.any():
        ce_loss = - (torch.log(x_norm[mask_low]/((1 - tolerance)*y_norm[mask_low]) + eps))
        loss[mask_low] = ce_loss

    #check for nan
    assert not torch.isnan(loss).any(), 'loss is nan in region 1'

    # Region 2: x_norm > (1 + tolerance): Scaled l2 loss on z = (x_norm - (1+tolerance))/k
    # negatives: tolerance does not matter, we multiply by y_norm, which will be 0 for negatives. Target becomes 0.
    mask_high = x_norm > (1 + tolerance)*y_norm
    if mask_high.any():
        z = (x_norm[mask_high] - ((1 + tolerance)*y_norm[mask_high])) / k
        if isinstance(n, int):
            loss[mask_high] = torch.abs(z)**n
        elif n=='huber':
            loss[mask_high] = F.huber_loss(torch.abs(z), torch.zeros_like(z), reduction='none')
        else:
            raise ValueError('n should be int or "huber"')

    # what if both x_norm and y_norm are zero? both conditions above will be false (0>0 and 0<0)
    mask_zero = (x_norm==0)&(y_norm==0)
    if mask_zero.any():
        loss[mask_zero] = torch.zeros_like(y_norm[mask_zero])
    
    #check for nan
    assert not torch.isnan(loss).any(), 'loss is nan in region 2'

    # Region 3: In-between: loss remains 0.
    
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction option: {reduction}")

def isnet_entropy(x,y,tolerance=0.15,
               reduction = 'none',
               eps = 1e-4, E=1):
    """
    Compute a piecewise loss (named isnet_entropy) where the target is normalized to 1.
    
    Steps:
      1. Normalize: x_norm = x / (y + 1e-4). For y > 0, the ideal value is 1.
      2. If x_norm < (1 - tolerance):  
             Use a BCE-like loss with a soft target of (1 - tolerance):
             L = -[ (1-tolerance)*log(x_norm + eps) + (tolerance)*log(1 - x_norm + eps) ]
      3. If x_norm > (1 + tolerance):  
             Uses a loss based on the isnet background loss, where we subtract the predicted tumor volume from 1+tolerance, then apply a saturating non-linearity (x/(x+1)), and then we apply a cross entropy loss with target 0.
      4. Otherwise, if x_norm is in [1-tolerance, 1+tolerance]: loss = 0.
      
    Args:
      x (torch.Tensor): Predicted values.
      y (torch.Tensor): Target values.
      tolerance (float): Tolerance defining the deadzone around 1.
      k (float): Scaling factor for the Huber branch (default 1). The higher, the less agressive the loss is for tumors predicted to be larger than they actually are.
      reduction (str): Reduction mode: 'none', 'mean', or 'sum'.
      eps (float): Small constant to avoid log(0). This controls the strength of your gradient at volume=0. This grad will be -1/eps
    
    Returns:
      torch.Tensor: The reduced loss.
    """
    # Normalize so that target becomes 1
    x_norm = x / (y + 1e-4)
    
    # Initialize loss tensor with zeros
    loss = torch.zeros_like(x_norm)
    
    # Region 1: x_norm < (1 - tolerance): BCE-like loss with soft target (1-tolerance)
    mask_low = x_norm < (1 - tolerance)
    if mask_low.any():
        ce_loss = - (torch.log(x_norm[mask_low]/(1 - tolerance) + eps))
        loss[mask_low] = ce_loss

    # Region 2: x_norm > (1 + tolerance): subtract tartget, activate with x/(x+1) saturating function, follow by cross entropy with target 0 
    mask_high = x_norm > (1 + tolerance)
    if mask_high.any():
        #subtract target, 1+tolerance
        z = (x_norm[mask_high] - (1 + E))
        #activate with x/(x+1) saturating function
        z = z/(z+1)
        #cross entropy with target 0
        loss[mask_high]=-torch.log(torch.ones(z.shape).type_as(z) - z + eps)

    # Region 3: In-between: loss remains 0.
    
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction option: {reduction}")
    



def plot_loss(ln: str = "huber", 
              y = 1,
              tolerance: float = 0.1, 
              k: float = 1.5, 
              num_points: int = 300, 
              limit: float = 3, 
              eps: float = 1e-2):
    """
    Plots a piecewise loss and the absolute value of its gradient (in log scale) as a function 
    of the predicted value x (with target y fixed to 1). The loss plotted is selected by the 
    'loss_type' parameter.
    
    Args:
        loss_type (str): Which loss to plot. Options are:
                         "huber"   -> huber_entropy,
                         "l2"      -> l2_entropy,
                         "isnet"   -> isnet_entropy.
        tolerance (float): The tolerance defining the deadzone.
        k (float): The scaling factor for the huber or l2 branch. (Ignored for "isnet".)
        num_points (int): Number of x points to sample.
        limit (float): Maximum value of x to sample.
        eps (float): Small constant for numerical stability.
    """
    import matplotlib.pyplot as plt

    # Create x_vals with requires_grad=True so that we can compute gradients.
    x_vals = torch.linspace(0, limit, num_points, requires_grad=True)


    # Set target y = 1 so that normalized target is 1.
    y_val = torch.ones_like(x_vals)*y
    
    
    # Choose the loss function based on loss_type:
    loss_label = f"Loss Type: {ln}"
    losses = ln_entropy(x_vals, y_val, tolerance=tolerance, k=k, n=ln, reduction='none', eps=eps)
    
    
    # Compute gradient of loss with respect to x_vals.
    gradients = torch.autograd.grad(outputs=losses, inputs=x_vals,
                                    grad_outputs=torch.ones_like(losses))[0]
    
    # Convert tensors to numpy arrays for plotting.
    x_np = x_vals.detach().numpy()
    losses_np = losses.detach().numpy()
    gradients_np = gradients.detach().numpy()
    
    # Create a two-panel plot.
    plt.figure(figsize=(12, 5))
    
    # Plot the loss.
    plt.subplot(1, 2, 1)
    plt.plot(x_np, losses_np, label=loss_label)
    plt.xlabel("Predicted value x")
    plt.ylabel("Loss")
    plt.title(f"{loss_label} (Target = 1)")
    plt.axvline(1-tolerance, color='r', linestyle='--', label=f"x = 1-tol")
    plt.legend()
    plt.grid(True)
    
    # Plot the absolute gradient on a log scale.
    plt.subplot(1, 2, 2)
    plt.plot(x_np, abs(gradients_np), label="|Gradient|")
    plt.xlabel("Predicted value x")
    plt.ylabel("Absolute Gradient")
    plt.title("Absolute Gradient of Loss")
    plt.yscale("log")
    plt.axvline(1-tolerance, color='r', linestyle='--', label=f"x = 1-tol")
    plt.legend()
    plt.grid(True, which="both")
    
    plt.tight_layout()
    plt.show()



def ISNetLikeNegativeLoss(x,d=0.9977,E=1, reduction='none'):
    assert len(x.shape)==5, f'Input to ISNetLikeNegativeLoss should be 5D, got {x.shape}'
    #global maxpool on spatial dimensions:
    x=GlobalWeightedRankPooling(x,d=d)
    #activation:
    x=x/(x+E)
    #cross entropy (pixel-wise):
    x=torch.clamp(x,max=1-1e-7)
    loss=-torch.log(torch.ones(x.shape).type_as(x)-x)
    #reduction:
    if reduction=='none':
        return loss
    elif reduction=='mean':
        return loss.mean()
    elif reduction=='sum':
        return loss.sum()
    else:
        raise ValueError('Reduction not supported')
    











############### BALL LOSS ####################

def create_ball_kernel(diameter, gaussian=False, gaussian_std=1.5):
    """
    Creates a 3D torch tensor (kernel) where there is a 'ball' of a given diameter.
    The diameter is first rounded up to the next odd integer. The kernel size is then
    computed to be 1.2 × (that odd diameter), rounded to the next odd integer.
    
    The ball is centered in this larger kernel. Inside the ball (hard cutoff at the
    ball boundary), values are set to 1 (or to a truncated Gaussian if `gaussian=True`).
    Outside the ball, values are 0. If `gaussian=True`, the Gaussian is centered at
    the ball center with standard deviation `gaussian_std * radius`.

    Parameters
    ----------
    diameter : float or int
        Desired diameter of the ball. Will be rounded up to the next odd integer.
    gaussian : bool, optional
        Whether to fill the ball with a Gaussian distribution, by default False.
    gaussian_std : float, optional
        Standard deviation factor (relative to the ball radius) if gaussian=True.
        For example, if the ball's radius is R and gaussian_std=1.5, the std is
        1.5*R, by default 1.5.

    Returns
    -------
    kernel : torch.FloatTensor
        A 3D tensor of shape (kernel_size, kernel_size, kernel_size) containing
        the ball (or Gaussian ball) centered in the kernel.
    """

    # --- Step 1: Round diameter to next odd integer ---
    diameter_ceil = math.ceil(diameter)
    if diameter_ceil % 2 == 0:
        diameter_ceil += 1
    diameter_odd = diameter_ceil  # The final odd diameter
    
    # --- Step 2: Compute kernel size as 1.2 * diameter_odd, also round up to next odd ---
    kernel_size_float = 1.2 * diameter_odd
    kernel_size_ceil = math.ceil(kernel_size_float)
    if kernel_size_ceil % 2 == 0:
        kernel_size_ceil += 1
    kernel_size = kernel_size_ceil  # The final odd kernel size
    
    # Ball radius (float)
    radius = diameter_odd / 2.0

    # --- Create 1D coordinate grid from 0..(kernel_size-1), shift so center is 0 ---
    center = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords_shifted = coords - center  # center at 0
    
    # --- Compute squared distance (3D) via broadcasting ---
    distance_squared = (coords_shifted[:, None, None] ** 2
                      + coords_shifted[None, :, None] ** 2
                      + coords_shifted[None, None, :] ** 2)
    
    # --- Hard cutoff mask for the ball ---
    mask = (distance_squared <= radius**2).float()
    
    if gaussian:
        # Scale std by the ball's actual radius
        std = gaussian_std * radius
        gaussian_values = torch.exp(-distance_squared / (2.0 * std**2))
        kernel = gaussian_values * mask
        # Normalize so that sum of kernel = 1
        kernel = kernel / kernel.sum()
    else:
        kernel = mask  # Binary ball kernel

    #assert the kernel size is odd
    assert kernel.shape[0] % 2 == 1, f'Kernel size should be odd, got {kernel.shape[0]}'
    
    return kernel


def save_ball_kernel(diameter, gaussian, gaussian_std, filename):
    """
    Wrapper function that creates a ball kernel using `create_ball_kernel`,
    prints the center and border values, and saves the kernel as a .nii.gz file.
    
    Args:
        diameter (int): Diameter of the ball.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball.
        gaussian_std (float): Standard deviation of the Gaussian.
        filename (str): Path for saving the NIfTI file (should end with .nii.gz).
    """
    # Create the kernel
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std)
    
    # Determine the center index (assuming symmetric kernel)
    center_idx = diameter // 2
    center_value = kernel[center_idx, center_idx, center_idx].item()
    
    # Determine the border value as the smallest nonzero value inside the ball.
    # (This should correspond roughly to the values at the edge.)
    border_value = kernel[kernel > 0].min().item()
    
    print(f"Center value: {center_value}")
    print(f"Border value: {border_value}")
    
    # Convert to numpy array (nibabel works with numpy)
    kernel_np = kernel.numpy()
    
    # Create a default affine (identity) matrix
    affine = np.eye(4)
    
    # Create and save the NIfTI image
    nii_img = nib.Nifti1Image(kernel_np, affine)
    nib.save(nii_img, filename)
    print(f"Saved ball kernel to {filename}")

def ball_convolution(x,diameter,gaussian, gaussian_std):
    """
    Performs a 3D convolution on the input tensor `x` using a ball kernel of diameter `diameter`.
    Optionally, the values inside the ball can follow a Gaussian distribution with standard deviation `gaussian_std`.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball.
        gaussian_std (float): Standard deviation of the Gaussian.
    
    Returns:
        torch.Tensor: Convolved tensor of shape (B, C, H, W, D).
    """
    #if diameter is not odd, add 1:
    if diameter%2==0:
        diameter+=1

    # Create the ball kernel
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    
    # Convert kernel to 5D tensor (B=1, C=1, H, W, D)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Perform the 3D convolution
    out = F.conv3d(x, kernel, padding=kernel.shape[-1]//2)

    assert out.shape == x.shape, f'Output shape should be the same as input shape, got {out.shape} and {x.shape}'
    return out

def insert_ball_old(out_spatial,best_center,diameter,margin):
    # Use a binary (non-Gaussian) ball kernel.
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)
    #we add the margin only here, we do not use the margin in the convolution, for better detection.
    
    # Create an empty volume for the ball mask with the same spatial shape as x.
    masked_volume = torch.zeros_like(out_spatial)
    H, W, D = masked_volume.shape
    d_half = binary_ball_kernel.shape[-1] // 2
    cx, cy, cz = best_center

    # For each dimension, compute the overlapping indices between the input volume and the ball kernel.
    # X-dimension:
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(H, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # Y-dimension:
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(W, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # Z-dimension:
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(D, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # Place the binary ball kernel into the masked_volume at the computed overlapping region.
    masked_volume[vol_x_min:vol_x_max, vol_y_min:vol_y_max, vol_z_min:vol_z_max] = \
        binary_ball_kernel[mask_x_min:mask_x_max, mask_y_min:mask_y_max, mask_z_min:mask_z_max]
    return masked_volume

def insert_ball(out_spatial, best_center, diameter, margin):
    """
    Places a 'ball' of size diameter * (1 + margin) into out_spatial at the 3D coordinate best_center.
    The 3D ordering is assumed to be (z, y, x).
    """
    # 1) Build the ball kernel for insertion
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)

    # 2) Prepare an empty volume with same shape as out_spatial
    masked_volume = torch.zeros_like(out_spatial)
    
    # 3) Extract shape in (z, y, x) order
    Z, Y, X = masked_volume.shape
    
    # 4) The kernel half-width
    d_half = binary_ball_kernel.shape[-1] // 2
    
    # 5) Unpack best_center as (cz, cy, cx)
    cz, cy, cx = best_center
    
    # 6) Compute overlap in Z dimension
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(Z, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # 7) Compute overlap in Y dimension
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(Y, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # 8) Compute overlap in X dimension
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(X, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # 9) Place the kernel region into masked_volume
    masked_volume[
        vol_z_min:vol_z_max,
        vol_y_min:vol_y_max,
        vol_x_min:vol_x_max
    ] = binary_ball_kernel[
        mask_z_min:mask_z_max,
        mask_y_min:mask_y_max,
        mask_x_min:mask_x_max
    ]

    return masked_volume

def isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume, margin=0.3):
    """
    Uses a ball convolution over x and applies a maximum operation to find the best
    fitting ball center. Then, it multiplies the input by a volume with the same size
    as the input, but with a binary ball placed at the given object center coordinate.
    Finally, after the multiplication, we find the top N voxels inside the remaining volume.
    N is the tumor volume.

    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball (for convolution).
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): Number of voxels to select as the tumor volume.
    
    Returns:
        torch.Tensor: A binary tumor mask of shape (H, W, D) with 1's in the top N voxels.
    """
    reduce=False
    if len(x.shape)==3:
        reduce=True
        x = x.unsqueeze(0).unsqueeze(0)
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    #round diameter
    diameter = np.round(diameter).astype(int)
    #round tumor volume
    tumor_volume = np.round(tumor_volume).astype(int)

    # Ensure the diameter is odd.
    if diameter % 2 == 0:
        diameter += 1

    # Create the ball kernel for convolution.
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    # Convert kernel to a 5D tensor (shape: 1, 1, H, W, D).
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    #assert volume is not larger than the number of voxels in the ball
    if tumor_volume > 100000:
        assert tumor_volume <= (kernel>0).sum()*1.2, f'Tumor volume should be smaller than the number of voxels in the ball, got {tumor_volume} and {(kernel>0).sum()}'

    if (kernel>0).sum() > tumor_volume:
        #we tolerate numerical erros within a margin of 0.2
        tumor_volume = (kernel>0).sum()-1

    
    # Perform 3D convolution.
    out = F.conv3d(x, kernel, padding=kernel.shape[-1] // 2)

    assert out.shape == x.shape, f"Output shape should match input shape, got {out.shape} vs {x.shape}"

    # --- Step 1: Find the best fitting ball center ---
    # Assume x is of shape (1, 1, H, W, D); take the spatial part.
    out_spatial = out[0, 0]  # shape: (H, W, D)
    max_idx = torch.argmax(out_spatial)
    best_center = np.unravel_index(max_idx.item(), out_spatial.shape)  # (cx, cy, cz)
    
    # --- Step 2: Create a binary ball mask at the best center ---
    masked_volume = insert_ball(out_spatial,best_center,diameter,margin)
    new_dim = diameter
    while masked_volume.sum() < tumor_volume:
        #if the ball is in the border of the image, its volume may be less than the tumor volume, We increase the size of the ball until we reach the tumor volume.
        old_dim = new_dim
        new_dim = int(np.round(new_dim * 1.1))
        print(f'Increasing ball size to {new_dim}, current volume is {masked_volume.sum()}, tumor volume is {tumor_volume}')
        if old_dim == new_dim:
            new_dim += 1
        if new_dim % 2 == 0:
            new_dim += 1
        if new_dim >= max(x.shape[-1], x.shape[-2], x.shape[-3]):
            break
        masked_volume = insert_ball(out_spatial,best_center,new_dim,margin)
    if tumor_volume < (50**3):
        assert (masked_volume.sum() > (diameter**3)*0.5), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume}'
    if tumor_volume > (6**3):
        assert (masked_volume.sum() < (diameter**3)*2), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume}'

    # --- Step 3: Multiply the input by the binary ball mask ---
    # x has shape (B, C, H, W, D); expand masked_volume to match.
    #assert no negative value in x
    assert (x >= 0).all(), f'Input tensor should not have negative values, got {x.min()}'
    masked_x = (x * masked_volume.unsqueeze(0).unsqueeze(0))

    # --- Step 4: Find the top N voxels in the masked volume ---
    # Remove batch and channel dimensions.
    masked_x_vol = masked_x[0, 0]
    flattened = masked_x_vol.view(-1)
    # Get indices of the top N voxel values.
    t=min(flattened.shape[-1]-1, tumor_volume)
    topN_values, topN_indices = torch.topk(flattened, t)
    #how many indices? Assert this matches the tumor volume
    assert len(topN_indices) == t, f'Expected {tumor_volume} indices, got {len(topN_indices)}'
    # Create a binary volume: set top N positions to 1, rest to 0.
    tumor_mask_flat = torch.zeros_like(flattened)
    tumor_mask_flat[topN_indices] = 1
    # Reshape to original spatial dimensions.
    tumor_mask = tumor_mask_flat.view_as(masked_x_vol)
    # Assert the sum here still matches the tumor volume.
    assert tumor_mask.sum() == t, f'Tumor mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    
    #ensure no tumor_max value is outside the ball
    tumor_mask = tumor_mask * masked_volume

    if reduce:
        tumor_mask = tumor_mask.squeeze(0).squeeze(0)

    iters = 0
    while tumor_volume < (50**3) and tumor_mask.sum() < tumor_volume*0.7:
        #zero values inside the ball may not be chosen as the top N voxels. In such cases, we dilate the mask
        print(f'dilating tumor mask, iteration {iters}, current volume is {tumor_mask.sum()}, tumor volume is {tumor_volume}')
        if iters >5:
            return tumor_mask
        #dilate the mask
        tumor_mask = dilate_volume(tumor_mask, 7)*masked_volume

    if tumor_volume < (50**3):
        assert (tumor_mask.sum() > tumor_volume*0.5), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'
    if tumor_volume > (5**3):
        assert (tumor_mask.sum() < tumor_volume*3), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    #assert it is binary
    assert (tumor_mask == 0).sum() + (tumor_mask == 1).sum() == tumor_mask.numel(), f'Tumor mask should be binary, got {tumor_mask.sum()}'

    return tumor_mask


counter3=100

def ball_loss(out, labels, unk_voxels, chosen_segment_mask, tumor_volumes, tumor_diameters, classes, apply_dice_loss,
              diameter_margin=0.03, gaussian=True, 
              gaussian_std=1.5, gwrp=True, gwrp_concentration=0.5, dilation_for_background=7,
              subseg_dilation=31,input_tensor=None, unk_dilation=1,
              sigmoid=True, standard_ce=False, class_weights=None,
              single_class=False):
    """
    This funciton first uses a ball loss to isolate the tumor. Then, it selects the top N voxels inside the ball as a "pseudo-label" and applies BCE loss per-voxel.
    Optionally, we can average the per-voxel BCE loss using GWRP weights calculated for the isolated tumor voxels. This will give more emphasis in increasing high confidence voxels.
    Args:
    x is the model output
    tumor_diameter is a tensor of size B,T,3, batch, number of tumors in the crop, and 3 diameters
    diameter_margin: how much much we want the ball diameter to be bigger than the maximum tumor diameter
    gaussian: if a gaussian kernel is used in the ball convolution for better centering on the tumor
    gaussian_std: the higher, the smaller the difference between the ball kernel center and border values
    gwrp: wether to use GRWP to average each BCE loss. If so, more weight is given to increasing high confidence voxels.
    sigmoid: wether to apply sigmoid to the output.
    dilation_for_background: we apply a dilation kernel of this size to the tumor pseudo-mask, and define everything outside this mask as background, and use BCE loss to make the backgropund 0
    subseg_dilation: how much we dilate the tumor subsegment. Indeed, we use a very high volume. Radiologists/AI may not be super precise when defining the subsegment, and tumors may grow out of organs, so we add a generous margin here.
    standard_ce: if True, we use a standard averaging for the BCE loss. Otherwise, we acerage the foreground and background voxel losses separately, and then sum the two losses.
    class_weights: optional 5D tensor to apply class weights. This is useful when dealing with imbalanced positives and negatives per class or datasets with many classes.
    Important: this loss assumes the output resolution is 1x1x1 mm, and that diammeters are in mm and volumes in mm^3. If the resolution is different, you should adjust the diameters and volumes accordingly or introduce a scaling factor.
    """
    global counter3

    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    if class_weights is not None:
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert len(class_weights.shape) == 5, f'Class weights should be 5D tensor, got {class_weights.shape}'
        #repeat channels to match the output shape
        class_weights = class_weights.repeat(out.shape[0], 1, out.shape[2], out.shape[3], out.shape[4])


    #get only the channels with lesions
    out = get_lesion_channels(out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes, assertion=False)
    unk_voxels = get_lesion_channels(unk_voxels, classes)
    labels = get_lesion_channels(labels, classes)
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)

    chosen_segment_mask = dilate_volume(chosen_segment_mask,subseg_dilation)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,unk_dilation)
    to_penalize = torch.ones_like(out)
    #remove the unk voxels from the penalization
    to_penalize = to_penalize * (1 - unk_voxels)
    #also remove the knwon labels
    to_penalize = to_penalize * (1 - labels)
    #but add back the chosen segment mask
    to_penalize = to_penalize + chosen_segment_mask
    #binarize
    to_penalize = (to_penalize > 0).float()


    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape

    losses = []
    losses_dice = []

    for B in range(out.shape[0]):#batch itens
        #assert diameters and violumes make sense
        assert torch.equal(tumor_diameters[B].sum(-1)>0, tumor_volumes[B]>0), f'Tumor diameters and volumes should be consistent, got {tumor_diameters[B]} and {tumor_volumes[B]}'
        
        #get correct batch and class
        x = out[B]
        tumor_seg = chosen_segment_mask[B]
        #current_x is still 4 D, with one class per tumor type. Assert at most one of these channels is non-zero (due to the chosen_segment_mask):
        assert (tumor_seg.sum((-1,-2,-3))>0).float().sum()<=1, f'Only one channel should be non-zero, got {tumor_seg.sum((-1,-2,-3))}'
        
        # if no tumor in this batch, create a zero pseudo label
        if tumor_seg.sum()==0 or tumor_volumes[B].sum()==0:
            # no tumor in this batch, create a zero pseudo label
            pseudo_mask = torch.zeros_like(x)
            if sigmoid:
                if not single_class:
                    #standard, use sigmoid
                    loss = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
                else:
                    #use softmax
                    loss = F.cross_entropy(x, pseudo_mask, reduction='none')
                #print('ball loss uses BCE with logits')
            else:
                if not single_class:
                    #assert x is in the range 0-1
                    assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                    #assert pseudo_mask is in the range 0-1
                    assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                    loss = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
                else:
                    #single class, but consider that softmax was already applied. Thus, use nll loss
                    #from one-hot to class indices: argmax
                    loss = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
            assert loss.shape == tumor_seg.shape
            loss = loss * to_penalize[B]
            if class_weights is not None:
                # apply class weights if provided
                loss = loss * class_weights[B]
            loss = loss.mean()
            if apply_dice_loss:
                if class_weights is not None:
                    w = class_weights[B]
                else:
                    w = None
                dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                losses_dice.append(dice_loss)
            losses.append(loss.mean())
            continue
        
        #get tumor class
        for c in range(x.shape[0]):
            if tumor_seg[c].sum()>0:
                x = x[c]
                penalize = to_penalize[B][c]
                if class_weights is not None:
                    c_weight = class_weights[B][c] #get the class weights for this batch and class
                else:
                    c_weight = None
                break
        tumor_seg = tumor_seg.sum(0)
        current_tumor_diameters = tumor_diameters[B]
        current_tumor_volumes = tumor_volumes[B]

        # Get the sort indices for tumor_volumes in descending order
        sorted_indices = torch.argsort(current_tumor_volumes, descending=True)

        # Filter indices to keep only those with volume > 0
        sorted_indices = sorted_indices[current_tumor_volumes[sorted_indices] > 0]
        #print('--------Sorted indices:', sorted_indices)
        #print('--------SORTED VOLUMES:', current_tumor_volumes[sorted_indices])
        #print('--------UNSORTED VOLUMES:', current_tumor_volumes)

        #Create the pseudo-mask
        pseudo_masks = []
        #update x for the next tumor: remove pseudo_mask, so that this tumor is not selected again.
        if sigmoid:
            x_iter = torch.sigmoid(x)*tumor_seg
        else:
            x_iter = x*tumor_seg
        for tumor_idx in sorted_indices:
            vol=current_tumor_volumes[tumor_idx].item()
            dia=current_tumor_diameters[tumor_idx]
            #get the maximum diameter
            max_diameter = torch.max(dia).item()
            assert max_diameter>0, f'Tumor diameter should be larger than 0, got {max_diameter}'
            assert vol>0, f'Tumor volume should be larger than 0, got {vol}'
            if vol==0 or max_diameter == 0:
                print('Found 0 tumor where it should not be')
                continue
            if max_diameter <= 1:
                print('Found 1mm diameter, increasing to 3')
                max_diameter = 3
            if vol <= 1:
                print('Found 1mm volume, increasing to 9')
                vol = 9
            #assert it is not zero
            #ball convolution: use isolate_tumor to get the top 'tumor_volume' voxels in the outpus, inside the best fitting ball position
            pseudo_mask = isolate_tumor(x_iter, diameter=max_diameter, margin=diameter_margin, gaussian=gaussian, 
                                        gaussian_std=gaussian_std, tumor_volume=vol)
            pseudo_masks.append(pseudo_mask)
            x_iter = x_iter * (1 - pseudo_mask) #remove the pseudo mask from the output, so that it is not selected again
        #stack the pseudo masks
        pseudo_mask = torch.stack(pseudo_masks).sum(0)
        #make sure the pseudo mask is binary
        pseudo_mask = (pseudo_mask > 0).float()

        #we can add a tolerance margin around the pseudo mask, where we do not penalize the outputs for not being zero
        if dilation_for_background>0:
            dilated_pseudo_mask=dilate_volume(pseudo_mask, dilation_for_background)
            border = dilated_pseudo_mask - pseudo_mask
            #assert border is binary
            assert (border>0).sum() == border.sum(), f'Border should be binary'
            assert border.shape == x.shape
        else:
            dilated_pseudo_mask = pseudo_mask
            border = torch.zeros_like(pseudo_mask)
        penalize=penalize * (1 - border)
        #penalize is a tensor with the voxels where we want to apply our losses to here

        #BCE loss with mask
        if sigmoid:
            if not single_class:
                BCE = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
            else:
                #single class
                BCE = F.cross_entropy(x, pseudo_mask, reduction='none')
        else:
            if not single_class:
                #assert x is in the range 0-1
                assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                #assert pseudo_mask is in the range 0-1
                assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                BCE = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
            else:
                #single class, but consider that softmax was already applied. Thus, use nll loss
                #from one-hot to class indices: argmax
                BCE = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
        assert (penalize.shape==BCE.shape), f'To penalize and BCE should have the same shape, got {penalize.shape} and {BCE.shape}'
        BCE = BCE * penalize #cut the loss gradient in the border. Remember that unk voxels were already removed from x

        #dice loss
        #dice loss
        if apply_dice_loss:
            #remove tumor surroundings, to avoid penalizing them: we are not super sure if this region is tumor or not.
            dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=penalize,sigmoid=sigmoid,class_weights=c_weight)
            if sigmoid:
                print('Dice loss:',dice_loss, 'Mean prediction:',torch.sigmoid(x).mean())
            else:
                print('Dice loss:',dice_loss, 'Mean prediction:',x.mean())
            #we make all voxels knwon because we alreay removed unknown voxels from x
            #print('Using dice loss inside the ball loss')

        if not standard_ce:
            #we separate foreground and background, calculate the average per-voxel loss for them separatelly, than sum it. We can use GRWP in the foreg. or not.
            if gwrp:
                #we do BCE for the entire channel, but we do not simply average it. We can use GWRP to average the tumor values (positive GT)
                #we add the pseudo-mask to boost its voxels values and concentrate GWRP there.
                assert pseudo_mask.sum() > 0, f'Pseudo mask should have at least one voxel, got {pseudo_mask.sum()}, volume is {vol} and diameter is {max_diameter}'
                if sigmoid:
                    foreg_weights = GlobalWeightedRankPooling(torch.sigmoid(x)*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                else:
                    foreg_weights = GlobalWeightedRankPooling(x*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                #print highest and lowest non-zero values in foreg_weights
                assert foreg_weights.sum() > 0.95 and foreg_weights.sum() < 1.05, f'GWRP weights should be normalized to 1, got {foreg_weights.sum()}'
                #renormlize gwrp weights so they sum to pseudo_mask.sum()
                foreg_weights = foreg_weights * pseudo_mask.sum()
                #print('GWRP Foreg weights range:', foreg_weights[foreg_weights>0].max(), foreg_weights[foreg_weights>0].min())
                #assert sum of foreg_weights is close to 1
                foreg_weights = foreg_weights*pseudo_mask
                assert BCE.shape == foreg_weights.shape, f'BCE and GWRP weights should have the same shape, got {BCE.shape} and {foreg_weights.shape}'
                loss_foreground = (BCE*foreg_weights)#.mean() #we can use mean here because 
            else:
                #print('Using simple mean for BCE loss')
                loss_foreground = (BCE*pseudo_mask)#.mean()
            
            #Background:
            bkg_weights = 1 - dilated_pseudo_mask
            loss_background = (BCE*bkg_weights)#.mean()
            
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(loss_background.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == loss_background.shape[0], f'Class weights {class_weights[B].shape} do not match loss_background shape {loss_background.shape}'
                loss_foreground = loss_foreground * c_weight
                loss_background = loss_background * c_weight
            loss_foreground = loss_foreground.mean()
            loss_background = loss_background.mean()

            loss = loss_foreground + loss_background
            losses.append(loss)#BCE loss
        else:
            #print('Using standard CE for BCE loss')
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(BCE.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == BCE.shape[0], f'Class weights {c_weight.shape} do not match BCE shape {BCE.shape}'
                BCE = BCE * c_weight
            BCE = BCE.mean()
            losses.append(BCE)#simple mean.

        if apply_dice_loss:
            losses_dice.append(dice_loss.mean())

        if counter3<10:

            counter3+=1
            os.makedirs('SanityBallLoss/'+str(counter), exist_ok=True)
            if sigmoid:
                save_tensor_as_nifti(torch.sigmoid(x),'SanityBallLoss/'+str(counter)+'/x')
            else:
                save_tensor_as_nifti(x,'SanityBallLoss/'+str(counter)+'/x')
            save_tensor_as_nifti(pseudo_mask,'SanityBallLoss/'+str(counter)+'/pseudo_mask')
            save_tensor_as_nifti(border,'SanityBallLoss/'+str(counter)+'/border')
            save_tensor_as_nifti(tumor_seg,'SanityBallLoss/'+str(counter)+'/tumor_segment')
            save_tensor_as_nifti((to_penalize[B].sum(0)>0).float(),'SanityBallLoss/'+str(counter)+'/to_penalize')
            if input_tensor is not None:
                save_tensor_as_nifti(input_tensor[B].squeeze(),'SanityBallLoss/'+str(counter)+'/input_volume')

            #save tumor volumes and diameters as yaml
            with open('SanityBallLoss/'+str(counter)+'/tumor_volumes.yaml', 'w') as file:
                yaml.dump(tumor_volumes.tolist(), file)
            with open('SanityBallLoss/'+str(counter)+'/tumor_diameters.yaml', 'w') as file:
                yaml.dump(tumor_diameters.tolist(), file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter)+'/known_voxels')
            l=losses[-1].item()
            if apply_dice_loss:
                l+=losses_dice[-1].item()
            if sigmoid:
                info=f'Volume in output: {torch.sigmoid(x).sum().item()}, Volume in report: {vol}, Loss: {l}'
            else:
                info=f'Volume in output: {x.sum().item()}, Volume in report: {vol}, Loss: {l}'
            print(info)
            #save the loss as yaml
            with open('SanityBallLoss/'+str(counter)+'/loss.yaml', 'w') as file:
                yaml.dump(l, file)
            #save the info as yaml
            with open('SanityBallLoss/'+str(counter)+'/info.yaml', 'w') as file:
                yaml.dump(info, file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter)+'/loss.yaml')

    return {'ball_loss_bce':torch.stack(losses).mean(),
            'ball_loss_dice':torch.stack(losses_dice).mean() if apply_dice_loss else torch.zeros_like(torch.stack(losses).mean())}


def save_tensor_as_nifti(tensor: torch.Tensor, filename: str):
    """
    Saves a torch tensor as a NIfTI file, assuming a voxel spacing of 1x1x1 mm.

    Args:
        tensor (torch.Tensor): A torch tensor of shape (H, W, D) or (1, H, W, D).
        filename (str): The output filename (should end with .nii or .nii.gz).
    """
    if 'nii.gz' not in filename:
        filename += '.nii.gz'

    # Ensure tensor is on CPU and convert to numpy array.
    np_array = tensor.detach().cpu().numpy()
    
    # If the tensor has an extra channel dimension, squeeze it.
    if np_array.ndim == 4 and np_array.shape[0] == 1:
        np_array = np_array.squeeze(0)
    
    # Create an identity affine (voxel sizes = 1 mm in all directions).
    affine = np.eye(4)
    
    # Create the NIfTI image and save.
    nifti_img = nib.Nifti1Image(np_array, affine)
    nib.save(nifti_img, filename)
    print(f"Saved NIfTI file to {filename}")


def apply_ball_convolution_and_save(input_size=(64, 64, 64), square_size=20,
                                    ball_diameter=15, gaussian=False, gaussian_std=3.0,
                                    output_filename='ball_convolution_output.nii.gz'):
    """
    Creates an input tensor with a centered cube (i.e., a 3D "square"),
    applies the ball convolution to it, prints the center coordinates of the input,
    prints the center of mass of the output, and saves the result as a NIfTI file.
    
    Args:
        input_size (tuple): Size of the 3D input (H, W, D).
        square_size (int): Size of the cube to insert in the center.
        ball_diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting in the ball.
        gaussian_std (float): Standard deviation for the Gaussian.
        output_filename (str): Path for the output NIfTI file.
    """
    # Create a 5D input tensor (B, C, H, W, D) filled with zeros
    x = torch.zeros((1, 1, *input_size), dtype=torch.float32)
    
    # Determine the center of the input
    center = [dim // 2 for dim in input_size]
    
    # Insert a cube (all ones) at the center of the input volume.
    half_square = square_size // 2
    x[0, 0,
    center[0]-half_square : center[0]+half_square+1,
    center[1]-half_square : center[1]+half_square+1,
    center[2]-half_square : center[2]+half_square+1] = 1.0

    # Print the center coordinates of the input
    print(f"Input center coordinates: {center}")
    
    # Apply the ball convolution over the input
    output = ball_convolution(x, ball_diameter, gaussian, gaussian_std)
    
    # Remove batch and channel dimensions and convert to a NumPy array
    output_np = output.squeeze().numpy()
    
    # Compute the center of mass of the output
    H, W, D = output_np.shape
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    total = np.sum(output_np)
    if total == 0:
        com = (0.0, 0.0, 0.0)
    else:
        com_x = np.sum(grid_x * output_np) / total
        com_y = np.sum(grid_y * output_np) / total
        com_z = np.sum(grid_z * output_np) / total
        com = (com_x, com_y, com_z)
    
    print(f"Center of mass of output: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
    
    # Create an identity affine (customize voxel sizes if needed)
    affine = np.eye(4)
    
    # Save the convolved output as a NIfTI file
    nii_img = nib.Nifti1Image(output_np, affine)
    nib.save(nii_img, output_filename)
    
    print(f"Saved ball convolution output to {output_filename}")


def generate_input_and_process_volume(input_size=(64, 64, 64), square_size=20, square_location='center',
                                        diameter=15, gaussian=False, gaussian_std=3.0, tumor_volume=100,
                                        output_input_filename='input_volume.nii.gz', output_mask_filename='tumor_mask.nii.gz'):
    """
    Generates an input volume with a cube (square in 3D) composed of random values, places it either in the
    center or in the corner of the volume, applies isolate_tumor, and saves both the input volume and the
    resulting tumor mask as NIfTI files.
    
    Args:
        input_size (tuple): The size of the 3D input volume (H, W, D).
        square_size (int): The edge-length of the cube to insert.
        square_location (str): Where to place the cube. Options: "center" or "corner".
        diameter (int): Diameter of the ball kernel for isolate_tumor.
        gaussian (bool): Whether to use Gaussian weighting in the ball convolution.
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): The number of voxels to select as the tumor volume.
        output_input_filename (str): File path to save the input volume (as .nii.gz).
        output_mask_filename (str): File path to save the tumor mask (as .nii.gz).
    
    Returns:
        None
    """

    # Create a 5D input tensor with shape (B, C, H, W, D)
    x = torch.zeros((1, 1, *input_size), dtype=torch.float32)
    
    # Insert a cube with random values
    if square_location.lower() == 'center':
        # Compute center and half-size
        center = [dim // 2 for dim in input_size]
        half_square = square_size // 2
        
        # Calculate starting indices so that the cube is centered
        start_x = center[0] - half_square
        start_y = center[1] - half_square
        start_z = center[2] - half_square
        
        # Make sure we get exactly square_size elements along each dimension
        x[0, 0, start_x:start_x+square_size, start_y:start_y+square_size, start_z:start_z+square_size] = \
            torch.rand((square_size, square_size, square_size))+0.5
    
    elif square_location.lower() == 'corner':
        # Place the cube at the (0,0,0) corner
        x[0, 0, 0:square_size, 0:square_size, 0:square_size] = torch.rand((square_size, square_size, square_size))
    
    else:
        raise ValueError("square_location must be either 'center' or 'corner'")
    
    # Save the input volume as a NIfTI file (save the spatial part: (H, W, D))
    input_np = x[0, 0].numpy()
    affine = np.eye(4)
    input_nii = nib.Nifti1Image(input_np, affine)
    nib.save(input_nii, output_input_filename)
    print(f"Saved input volume to {output_input_filename}")
    
    # Apply isolate_tumor to the input volume
    tumor_mask = isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume)
    
    # Save the tumor mask as a NIfTI file (convert to uint8 for a binary mask)
    tumor_mask_np = tumor_mask.numpy().astype(np.uint8)
    tumor_mask_nii = nib.Nifti1Image(tumor_mask_np, affine)
    nib.save(tumor_mask_nii, output_mask_filename)
    print(f"Saved tumor mask to {output_mask_filename}")