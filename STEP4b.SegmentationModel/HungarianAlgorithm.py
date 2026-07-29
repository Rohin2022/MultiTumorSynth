import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from einops import rearrange, repeat
from itertools import chain


def group_classes(class_names):
    """
    Create a class_groups object that groups lesion indices by lesion type.

    Args:
        class_names (list): List of class names.

    Returns:
        class_groups (list): List of lists, where each inner list contains indices of a specific lesion type.
    """
    class_groups = []
    lesion_map = {}

    # Iterate over the class names with their indices
    for idx, name in enumerate(class_names):
        # Check if the class name is a lesion
        if "_lesion_" in name:
            # Extract the lesion type prefix (e.g., "kidney_lesion", "pancreatic_lesion")
            lesion_type = "_".join(name.split("_")[:2])

            # Group the indices by lesion type
            if lesion_type not in lesion_map:
                lesion_map[lesion_type] = []
            lesion_map[lesion_type].append(idx)

    # Append the grouped indices to the class_groups list
    class_groups.extend(lesion_map.values())

    return class_groups

#@torch.jit.script
def cdice_similarity_old(input_mask, target_mask, eps=1e-5):
    """
    input mask: (B, N, HW) #probabilities [0, 1]
    target_mask: (B, K, HW) #binary
    """

    input_mask = input_mask.unsqueeze(2) #(B, N, 1, HW)
    target_mask = target_mask.unsqueeze(1) #(B, 1, K, HW)

    #binarize
    binary_input = (input_mask > 0.5).to(torch.uint8)
    binary_target = (target_mask > 0.5).to(torch.uint8)

    #(B, N, 1, HW) * (B, 1, K, HW) --> (B, N, K, HW)
    intersection = torch.sum(binary_input * binary_target, dim=-1)
    cardinalities = binary_input + binary_target
    cardinalities = torch.sum(cardinalities, dim=-1)

    dice = ((2. * intersection + eps) / (cardinalities + eps))
    return dice

def cdice_similarity(input_mask, target_mask, eps=1e-5):
    """
    This implementation should consume less memory than the previous one. The previous one caused RAM issues.
    input_mask: (B, N, HW) [already thresholded, e.g. boolean 0/1]
    target_mask: (B, K, HW) [boolean 0/1]
    Returns: (B, N, K) of pairwise dice
    """
    # 1) Compute per-channel sums (|A| and |B|):
    #    sum_input: (B, N)
    #    sum_target: (B, K)
    sum_input = input_mask.sum(dim=-1)      # shape (B, N)
    sum_target = target_mask.sum(dim=-1)    # shape (B, K)

    # 2) Compute intersections via a batched matrix multiply:
    #    intersection[i, n, k] = sum over HW of input_mask[i, n] * target_mask[i, k]
    #    This can be done with torch.einsum, torch.bmm, or .matmul with rearrange.
    #    For example:
    intersection = torch.einsum('bnh, bkh -> bnk', input_mask.float(), target_mask.float())
    #   shape: (B, N, K)

    # 3) Compute dice: 2 * intersection / (|A| + |B|)
    #    But note union = |A| + |B| - intersection, so for standard dice
    #    2|A∩B| / (|A| + |B|). We'll just do:
    cardinalities = sum_input.unsqueeze(-1) + sum_target.unsqueeze(1)  # shape (B, N, K)
    dice = (2.0 * intersection + eps) / (cardinalities + eps)
    return dice

#@torch.jit.script
def dice_score(input_mask, target_mask, eps=1e-5):
    """
    input mask: (B * K, HW) #probabilities [0, 1]
    target_mask: (B * K, HW) #binary
    """
    print('Max input:',input_mask.max())
    print('Max target:',target_mask.max())

    dims = tuple(range(1, input_mask.ndimension()))
    input_mask=input_mask.to(torch.uint8)
    target_mask=target_mask.to(torch.uint8)
    intersections = torch.sum(input_mask * target_mask, dims) #(B, N)
    print('Intersections:',intersections)
    cardinalities = torch.sum(input_mask + target_mask, dims)
    print('union:',cardinalities)
    dice = ((2. * intersections + eps) / (cardinalities + eps))
    return dice

class HungarianMatcher(nn.Module):
    """
    Heavily inspired by https://github.com/facebookresearch/detr/blob/master/models/matcher.py.
    """

    def __init__(self, class_channels=None):
        super(HungarianMatcher, self).__init__()
        #add class channels: list of lists. Each list has the channels for a given class (e.g., kidney lesions)
        #we will only match within classes
        self.class_channels = class_channels


    @torch.no_grad()
    def forward(self, input_mask, target_mask, input_class_prob=None, target_class=None, target_sizes=None, one_class=False,to_cpu=False):
        #This is a wrapper of run. It applies the matching for outputs with fixed channels. I.e., channel 1 is always kidney lesion, channel 2 too, channel 5 is liver,.....
        #self.class_channels is a list of lists. Each list has the channels for a given class (e.g., kidney lesions). We will only match within classes. 
        if to_cpu:
            input_mask=input_mask.cpu()
            target_mask=target_mask.cpu()
            if input_class_prob is not None:
                input_class_prob=input_class_prob.cpu()
            if target_class is not None:
                target_class=target_class.cpu()
            if target_sizes is not None:
                target_sizes=target_sizes.cpu()

        #make input_mask and target_mask boolean
        input_mask = input_mask > 0.5
        target_mask = target_mask > 0.5
        if self.class_channels is None:
            return self.run(input_class_prob=input_class_prob, input_mask=input_mask, target_class=target_class, 
                            target_mask=target_mask, target_sizes=target_sizes, one_class=one_class)
        else:
            # Match within classes
            indices_inpt = [[], []]
            indices_tgt = [[], []]

            # Track handled channels
            handled_channels = set()

            # Iterate over each class group
            for class_channels in self.class_channels:
                assert class_channels == list(range(class_channels[0], class_channels[-1] + 1))  # Ensure valid ranges
                handled_channels.update(class_channels)

                # Create partial masks for current class
                input_mask_partial = input_mask[:, class_channels]
                target_mask_partial = target_mask[:, class_channels]

                # Partial probabilities and classes (if provided)
                input_class_prob_partial = input_class_prob[:, class_channels] if input_class_prob is not None else None
                target_class_partial = target_class[:, class_channels] if target_class is not None else None

                # Run the matcher for this class group
                inp_pos_indices, tgt_pos_indices, inp_neg_indices = self.run(
                    input_class_prob=input_class_prob_partial,
                    input_mask=input_mask_partial,
                    target_class=target_class_partial,
                    target_mask=target_mask_partial,
                    target_sizes=None,
                    one_class=True
                )

                # Offset the channel indices to map back to the full tensor
                channel_offset = class_channels[0]

                indices_inpt[0].append(inp_pos_indices[0])
                indices_inpt[1].append(inp_pos_indices[1] + channel_offset)
                indices_tgt[0].append(tgt_pos_indices[0])
                indices_tgt[1].append(tgt_pos_indices[1] + channel_offset)

                # Map negative indices to trailing zero masks
                num_class_channels = len(class_channels)
                tgt_neg_indices = torch.arange(num_class_channels - len(inp_neg_indices[1]),
                                            num_class_channels, device=input_mask.device)
                indices_inpt[0].append(inp_neg_indices[0])
                indices_inpt[1].append(inp_neg_indices[1] + channel_offset)
                indices_tgt[0].append(inp_neg_indices[0])
                indices_tgt[1].append(tgt_neg_indices + channel_offset)

            # Handle unmatched channels by directly mapping them
            all_channels = set(range(input_mask.size(1)))
            unmatched_channels = sorted(all_channels - handled_channels)
            if unmatched_channels:
                for unmatched_channel in unmatched_channels:
                    batch_indices = torch.arange(input_mask.size(0), device=input_mask.device)
                    unmatched_indices = torch.full((input_mask.size(0),), unmatched_channel, device=input_mask.device)

                    # Append directly mapped indices
                    indices_inpt[0].append(batch_indices)
                    indices_inpt[1].append(unmatched_indices)
                    indices_tgt[0].append(batch_indices)
                    indices_tgt[1].append(unmatched_indices)

            # Concatenate all indices to ensure consistent dimensions
            inputs_ids = (torch.cat(indices_inpt[0], dim=0), torch.cat(indices_inpt[1], dim=0))
            outputs_ids = (torch.cat(indices_tgt[0], dim=0), torch.cat(indices_tgt[1], dim=0))

            return inputs_ids, outputs_ids
            
    @torch.no_grad()
    def run(self, input_class_prob, input_mask, target_class, target_mask, target_sizes=None,
                one_class=False):
        """
        input_class_prob: (B, N, N_CLASSES) #probabilities --- for unet, this is fixed, just depending on the label order
        input_mask: (B, N, HxWxL) #probabilities [0, 1] --- batch, classes, height, width
        target_class: (B, K) #long indices
        target_mask: (B, K, HxWxL) #bool --- batch, classes, height, width
        target_sizes: (B,) #number of masks that not are padding (i.e. no class) --- Will be the number of zero masks, K - (target_mask.sum(dim=-1)==0).sum().item()
        """
        if len(input_mask.shape) != 3:
            input_mask=rearrange(input_mask, 'b n h w l -> b n (h w l)')
            target_mask=rearrange(target_mask, 'b k h w l -> b k (h w l)')

        device = input_mask.device
        B, N = input_mask.size()[:2]
        K = target_mask.size(1)
        zero_in_the_middle = False

        if target_sizes is None:
            sum_masks = target_mask.sum(dim=-1)
            zero_sizes = (sum_masks == 0).sum(dim=-1)
            #check if, for each batch item, the zeros are at the end
            #print('Zero sizes:',zero_sizes)
            for i in range(B):
                if zero_sizes[i] > 0:
                    #print('Sum masks:',sum_masks[i])
                    if not (sum_masks[i, -zero_sizes[i]:] == 0).all():#zero masks should be at the end for us to ignore them in the Hungarian algorithm. If not, we consider them, increasing computational cost.
                        zero_in_the_middle = True
            target_sizes = K - zero_sizes

        #we want similarity matrices to size (B, N, K) --- For each batch item, a B X N matrix, which gives the similarity between output n and target k
        #where N is number of predicted objects and K is number of gt objects
        #(B, N, C)[(B, N, K)] --> (B, N, K)
        if not one_class:
            sim_class = input_class_prob.gather(-1, repeat(target_class, 'b k -> b n k', n=N)) # ---- with fixed unet output channels, this is binary
        sim_dice = cdice_similarity(input_mask, target_mask)

        if not one_class:
            #final cost matrix (RQ x SQ from the paper, eqn 9)
            sim = (sim_class * sim_dice).cpu() #(B, N, K)
        else:
            sim = sim_dice.cpu()

        #each example in batch, ignore null objects in target (i.e. padding)
        if not zero_in_the_middle:
            indices = [linear_sum_assignment(s[:, :e], maximize=True) for s,e in zip(sim, target_sizes)]
        else:
            #pedro: we may have zero masks in the middle of the target, because we use random cropping.
            indices = [linear_sum_assignment(s[:, :], maximize=True) for s,e in zip(sim, target_sizes)]

        #at this junctions everything is matched, now it's just putting
        #the indices into easily usable formats

        input_pos_indices = []
        target_pos_indices = []
        input_neg_indices = []
        input_indices = np.arange(0, N)
        for i, (inp_idx, tgt_idx) in enumerate(indices):
            input_pos_indices.append(torch.as_tensor(inp_idx, dtype=torch.long, device=device))
            target_pos_indices.append(torch.as_tensor(tgt_idx, dtype=torch.long, device=device))
            input_neg_indices.append(
                torch.as_tensor(
                    np.setdiff1d(input_indices, inp_idx), dtype=torch.long, device=device
                )
            )

        #here the lists of indices have variable lengths
        #and sizes; make 1 tensor of size (B * N_pos) for all
        #positives first: shared by input_pos_indices and target_pos_indices
        batch_pos_idx = torch.cat(
            [torch.full_like(pos, i) for i, pos in enumerate(input_pos_indices)]
        )
        batch_neg_idx = torch.cat(
            [torch.full_like(neg, i) for i, neg in enumerate(input_neg_indices)]
        )
        input_pos_indices = torch.cat(input_pos_indices)
        target_pos_indices = torch.cat(target_pos_indices)
        input_neg_indices = torch.cat(input_neg_indices)

        inp_pos_indices = (batch_pos_idx, input_pos_indices)
        tgt_pos_indices = (batch_pos_idx, target_pos_indices)
        inp_neg_indices = (batch_neg_idx, input_neg_indices)

        #To match inputs and masks, do:
        #matched_input_mask = input_mask[inp_pos_indices]
        #matched_target_mask = target_mask[tgt_pos_indices]
        #negative_mask = input_mask[neg_indices]
        #kmax-ddeplab: Hungarian algorithm is only applied to the final output, and the same indices are used in deep superivision
        return inp_pos_indices, tgt_pos_indices, inp_neg_indices

class PQLoss(nn.Module):
    def __init__(self, alpha=0.75, eps=1e-5):
        super(PQLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.xentropy = nn.CrossEntropyLoss(reduction='none')
        self.matcher = HungarianMatcher()
        self.negative_xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, input_class, target_mask, target_class, target_sizes):
        """
        input_class: (B, N, N_CLASSES) #logits
        input mask: (B, N, H, W) #probabilities [0, 1]
        target_class: (B, K) #long indices
        target_mask: (B, K, H, W) #binary
        """
        #apply softmax to get probabilities from logits
        B, N, num_classes = input_class.size()
        input_mask = F.softmax(input_mask, dim=1)
        input_class_prob = F.softmax(input_class, dim=-1)
        input_mask = rearrange(input_mask, 'b n h w -> b n (h w)')
        target_mask = rearrange(target_mask, 'b k h w -> b k (h w)')

        #match input and target
        inp_pos_indices, tgt_pos_indices, neg_indices = self.matcher(
            input_class_prob, input_mask, target_class,
            target_mask, target_sizes
        )

        #select masks and labels by indices
        #(B < len(inp_pos_indices) <= B * K)
        #(0 <= len(neg_indices) <= B * (N - K))
        matched_input_class = input_class[inp_pos_indices]
        matched_input_class_prob = input_class_prob[inp_pos_indices]
        matched_target_class = target_class[tgt_pos_indices]
        negative_class = input_class[neg_indices]

        matched_input_mask = input_mask[inp_pos_indices]
        matched_target_mask = target_mask[tgt_pos_indices]
        negative_mask = input_mask[neg_indices]

        #NP is len(inp_pos_indices)
        #NN is len(neg_indices)
        with torch.no_grad():
            class_weight = matched_input_class_prob.gather(-1, matched_target_class[:, None]) #(NP,)
            dice_weight = dice_score(matched_input_mask, matched_target_mask, self.eps) #(NP,)

        cross_entropy = self.xentropy(matched_input_class, matched_target_class) #(NP,)
        dice = dice_score(matched_input_mask, matched_target_mask, self.eps) #(NP,)
    
        #eqn 10
        #(1 - dice) is so that the minimum loss value is 0 and not -1
        l_pos = (class_weight * (1 - dice) + dice_weight * cross_entropy).mean()

        #eqn 11
        negative_target_class = torch.zeros(
            size=(len(negative_class),), dtype=target_class.dtype, device=target_class.device
        )
        l_neg = self.negative_xentropy(negative_class, negative_target_class).mean()
        
        #eqn 12
        return self.alpha * l_pos + (1 - self.alpha) * l_neg

#-----------------------
### Auxiliary Losses ###
#-----------------------

class InstanceDiscLoss(nn.Module):
    def __init__(self, temp=0.3, eps=1e-5):
        super(InstanceDiscLoss, self).__init__()
        self.temp = temp
        self.eps = eps

    def forward(self, mask_features, target_mask, target_sizes):
        """
        mask_features: (B, D, H, W) #g
        target_mask: (B, K, H, W) #m
        """

        #downsample input and target by 4 to get (B, H/4, W/4)
        mask_features = mask_features[..., ::4, ::4]
        target_mask = target_mask[..., ::4, ::4]

        device = mask_features.device

        #eqn 16
        t = torch.einsum('bdhw,bkhw->bkd', mask_features, target_mask)
        t = F.normalize(t, dim=-1) #(B, K, D)

        #get batch and mask indices from target_sizes
        batch_indices = []
        mask_indices = []
        for bi, size in enumerate(target_sizes):
            mindices = torch.arange(0, size, dtype=torch.long, device=device)
            mask_indices.append(mindices)
            batch_indices.append(torch.full_like(mindices, bi))

        batch_indices = torch.cat(batch_indices, dim=0) #shape: (torch.prod(target_sizes), )
        mask_indices = torch.cat(mask_indices, dim=0)

        #create logits and apply temperature
        logits = torch.einsum('bdhw,bkd->bkhw', mask_features, t)
        logits = logits[batch_indices, mask_indices] #(torch.prod(target_sizes), H, W)
        logits /= self.temp

        #select target_masks
        m = target_mask[batch_indices, mask_indices] #(torch.prod(target_sizes), H, W)

        #flip so that there are HW examples for torch.prod(target_sizes) classes
        logits = rearrange(logits, 'k h w -> (h w) k')
        m = rearrange(m, 'k h w -> (h w) k')

        #eqn 17
        numerator = torch.logsumexp(m * logits, dim=-1) #(HW,)
        denominator = torch.logsumexp(logits, dim=-1) #(HW,)
        
        #log of quotient is difference of logs
        return (-numerator + denominator).mean()

class MaskIDLoss(nn.Module):
    def __init__(self):
        super(MaskIDLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, target_mask):
        """
        input_mask: (B, N, H, W) #logits
        target_mask: (B, H, W) #long indices of maskID in N
        """
        return self.xentropy(input_mask, target_mask)

class SemanticSegmentationLoss(nn.Module):
    def __init__(self, method='cross_entropy'):
        super(SemanticSegmentationLoss, self).__init__()
        if method != 'cross_entropy':
            raise NotImplementedError
        else:
            #they don't specify the loss function
            #could be regular cross entropy or
            #dice loss or focal loss etc.
            #keep it simple for now
            self.xentropy = nn.CrossEntropyLoss()

    def forward(self, input_mask, target_mask):
        """
        input_mask: (B, NUM_CLASSES, H, W) #logits
        target_mask: (B, H, W) #long indices
        """
        return self.xentropy(input_mask, target_mask)

class MaXDeepLabLoss(nn.Module):
    def __init__(
        self,
        pq_loss_weight=3,
        instance_loss_weight=1,
        maskid_loss_weight=0.3,
        semantic_loss_weight=1,
        alpha=0.75,
        temp=0.3,
        eps=1e-5
    ):
        super(MaXDeepLabLoss, self).__init__()
        self.pqw = pq_loss_weight
        self.idw = instance_loss_weight
        self.miw = maskid_loss_weight
        self.ssw = semantic_loss_weight
        self.pq_loss = PQLoss(alpha, eps)
        self.instance_loss = InstanceDiscLoss(temp, eps)
        self.maskid_loss = MaskIDLoss()
        self.semantic_loss = SemanticSegmentationLoss()

    def forward(self, input_tuple, target_tuple):
        """
        input_tuple: (input_masks, input_classes, input_semantic_segmentation) Tensors
        target_tuple: (gt_masks, gt_classes, gt_semantic_segmentation) NestedTensors
        """
        input_masks, input_classes, input_ss = input_tuple
        gt_masks, gt_classes, gt_ss = [t.tensors for t in target_tuple]
        target_sizes = target_tuple[0].sizes

        pq = self.pq_loss(input_masks, input_classes, gt_masks, gt_classes, target_sizes)
        instdisc = self.instance_loss(input_masks, gt_masks.float(), target_sizes)

        #create the mask for maskid loss using argmax on ground truth
        maskid = self.maskid_loss(input_masks, gt_masks.argmax(1))
        semantic = self.semantic_loss(input_ss, gt_ss)

        loss_items = {'pq': pq.item(), 'semantic': semantic.item(), 
                      'maskid': maskid.item(), 'instdisc': instdisc.item()}

        total_loss = self.pqw * pq + self.ssw * semantic + self.miw * maskid + self.idw * instdisc 

        return total_loss, loss_items