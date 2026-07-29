import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block, get_norm, get_act
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb
import numpy as np

from .trans_layers import TransformerBlock

def make_classifier(
    chan_num,                        # list of channels from your encoder
    dim_head,                        # list/tuple of dim‑head per level
    conv_block,
    expansion,
    attn_drop,
    proj_drop,
    map_size,
    proj_type,
    norm,
    act,
    out_class_number,
    binarize_input=False,
    num_input_ch=None,
    class_list_cls=None,
    class_list_seg=None,
):
    """Return nn.Sequential(down_after_1, aux, down_after_2, aux, down_after_3)."""

    if num_input_ch is None:
        num_input_ch = chan_num[-1]
    aux = aux_layer()
    down_after_1 = down_block(num_input_ch, chan_num[-1], 1, 0, conv_block=conv_block,
                            kernel_size=[3,3,3], down_scale=[2,2,2], heads=1,
                            dim_head=dim_head[-1], expansion=expansion, attn_drop=attn_drop,
                            proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
    down_after_2 = down_block(chan_num[-1], chan_num[-1], 1, 0, conv_block=conv_block,
                            kernel_size=[3,3,3], down_scale=[2,2,2], heads=1,
                            dim_head=dim_head[-1], expansion=expansion, attn_drop=attn_drop,
                            proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
    down_after_3 = down_block(chan_num[-1], chan_num[-1]*2, 1, 1, conv_block=conv_block,
                            kernel_size=[3,3,3], down_scale=[2,2,2], heads=4,
                            dim_head=(chan_num[-1])//4, expansion=expansion, attn_drop=attn_drop,
                            proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
    #we use a few medformer layer to reduce the dimensionality of the last deep convolutional output (before the final layer)
    extra_layer=nn.Sequential(down_after_1,aux,down_after_2, aux, down_after_3)
    # Add a classification branch after the segmentation decoder
    classifier = ClassificationBranch(in_dim=chan_num[-1]*2, 
                                            num_classes=out_class_number, reducer=False, #we have few channels, so no need to reduce them
                                            extra_layer=extra_layer, 
                                            heads=4, dim_head=16, mlp_dim=256, reduced_dim=chan_num[-1]*2,
                                            binarize_input=binarize_input,
                                            class_list_cls=class_list_cls,
                                            class_list_seg=class_list_seg,
                                            )
    
    classifier.num_input_ch = num_input_ch
    
    return classifier

class ClassificationBranch(nn.Module):
    def __init__(self, in_dim=160, reduced_dim=64, heads=4, dim_head=16, mlp_dim=320, 
                 num_classes=3, extra_layer=None,
                 reducer=True,
                 binarize_input=False,
                 class_list_cls=None,
                 class_list_seg=None,
                 ):
        """
        For multi-tumor classification, the voxel_choice input indicates which tumor we want to classify. It is a binary tensor,
        with the same shape as the input, and one voxel is set to 1. The rest are 0. To classify multiple tumors, just run this module
        multiple times, each time with a different voxel_choice input. At inference, you can take the centers of all tumors predicted in 
        segmentation.
        """
        
        super().__init__()
        
        # Add a reducer to lower the channel dimension
        if reducer:
            self.reducer = nn.Conv3d(in_dim, reduced_dim, kernel_size=1)
        else:
            self.reducer = nn.Identity()
        # Optionally, add an extra layer if needed
        self.extra_layer = extra_layer
        # Use a transformer block with the reduced dimension
        self.transformer = TransformerBlock(
            dim=reduced_dim,         # embedding dimension is now reduced_dim
            depth=1,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )
        # Classification head from the reduced dimension to num_classes
        self.head = nn.Linear(reduced_dim, num_classes)
        self.binarize_input = binarize_input
        self.class_list_cls = class_list_cls
        self.class_list_seg = class_list_seg

    def forward(self, x, segmentation_out=None, voxel_choice=None):
        """
        x: features of the segmenter
        segmentation_out: segmentation output from the segmenter
        voxel_choice: binary tensor indicating which tumor to classify
        """
        
        if segmentation_out is not None:
            #concatenate the segmentation output with the features
            x = torch.cat((x, segmentation_out), dim=1)
        if voxel_choice is not None:
            #concatenate the voxel_choice with the features
            x = torch.cat((x, voxel_choice), dim=1)

            
        if self.binarize_input:
            #during training, we randomly choose to skip binarization, or binarize with threshold 0.5, or binarize with a random threshold between 0.1 and 0.9
            #the threshold used is added as an additional channel to the input
            #assert self.class_list_cls is inside self.class_list_seg
            if self.class_list_cls is None or self.class_list_seg is None:
                raise ValueError('class_list_cls and class_list_seg must be provided when binarize_input is True.')
            assert all(c in self.class_list_seg for c in self.class_list_cls), f"All classes in class_list_cls must be in class_list_seg, got class_list_cls:{self.class_list_cls} and class_list_seg:{self.class_list_seg}"
            lesion_classes_in_seg = [i for i,c in enumerate(self.class_list_seg) if c in self.class_list_cls]
            if self.training:
                if torch.rand(1).item() > 0.5: #50% probability we simply skip binarization during training
                    x = torch.sigmoid(x/4) #sigmoid for probabilites, scaled down to avoid saturation and preserve gradients
                    #add an additional channel of zeros, meaning that you did not binarize
                    skip_channel = torch.zeros_like(x[:, :1, :, :, :])
                    x = torch.cat((x, skip_channel), dim=1)
                    print('Skipping binarization during training',flush=True)
                else:
                    x = torch.sigmoid(x) #no need to care about saturation during binarization
                    #binarize
                    x=x.detach()
                    #randomly select a threshold between 0.1 and 0.9
                    th_scalar = torch.rand(1).item() * 0.8 + 0.1
                    #we onlt want this thresholding to apply to lesion classes, not organs
                    #now make threshold 0.5 for non-lesion classes
                    non_lesion_mask = torch.ones((1, x.size(1), 1, 1, 1), device=x.device, dtype=x.dtype)
                    non_lesion_mask[:, lesion_classes_in_seg, :, :, :] = 0
                    threshold = th_scalar * (1 - non_lesion_mask) + 0.5 * non_lesion_mask
                    #threshold x
                    x = (x > threshold).float()
                    #add an additional channel of threshold, meaning that you binarized with this threshold
                    threshold_channel = torch.full_like(x[:, :1, :, :, :], th_scalar)
                    x = torch.cat((x, threshold_channel), dim=1)
                    #we want to tell the model what was the threshold we used!
                    print('Binarizing with threshold', th_scalar, 'during training',flush=True)
                #check which masks are empty, we can ignore the classifier for these samples
                lesion_seg = x[:, lesion_classes_in_seg, :, :, :]
                empty_masks = (lesion_seg.sum(dim=[-1,-2,-3],keepdim=False) == 0).float()
            else:
                x = torch.sigmoid(x) #no need to care about saturation during binarization
                #inference mode: we want to run over thresholded outputs, to avoid shortcuts and ensure classification is based on segmentation
                # we want to binarize with all thresholds between 0.1 and 0.9 and average the results
                #we expand on the batch dimension
                thresholds = [0.1,0.3,0.5,0.7,0.9] #you can add more thresholds for more accuracy
                thresholded_x = []
                non_lesion_mask = torch.ones((1, x.size(1), 1, 1, 1), device=x.device, dtype=x.dtype)
                non_lesion_mask[:, lesion_classes_in_seg, :, :, :] = 0
                empty_masks_list = []
                for th_scalar in thresholds:
                    th = th_scalar * (1 - non_lesion_mask) + 0.5 * non_lesion_mask
                    x_th = (x > th).float()
                    threshold_channel = torch.full_like(x_th[:, :1, :, :, :], th_scalar)
                    x_th = torch.cat((x_th, threshold_channel), dim=1)
                    thresholded_x.append(x_th)
                    lesion_seg = x_th[:, lesion_classes_in_seg, :, :, :]
                    empty_masks = (lesion_seg.sum(dim=[-1,-2,-3],keepdim=False) == 0).float()
                    empty_masks_list.append(empty_masks)
                x = torch.cat(thresholded_x, dim=0)  # new batch size is B * len(thresholds)
                empty_masks = torch.cat(empty_masks_list, dim=0)
            
            #we need to find which lesion masks are empty
            
            
                
        # x is [B, in_dim, D, H, W]
        #print('Shape of x in classification branch:', x.shape)
        if self.extra_layer is not None:
            x, tmp_map = self.extra_layer(x)
        else:
            tmp_map = torch.zeros(1, device=x.device)  # dummy value so gradient flows if needed


        x = self.reducer(x)  # now x becomes [B, reduced_dim, D, H, W]

        # Flatten and rearrange to [B, L, reduced_dim]
        B, C, D, H, W = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        # Pass through the transformer block
        x = self.transformer(x)  # remains [B, L, reduced_dim]
        # Global average pooling
        x = x.mean(dim=1)  # [B, reduced_dim]
        # Classification head produces output [B, num_classes]
        x = self.head(x)
        # Ensure gradient flows through tmp_map (if needed for DDP)
        x = x + 0 * tmp_map.sum()
        
        if self.binarize_input:
            #assert that number of lesion classes in seg is equal to number of classes in cls
            assert len(lesion_classes_in_seg) == x.shape[-1], f"Number of lesion classes in segmentation ({len(lesion_classes_in_seg)}) must be equal to number of classes in classification output ({x.shape[-1]})"
            #for cases where masks were empty, we ignore the output
            x = x*(1-empty_masks) + (-10)*empty_masks  # set to large negative value for empty masks (low probability after sigmoid)
            if not self.training:
                #in inference mode, we expanded the batch dimension by len(thresholds). 
                # Now, we need to average the results back to original batch size
                orig_batch_size = x.shape[0] // len(thresholds)
                #add a new dimension for thresholds
                x = x.view(orig_batch_size, len(thresholds), -1)  # [B, len(thresholds), num_classes]
                #take probabilities
                x = torch.sigmoid(x)
                #average over thresholds
                x = x.mean(dim=1)  # [B, num_classes]
                #convert back to logits
                x = torch.logit(x, eps=1e-6)
        return x
    
class Gate(nn.Module):
    def __init__(self,class_list_seg,class_list_cls,normalize=False):
        super(Gate, self).__init__()
        self.class_list_seg = class_list_seg
        self.class_list_cls = class_list_cls
        self.normalize = normalize

        self.gated_classes = {}
        #get tumor class indices in both class_list_seg and class_list_cls
        for i in range(len(self.class_list_seg)):
            for j in range(len(self.class_list_cls)):
                if self.class_list_seg[i] == self.class_list_cls[j]:
                    seg_idx = i
                    cls_idx = j
                    self.gated_classes[self.class_list_seg[i]] = {
                        'seg_idx': seg_idx,
                        'cls_idx': cls_idx
                    }
                    break

    def forward(self, x_seg, x_cls):
        # x_seg: [B, C_all, D, H, W]
        # x_cls: [B, C_cls] -- only tumor classes
        #assert shapes
        assert len(x_seg.shape) == 5, "x_seg should be 5D tensor"
        assert len(x_cls.shape) == 2, "x_cls should be 2D tensor"
        #assert both tensors are between 0 and 1 (sigmoid output)
        assert torch.all(x_seg >= 0) and torch.all(x_seg <= 1), "x_seg should be between 0 and 1"
        assert torch.all(x_cls >= 0) and torch.all(x_cls <= 1), "x_cls should be between 0 and 1"

        out=[]
        for i,class_name in enumerate(self.class_list_seg):
            if class_name not in self.gated_classes:
                out.append(x_seg[:,i,:,:,:])
            else:
                seg_idx = self.gated_classes[class_name]['seg_idx']
                cls_idx = self.gated_classes[class_name]['cls_idx']

                # Get the segmentation and classification maps
                seg_map = x_seg[:, seg_idx, :, :, :]
                clss = x_cls[:, cls_idx]

                if self.normalize:
                    #the values of seg_map that are above 0.5 are multiplied by the maximum value of seg_map, per B
                    seg_map = self.positive_normalization(seg_map)
                
                #gating: multiply the segmentation map with the classification output. Basically, we make the probabilities conditional
                # on the classification output
                seg_map = seg_map * clss.view(-1, 1, 1, 1)

                # Append the gated segmentation map to the output list
                out.append(seg_map)

        # Concatenate the output list along the channel dimension
        out = torch.stack(out, dim=1)
        return out


    def positive_normalization(self,seg_map):
        """
        seg_map: Tensor of shape (B, H, W, D).

        For each batch b:
        - Values > 0.5 are divided by max(seg_map[b]).
        - Values <= 0.5 remain unchanged.
        """
        # seg_map is [B, H, W, D]
        B = seg_map.shape[0]

        # 1) Find max over each batch item: shape [B]
        max_vals = seg_map.view(B, -1).max(dim=1).values
        # 2) Reshape for broadcasting: shape [B, 1, 1, 1]
        max_vals = max_vals.view(B, 1, 1, 1)

        # 3) Create a mask for values > 0.5
        mask = seg_map > 0.5
        # 4) Use torch.where to replace values > 0.5 with value / max_val[b]
        out = torch.where(mask, seg_map / (max_vals+1e-5).detach(), seg_map)#detach is key! We do not want the gradient flowing through the denominator, it shoudl be treated as a constant

        #print('Max and min of gate-normalized out:', out.max(), out.min())
        return out


class ePAIStage2Aggregator(nn.Module):
    """The idea of this module is to concatenate a CT scan and a lesion label, provided by the stage one. Stage 2 will classify the lesion type (pdac, pnet, cyst).
    We provide multiple ways to concatenate this information:
    - concat: we add a new channel to inconv.conv1, the first convolution in medformer. We keep the other kernels unchanged, and we initialize the new kernel with very small numbers.
    - sum: we add the lesion label, divided by 5, to the CT.
    - borders: we conpute the lesion label borders, and make them 0.8 in the CT.
    This module has no learnable parameter. Thus, you do not need to care about passing it to the optimizer, and you can apply it to the model after instantiation. Just like update_output_layer_onk."""

    def __init__(self, mode, model=None):
        assert mode in ['concat', 'sum', 'borders'], f"Invalid mode: {mode}. Choose from ['concat', 'sum', 'borders']"
        super(ePAIStage2Aggregator, self).__init__()
        self.mode = mode
        if mode == 'concat':
            model.inc.conv1 = expand_conv3d_input(model.inc.conv1, init_std=1e-3)#substitute the model's first convolution by one with one more channel
        elif mode == 'sum':
            # We do not need to change the model, we just need to add the lesion label to the input
            pass
        elif mode == 'borders':
            self.edge_detector = EdgeDetector3D()  # Initialize the edge detector
            
    def forward(self, img, lesion_label=None):
        """
        This function is used to pre-process the image and label, depending on the mode.
        """
        #ensure image and label are 5D: B, C, D, H, W
        assert len(img.shape) == 5, f"Image should be 5D tensor, got {len(img.shape)}D"
        assert len(lesion_label.shape) == 5, f"Lesion label should be 5D tensor, got {lesion_label.shape}"
        assert img[0].shape == lesion_label[0].shape, f"Image and lesion label should have the same spatial dimensions, got {img[0].shape} and {lesion_label[0].shape}"
        
        if self.mode == 'concat':
            # Concatenate the lesion label as a new channel, remember that the img comes first (see expand_conv3d_input)
            img = torch.cat((img, lesion_label), dim=1)  # Add lesion label as a new channel
            return img
        elif self.mode == 'sum':
            img = img + lesion_label / 5.0  # Add the lesion label, scaled down
            return img
        elif self.mode == 'borders':
            # Compute the edges of the lesion label and add them to the image
            lesion_edges = self.edge_detector(lesion_label)
            #now, we need to make these edges 0.8. We first erase the original edges from img:
            img = img * (1 - lesion_edges)  # Set edges to 0
            #now, we set the edges to 0.8. 0.8 is quite arbitrary indeed. But the idea is: We do not want to set to 0 or 1, because the ct may have large areas of 0 and 1, due to clipping of HU. Conversely, a large area of 0.8, with no variation at all, is quite impossible. So, it should be easy for the network to detect.
            img = img + lesion_edges * 0.8  # Set edges to 0.8
            return img
        

class EdgeDetector3D(nn.Module):
    """
    A fixed (non‑learnable) 3D edge detector using the 26‑connected Laplacian.
    
    Given an input binary volume of shape (B,1,D,H,W) or (1,D,H,W) or (D,H,W),
    the module convolves it with a 3×3×3 kernel where:
      - center weight = +26
      - all 26 neighbors = −1
    This yields zero in uniform regions and non‑zero at any 26‑connected border.
    The output is then thresholded to a binary edge map.
    """
    def __init__(self):
        super().__init__()
        # build the 3×3×3 kernel
        w = torch.full((3,3,3), -1.0, dtype=torch.float32)
        w[1,1,1] = 26.0
        # register as buffer so it's saved but not learnable
        self.register_buffer('kernel', w.unsqueeze(0).unsqueeze(0))  # shape (1,1,3,3,3)
        #this is a laplacian kernel. The output is not zero only if some of the voxels covered by the kernel are different. Otherwise, the output is zero. It is an edge detector.

    def forward(self, x: torch.Tensor, th=0.5):
        """
        x: torch.Tensor with shape
             - (D,H,W), or
             - (1, D,H,W), or
             - (B,1, D,H,W)
        we binarize the input at th (0.5 by default).
        returns a float tensor of same spatial dims, 1.0 on edges, 0.0 elsewhere.
        """
        # add batch+channel dims if missing
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 4:  # assume (1,D,H,W)
            x = x.unsqueeze(0)
            squeeze_batch = True
            
        # ensure data is binary
        x = (x > th).float()

        # convolve with fixed Laplacian kernel
        resp = F.conv3d(x, self.kernel.type_as(x), padding=1)

        # threshold absolute response to get binary edges
        edges = (resp.abs() > 0).float()

        # remove extra dims if needed
        if squeeze_batch:
            edges = edges.squeeze(0).squeeze(0)
        return edges


import torch
import torch.nn as nn

def expand_conv3d_input(orig_conv: nn.Conv3d, init_std: float = 1e-3) -> nn.Conv3d:
    """
    Return a new Conv3d with one additional input channel.
    
    The original weights are copied exactly into the first `orig_conv.in_channels` input channels of the new layer,
    preserving channel order: the new layer's first input channels correspond to the original layer's inputs.
    The extra new channel is appended AT THE END, and its weights are initialized from N(0, init_std^2)->default is a small initialization.
    If the original layer has a bias, it is copied unchanged.
    
    Args:
        orig_conv (nn.Conv3d): the existing convolution to expand.
        init_std (float): standard deviation for initializing the new-channel weights.
        
    Returns:
        nn.Conv3d: a new convolutional layer with `orig_conv.in_channels + 1` input channels,
                    original weights preserved, and one new channel initialized small.
    """
    # Extract original parameters
    old_weights = orig_conv.weight.data.clone()
    old_bias = orig_conv.bias.data.clone() if orig_conv.bias is not None else None

    old_in_ch = orig_conv.in_channels
    out_ch = orig_conv.out_channels
    kD, kH, kW = orig_conv.kernel_size
    stride = orig_conv.stride
    padding = orig_conv.padding
    dilation = orig_conv.dilation
    groups = orig_conv.groups
    has_bias = orig_conv.bias is not None

    # Create expanded conv layer
    new_conv = nn.Conv3d(
        in_channels=old_in_ch + 1,
        out_channels=out_ch,
        kernel_size=(kD, kH, kW),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=has_bias,
    )

    # Copy weights and initialize new channel
    with torch.no_grad():
        # Copy existing weights
        new_conv.weight[:, :old_in_ch, ...].copy_(old_weights)
        # Initialize new channel weights to small random values
        torch.nn.init.normal_(
            new_conv.weight[:, old_in_ch:old_in_ch+1, ...],
            mean=0.0, std=init_std
        )
        # Copy bias if present
        if has_bias:
            new_conv.bias.copy_(old_bias)

    return new_conv
class aux_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out, map_out = x
        return out + 0 * map_out.sum()

class MedFormer(nn.Module):
    
    def __init__(self, 
        in_chan, 
        num_classes, 
        base_chan=32, 
        map_size=[4,8,8], 
        conv_block='BasicBlock', 
        conv_num=[2,1,0,0, 0,1,2,2], 
        trans_num=[0,1,2,2, 2,1,0,0], 
        chan_num=[64,128,256,320,256,128,64,32], 
        num_heads=[1,4,8,16, 8,4,1,1], 
        fusion_depth=2, 
        fusion_dim=320, 
        fusion_heads=4, 
        expansion=4, attn_drop=0., 
        proj_drop=0., 
        proj_type='depthwise', 
        norm='in', 
        act='gelu', 
        kernel_size=[3,3,3,3], 
        scale=[2,2,2,2], 
        aux_loss=False,
        classification_branch=False,
        gate_cls=False,normalize_on_gate=False,
        class_list_seg=None,class_list_cls=None,
        aggregator_mode = None,  # \'concat', 'sum', 'borders', or None
        cls_on_output=False,  # Whether to add a classification branch after the segmentation decoder as well
        cls_on_segmentation=False,  # Whether to add a classification branch after the segmentation output (and in deep supervision)
        binarize_cls_on_segmentation=False, # Whether to binarize the input of the classifier on segmentation output
        clip_branch=False,
        clip_feats=768,
        attenuation_cls='none',
        train_att_MLP_on_mask_only=False,
        tumor_classifier=False,
        loss_weight_att = 1,
        loss_weight_cls=1,
        age_and_sex_into_classifier=False,
        ):
        super().__init__()


        #if conv_block == 'BasicBlock':
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]

        
        conv_block = get_block(conv_block)
        norm = get_norm(norm)
        act = get_act(act)

        self._cls_build_cfg = dict(
            chan_num=chan_num,
            dim_head=dim_head,
            conv_block=conv_block,
            expansion=expansion,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            map_size=map_size,
            proj_type=proj_type,
            norm=norm,
            act=act,
        )
        
        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, block=conv_block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block=conv_block, kernel_size=kernel_size[1], down_scale=scale[0], norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block=conv_block, kernel_size=kernel_size[2], down_scale=scale[1], heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block=conv_block, kernel_size=kernel_size[3], down_scale=scale[2], heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)


        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block=conv_block, kernel_size=kernel_size[3], up_scale=scale[3], heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)

        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block=conv_block, kernel_size=kernel_size[2], up_scale=scale[2], heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True, no_map_out=True)

        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block=conv_block, kernel_size=kernel_size[1], up_scale=scale[1], norm=norm, act=act, map_shortcut=False)

        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block=conv_block, kernel_size=kernel_size[0], up_scale=scale[0], norm=norm, act=act, map_shortcut=False)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv3d(chan_num[5], num_classes, kernel_size=1)

        self.outc = nn.Conv3d(chan_num[7], num_classes, kernel_size=1)

        if classification_branch:
            self.classification_branch = ClassificationBranch(num_classes=len(class_list_cls),
                                                              extra_layer=down_block(chan_num[3], chan_num[3]//2, 0, 1, conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=4, dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True))
        else:
            self.classification_branch = None
            
        if clip_branch:
            self.clip_branch = ClassificationBranch(num_classes=clip_feats,
                                                    extra_layer=down_block(chan_num[3], chan_num[3]//2, 0, 1, conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=4, dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True))
        else:
            self.clip_branch = None

        if gate_cls:
            self.gate_cls = Gate(class_list_seg=class_list_seg, class_list_cls=class_list_cls, normalize=normalize_on_gate)
        else:
            self.gate_cls = None

        if gate_cls and not classification_branch:
            raise ValueError('Gate cls is True but classification branch is False. This is not allowed.')
        
        
        if cls_on_output:
            self.cls_on_output = make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                out_class_number=len(class_list_cls))
        else:
            self.cls_on_output = None
            
        self.age_and_sex_into_classifier = age_and_sex_into_classifier
        if cls_on_segmentation:
            if age_and_sex_into_classifier:
                num_input_ch = num_classes+2
            else:
                num_input_ch = num_classes
            if binarize_cls_on_segmentation:
                num_input_ch += 1
            self.cls_on_segmentation = make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                out_class_number=len(class_list_cls),num_input_ch = num_input_ch,
                                                binarize_input=binarize_cls_on_segmentation,
                                                class_list_seg=class_list_seg,class_list_cls=class_list_cls)
            #binarize_cls_on_segmentation: makes the input of the classifier binary in 50% of cases during training, and with multiple thresholds averaged during inference
        else:
            self.cls_on_segmentation = None
            
        self.aggregator_mode = aggregator_mode
        self.aggregator = None
        
        if class_list_seg is not None:
            patterns = ('lesion', 'pdac', 'pnet', 'cyst')
            tumor_cls = [c for c in class_list_seg if any(p in c for p in patterns)]
        
        if attenuation_cls!= 'none':
            assert attenuation_cls in ['none', 'simple', 'MLP','large','neuron'], f"Invalid attenuation_cls: {attenuation_cls}. Choose from ['none', 'simple', 'MLP','neuron','large']"
            if attenuation_cls == 'MLP':
                self.att_classifier = attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only, loss_weight=loss_weight_att)
            elif attenuation_cls == 'neuron':
                self.att_classifier = attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only, loss_weight=loss_weight_att,
                                                           out_features=1)
            elif attenuation_cls == 'simple':
                self.att_classifier = simple_classifier(class_list_seg)
            elif attenuation_cls == 'large':
                m =  make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                       out_class_number=3*len(tumor_cls), num_input_ch = (2*len(tumor_cls)+1))
                self.att_classifier=attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only,
                                                                   model=m,calculate_HU=False, loss_weight=loss_weight_att)
        else:
            self.att_classifier = None
            
        if tumor_classifier:
            m =  make_classifier(chan_num, dim_head, conv_block, expansion, attn_drop, proj_drop, map_size, proj_type, norm, act,
                                                       out_class_number=10*len(tumor_cls)*3, num_input_ch = (2*len(tumor_cls)))
            self.tumor_classifier = attribute_classifier(class_list_seg,train_on_mask_only=train_att_MLP_on_mask_only,
                                                                   model=m,calculate_HU=False,mode='tumor_classifier',
                                                                   loss_weight=loss_weight_cls)
        else:
            self.tumor_classifier = None
            
       


        
    
    def set_aggregator(self):
        self.aggregator = ePAIStage2Aggregator(mode=self.aggregator_mode, model=self)


    def forward(self, x, stage_1_out=None,labels=None, cut_attenuation_grad = False,
                age=None,sex=None):
        
        if self.aggregator is not None:
            # If the aggregator is set, apply it to the input image and stage 1 output
            assert stage_1_out is not None, "Stage 1 output must be provided when using the aggregator"
            x = self.aggregator(x, stage_1_out)
       
        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)

        if self.classification_branch:
            y_class = self.classification_branch(x4)
        else:
            y_class = None
            
        if self.clip_branch:
            y_clip = self.clip_branch(x4)
        else:
            y_clip = None
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        

        out, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, semantic_map = self.up2(out, x2, semantic_map, map_list[0])
        
        if self.age_and_sex_into_classifier:
            # age, sex: [B,1]
            B = x.shape[0]
            device = x.device
            dtype = x.dtype

            if self.training:
                keep_age = (torch.rand(B, 1, device=device) > 0.5).to(dtype)  # 50% keep
                keep_sex = (torch.rand(B, 1, device=device) > 0.5).to(dtype)
            else:
                keep_age = torch.ones(B, 1, device=device, dtype=dtype)  # keep all during inference
                keep_sex = torch.ones(B, 1, device=device, dtype=dtype)

            age = age.to(device=device, dtype=dtype) * keep_age
            sex = sex.to(device=device, dtype=dtype) * keep_sex

            age = age.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])
            sex = sex.view(B, 1, 1, 1, 1).expand(B, 1, *x.shape[-3:])


        if self.aux_loss:
            aux_out = self.aux_out(out)
            aux_out = F.interpolate(aux_out, size=x.shape[-3:], mode='trilinear', align_corners=True)
            if self.att_classifier is not None:
                aux_att = self.att_classifier(x,aux_out,labels,cut_attenuation_grad=cut_attenuation_grad)
            else:
                aux_att = None
            if self.cls_on_segmentation is not None:
                if self.age_and_sex_into_classifier:
                    aux_out_age_sex = torch.cat((aux_out, age, sex), dim=1)
                    y_class_on_seg_aux = self.cls_on_segmentation(aux_out_age_sex)
                else:
                    y_class_on_seg_aux = self.cls_on_segmentation(aux_out)
            else:
                y_class_on_seg_aux = None
        else:
            aux_out = None
            aux_att = None
            y_class_on_seg_aux = None

        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)
        
        if self.cls_on_output is not None:
            #print('Shape of out is:', out.shape)                
            y_class_2 = self.cls_on_output(out)
            #y_class is now [B, num_classes_cls], where num_classes_cls is the number of classes in the classification branch
        else:
            y_class_2 = None
    
        out = self.outc(out)
        
        if self.cls_on_segmentation is not None:
            if self.age_and_sex_into_classifier:
                print(f'Age and sex fed to classifier', flush=True)
                out_age_sex = torch.cat((out, age, sex), dim=1)
                y_class_on_seg = self.cls_on_segmentation(out_age_sex)
            else:
                y_class_on_seg = self.cls_on_segmentation(out)
        else:
            y_class_on_seg = None

        if self.gate_cls:
            out = self.gate_cls(torch.sigmoid(out), torch.sigmoid(y_class))
            #assert out is in the range 0-1
            assert (out>=0).all() and (out<=1).all(), f'Gate out is not in the range 0-1, its min is: {out.min()}, its max is: {out.max()}'
            #remember to remove sigmoid from the dice loss and not use BCE with logits loss---we are already applying sigmoid here. Can you do gating before the sigmoid?
            #raise ValueError('Max and min of out:', out.max(), out.min())
            
        if self.att_classifier is not None:
            out_att = self.att_classifier(x,out,labels,cut_attenuation_grad=cut_attenuation_grad)
        else:
            out_att = None
        
        if self.tumor_classifier is not None:
            y_tumor = self.tumor_classifier(x, out, labels, cut_attenuation_grad=cut_attenuation_grad)
        else:
            y_tumor = None
        
        return self.prepare_return(out, aux_out=aux_out, y_class=y_class, y_class_2=y_class_2,
                                   y_clip=y_clip, aux_att=aux_att, out_att=out_att, y_tumor=y_tumor,
                                   y_class_on_seg_aux=y_class_on_seg_aux, y_class_on_seg=y_class_on_seg)
        
    def prepare_return(
        self,
        out,
        aux_out=None,
        y_class=None,
        y_class_2=None,
        y_clip=None,
        aux_att=None,
        out_att=None,
        y_tumor=None,
        y_class_on_seg_aux=None,
        y_class_on_seg=None,
    ):
        # 1) Build the primary output exactly as before
        primary = [out, aux_out] if self.aux_loss else out
        
        retur = {'segmentation': primary}
        
        if self.classification_branch:
            retur['classification'] = y_class
        if self.cls_on_output is not None:
            retur['classification on output'] = y_class_2
        if self.clip_branch:
            retur['clip'] = y_clip
        if out_att is not None:
            retur['attenuation'] = [aux_att,out_att] if aux_att is not None else out_att
        if y_tumor is not None:
            retur['tumor diameters'] = y_tumor
        if self.cls_on_segmentation is not None:
            retur['classification on segmentation'] = [y_class_on_seg_aux,y_class_on_seg] if y_class_on_seg_aux is not None else y_class_on_seg

        return retur


        
    def rebuild_cls_on_segmentation(self, num_input_ch: int, verbose: bool = False):
        """
        Rebuild cls_on_segmentation so its first layers match the new segmentation channel count.
        Preserves pretrained weights by:
        (1) copying back any parameters/buffers whose keys+shapes match
        (2) for conv-like tensors (4D/5D) whose shapes differ only in channel dims,
            copying overlapping channels
        (3) for 1D tensors (bias/norm/running stats), copying overlapping entries
        """
        if self.cls_on_segmentation is None:
            return

        old_cls = self.cls_on_segmentation
        old_sd = old_cls.state_dict()

        out_class_number = old_cls.head.out_features
        binarize_input = getattr(old_cls, "binarize_input", False)

        cfg = self._cls_build_cfg
        new_cls = make_classifier(
            chan_num=cfg["chan_num"],
            dim_head=cfg["dim_head"],
            conv_block=cfg["conv_block"],
            expansion=cfg["expansion"],
            attn_drop=cfg["attn_drop"],
            proj_drop=cfg["proj_drop"],
            map_size=cfg["map_size"],
            proj_type=cfg["proj_type"],
            norm=cfg["norm"],
            act=cfg["act"],
            out_class_number=out_class_number,
            num_input_ch=num_input_ch,
            binarize_input=binarize_input,
            class_list_cls=getattr(old_cls, "class_list_cls", None),
            class_list_seg=getattr(old_cls, "class_list_seg", None),
        )

        new_sd = new_cls.state_dict()
        load_sd = {}

        copied_exact = 0
        copied_partial = 0
        skipped = 0

        for k, v_new in new_sd.items():
            v_old = old_sd.get(k, None)
            if v_old is None:
                skipped += 1
                continue

            # (1) exact match
            if v_old.shape == v_new.shape:
                load_sd[k] = v_old
                copied_exact += 1
                continue

            # (2) conv weights: 2D conv -> 4D [O,I,kH,kW], 3D conv -> 5D [O,I,kD,kH,kW]
            if v_old.ndim in (4, 5) and v_new.ndim == v_old.ndim:
                # require kernel sizes match
                if v_old.shape[2:] == v_new.shape[2:]:
                    tmp = v_new.clone()
                    o = min(v_old.shape[0], v_new.shape[0])
                    i = min(v_old.shape[1], v_new.shape[1])
                    tmp[:o, :i, ...] = v_old[:o, :i, ...]
                    load_sd[k] = tmp
                    copied_partial += 1
                    continue

            # (3) 1D tensors: bias, norm weight/bias, running stats
            if v_old.ndim == 1 and v_new.ndim == 1:
                tmp = v_new.clone()
                n = min(v_old.shape[0], v_new.shape[0])
                tmp[:n] = v_old[:n]
                load_sd[k] = tmp
                copied_partial += 1
                continue

            # otherwise skip
            skipped += 1

        missing, unexpected = new_cls.load_state_dict(load_sd, strict=False)

        if verbose:
            print(
                f"[rebuild_cls_on_segmentation] exact={copied_exact} partial={copied_partial} skipped={skipped} "
                f"missing_after_load={len(missing)} unexpected_after_load={len(unexpected)}",
                flush=True
            )

        self.cls_on_segmentation = new_cls



def update_output_layer_onk(model, original_classes, new_classes, copy_pancreas=False,binarize_cls_on_segmentation=False,
                            age_and_sex=False):
    """
    Update the model's final output layers so that they produce outputs for the new set of classes.
    For segmentation layers (model.outc and model.aux_out), we update them to have len(new_classes) outputs.
    For the classification branch (model.classification_branch.head) we update it only for lesion classes,
    that is, only classes whose name contains 'lesion'. Similarly, for the Gate module, we set:
        - class_list_seg = new_classes  (all segmentation classes)
        - class_list_cls = new_classes filtered to those containing 'lesion' / benign/malignant.
    
    Args:
        model (nn.Module): The pretrained model instance that has attributes outc, and possibly aux_out, classification_branch, gate_cls.
        original_classes (list of str): The original full list of class names (e.g., segmentation channels) used in the checkpoint.
        new_classes (list of str): The new full list of class names (e.g., segmentation channels).
        copy_pancreas (bool): If True, copy the weights of the pancreas class from the original model to all classes in the new model.
    Returns:
        model: The updated model.
    """
    # For classification, consider only classes with the word "lesion".
    new_class_cls = [cls for cls in new_classes if (("background" in cls) or ("lesion" in cls) or ('pdac' in cls) or ('pnet' in cls) or ('cyst' in cls))]
    old_class_cls = [cls for cls in original_classes if ("lesion" in cls)]
    new_class_no_malig_benign = [cls for cls in new_classes if ('malig' not in cls) and ('benign' not in cls)]
    malig_cls = any(('malign' in c) or ('benign' in c) for c in new_classes)
    malig_classes = [cls for cls in new_classes if (('malign' in cls) or ('benign' in cls))]
    new_class_cls = new_class_cls + malig_classes
    

    # Helper: update a Conv3d layer given an old layer and a desired new number of output channels.
    def update_conv(old_conv, new_out_channels, full_class_list):
        in_channels = old_conv.in_channels
        new_conv = nn.Conv3d(
            in_channels,
            new_out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
        )
        # For each new class in full_class_list, if it exists in original_classes, copy the corresponding weight.
        for new_idx, new_cls in enumerate(full_class_list):
            if (new_cls not in original_classes) and copy_pancreas:
                # Copy the pancreas class weights to all new classes.
                orig_idx = original_classes.index('pancreatic_lesion')
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
        
            if new_cls in original_classes:
                orig_idx = original_classes.index(new_cls)
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
            
            if new_cls not in original_classes and (('malignant' in new_cls) or ('benign' in new_cls)) and \
                (new_cls.replace('malignant','lesion').replace('benign','lesion') in original_classes):
                # When adding the benign/malignant classes, use the weights for the old lesion class for initialization.
                orig_idx = original_classes.index(new_cls.replace('malignant','lesion').replace('benign','lesion'))
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
                
        return new_conv

    # Update model.outc (segmentation layer) using the full new_classes.
    old_outc = model.outc
    if old_outc.out_channels != len(new_classes):
        print("Updating model.outc from {} to {} outputs".format(old_outc.out_channels, len(new_classes)))
        model.outc = update_conv(old_outc, len(new_classes), new_classes)
    else:
        print("model.outc already has {} outputs.".format(len(new_classes)))

    # Update model.aux_out if present.
    if hasattr(model, 'aux_out') and model.aux_loss:
        old_aux = model.aux_out
        if old_aux.out_channels != len(new_classes):
            print("Updating model.aux_out from {} to {} outputs".format(old_aux.out_channels, len(new_classes)))
            model.aux_out = update_conv(old_aux, len(new_classes), new_classes)
        else:
            print("model.aux_out already has {} outputs.".format(len(new_classes)))
    
    # Update classification branch head.
    if hasattr(model, 'classification_branch') and (model.classification_branch is not None):
        old_head = model.classification_branch.head
        if old_head.out_features != len(new_class_cls):
            print("Updating classification branch head from {} to {} outputs".format(old_head.out_features, len(new_class_cls)))
            in_features = old_head.in_features
            new_head = nn.Linear(in_features, len(new_class_cls))
            # Copy weights for overlapping lesion classes.
            for new_idx, new_cls in enumerate(new_class_cls):
                if (new_cls not in original_classes) and copy_pancreas:
                    orig_idx = old_class_cls.index('pancreatic_lesion')
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
                if new_cls in old_class_cls:
                    orig_idx = old_class_cls.index(new_cls)
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
            model.classification_branch.head = new_head
        else:
            print("Classification branch head already has {} outputs.".format(len(new_class_cls)))
        assert model.classification_branch.head.out_features == len(new_class_cls)
    
    
    if hasattr(model, 'cls_on_output') and model.cls_on_output is not None:
        old_head = model.cls_on_output.head
        if old_head.out_features != len(new_class_cls):
            print("Updating cls_on_output branch head from {} to {} outputs".format(old_head.out_features, len(new_class_cls)))
            in_features = old_head.in_features
            new_head = nn.Linear(in_features, len(new_class_cls))
            # Copy weights for overlapping lesion classes.
            for new_idx, new_cls in enumerate(new_class_cls):
                if (new_cls not in original_classes) and copy_pancreas:
                    orig_idx = old_class_cls.index('pancreatic_lesion')
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
                if new_cls in old_class_cls:
                    orig_idx = old_class_cls.index(new_cls)
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
            model.cls_on_output.head = new_head
        else:
            print("cls_on_output branch head already has {} outputs.".format(len(new_class_cls)))
        assert model.cls_on_output.head.out_features == len(new_class_cls)
        
    if hasattr(model, 'cls_on_segmentation') and model.cls_on_segmentation is not None:
        old_head = model.cls_on_segmentation.head
        if age_and_sex:
            expected_input_ch = len(new_classes) + 2
        else:
            expected_input_ch = len(new_classes)
        if  binarize_cls_on_segmentation:
            expected_input_ch += 1
        change_input_ch = (getattr(model.cls_on_segmentation, "num_input_ch", None) != expected_input_ch)
        if old_head.out_features != len(new_class_cls) or change_input_ch:
            model.cls_on_segmentation.class_list_seg = new_classes.copy()
            model.cls_on_segmentation.class_list_cls = new_class_cls.copy()
            #copy old weights:
            if change_input_ch:
                model.rebuild_cls_on_segmentation(num_input_ch=expected_input_ch)
                print(f'Number of classes: {len(new_classes)}')
                print(f'Class list: {new_classes}')
            print("Updating cls_on_segmentation branch head from {} to {} outputs".format(old_head.out_features, len(new_class_cls)))
            in_features = old_head.in_features
            new_head = nn.Linear(in_features, len(new_class_cls))
            
            # Copy weights for overlapping lesion classes.
            for new_idx, new_cls in enumerate(new_class_cls):
                if (new_cls not in original_classes) and copy_pancreas:
                    orig_idx = old_class_cls.index('pancreatic_lesion')
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
                if new_cls in old_class_cls:
                    orig_idx = old_class_cls.index(new_cls)
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
            model.cls_on_segmentation.head = new_head
        else:
            print("cls_on_segmentation branch head already has {} outputs.".format(len(new_class_cls)))
        assert model.cls_on_segmentation.head.out_features == len(new_class_cls)


    
    # Update Gate module: segmentation list is new_classes; classification list is new_class_cls.
    if hasattr(model, 'gate_cls') and (model.gate_cls is not None):
        print("Updating Gate module class lists.")
        model.gate_cls.class_list_seg = new_class_no_malig_benign.copy()
        model.gate_cls.class_list_cls = new_class_cls.copy()
        # Rebuild internal mapping.
        model.gate_cls.gated_classes = {}
        for i, seg_cls in enumerate(model.gate_cls.class_list_seg):
            for j, cls_cls in enumerate(model.gate_cls.class_list_cls):
                if seg_cls == cls_cls:
                    model.gate_cls.gated_classes[seg_cls] = {'seg_idx': i, 'cls_idx': j}
                    break

    if hasattr(model,'att_classifier') and model.att_classifier is not None:
        model.att_classifier.class_list=new_class_no_malig_benign
    if hasattr(model,'tumor_classifier') and model.tumor_classifier is not None:
        model.tumor_classifier.class_list=new_class_no_malig_benign

    return model


def tumor_to_organ(tumor_name):
    """
    Convert a tumor class name to an organ class name:
    - Remove '_lesion'
    - Substitutes some known patterns (like 'pancreatic' -> 'pancreas')
    - Randomly chooses a sided version for kidney, adrenal_gland, lung, or femur.
    """
    base = tumor_name.replace('_lesion', '')
    lower_base = base.lower()
    if lower_base == 'pancreatic':
        return 'pancreas'
    elif lower_base == 'kidney':
        return ['kidney_right', 'kidney_left']
    elif lower_base == 'adrenal_gland':
        return ['adrenal_gland_right', 'adrenal_gland_left']
    elif lower_base == 'adrenal':
        return ['adrenal_gland_right', 'adrenal_gland_left']
    elif lower_base == 'lung':
        return ['lung_right', 'lung_left']
    elif lower_base == 'femur':
        return ['femur_right', 'femur_left']
    elif lower_base == 'gallbladder':
        return 'gall_bladder'#added 28 apr 2025
    else:
        return base

from functools import reduce
import operator

def get_mask_lesion_masks(out: torch.Tensor, class_list):
    """
    Returns:
      tumors        (B,T,D,H,W)
      tumour_organs (B,T,D,H,W)  – matched organ mask for each tumour
    """
    idx = {c: i for i, c in enumerate(class_list)}  # O(1) lookup

    tumors, organs = [], []
    indices = []

    for cls in class_list:
        if any(tag in cls for tag in ('lesion', 'pdac', 'pnet', 'cyst')):
            tumors.append(out[:, idx[cls]])          # (B,D,H,W)
            indices.append(idx[cls])  # keep track of indices for organ mask
            org = tumor_to_organ(cls)
            if isinstance(org, list):
                if any(o not in idx for o in org):
                    organ_mask = torch.ones_like(out[:, 0])
                else:
                    masks = [out[:, idx[o]] for o in org]
                    # element‑wise max across sides
                    organ_mask = reduce(torch.maximum, masks)
            else:
                if org not in idx:
                    organ_mask = torch.ones_like(out[:, 0])
                else:
                    organ_mask = out[:, idx[org]]
            organs.append(organ_mask)

    tumors = torch.stack(tumors, dim=1)         # (B,T,…
    organs = torch.stack(organs, dim=1)
    return tumors, organs, indices
     

def extract_hu(ct,
               out,
               class_list):
    """
    Extracts “soft” HU statistics (mean and standard deviation) from the CT volume,
    using mask_tumor and mask_organ as weights.
    """
    assert ct.dim() == 5, f"CT input must be 5D tensor (B, C, D, H, W), got {ct.dim()}D"
    assert out.dim() == 5, f"Mask tumor must be 5D tensor (B, C, D, H, W), got {out.dim()}D"
    
    out = torch.sigmoid(out)  # ensure mask is in [0,1]
    eps = 7e-5  # small epsilon to avoid division by zero
    
    if out.isnan().any():
        raise ValueError("NaN values detected in segmenter's output, cannot compute HU statistics.")
    
    mask_tumor, mask_organ, indices = get_mask_lesion_masks(out, class_list)
    
    
    #standardize ct - this is needed for meand and std not to explode
    vmin = ct.amin(dim=(-1, -2, -3), keepdim=True)  # per‑volume minimum
    vmax = ct.amax(dim=(-1, -2, -3), keepdim=True)  # per‑volume maximum
    ct   = (ct - vmin) / (vmax - vmin + eps)
    
    

    mask_o = mask_organ * (1.0 - mask_tumor)
    mask_t = mask_tumor

    # 2) Compute weighted (soft) mean for the organ:
    #    numerator = sum(mask_o * ct)
    #    denominator = sum(mask_o)
    soft_mean_o = (mask_o * ct).sum(dim=(-1,-2,-3), keepdim=True) / (mask_o.sum(dim=(-1,-2,-3), keepdim=True) + eps)

    # 3) Compute weighted (soft) variance for the organ:
    #    var = sum[ mask_o * (ct - mean_o)^2 ] / sum(mask_o)
    
    soft_var_o = ( mask_o * (ct - soft_mean_o).pow(2) ).sum(dim=(-1,-2,-3), keepdim=True) / (mask_o.sum(dim=(-1,-2,-3), keepdim=True) + eps)
    soft_std_o = torch.sqrt(soft_var_o.clamp(min=0) + eps)

    # 4) Repeat for the tumor region:
    soft_mean_t = (mask_t * ct).sum(dim=(-1,-2,-3), keepdim=True) / (mask_t.sum(dim=(-1,-2,-3), keepdim=True) + eps)
    soft_var_t = ( mask_t * (ct - soft_mean_t).pow(2) ).sum(dim=(-1,-2,-3), keepdim=True) / (mask_t.sum(dim=(-1,-2,-3), keepdim=True) + eps)
    soft_std_t = torch.sqrt(soft_var_t.clamp(min=0) + eps)
    
    soft_mean_o = soft_mean_o.squeeze(-1).squeeze(-1).squeeze(-1)
    soft_mean_t = soft_mean_t.squeeze(-1).squeeze(-1).squeeze(-1)
    soft_std_o = soft_std_o.squeeze(-1).squeeze(-1).squeeze(-1)
    soft_std_t = soft_std_t.squeeze(-1).squeeze(-1).squeeze(-1)

    # 5) stack in the last dimension
    retur = [soft_mean_o, soft_std_o, soft_mean_t, soft_std_t, soft_mean_t - soft_mean_o, soft_std_t - soft_std_o, mask_o.mean(dim=(-1,-2,-3)), mask_t.mean(dim=(-1,-2,-3))]
    retur = torch.stack(retur, dim=-1)  # shape (B, C, 8)
    
    expected_shape = torch.Size((ct.shape[0], mask_tumor.shape[1], 8))
    assert retur.shape == expected_shape
    
    #check for nans
    if torch.isnan(retur).any():
        print(f'NaN detected in HU statistics output:, {retur}',flush=True)
        return torch.zeros_like(retur.detach()), indices
        #raise ValueError(f"NaN detected in HU statistics output:, {retur}")

    return retur, indices

def pad_channels(att,class_list, indices):
    #pad channels so that the shape of att is B,len(class_list),3
    tmp=[]
    for i in range(len(class_list)):
        if i in indices:
            tmp.append(att[:, indices.index(i)])
        else:
            tmp.append(torch.zeros_like(att[:, 0]))  # zero padding for missing classes
    att = torch.stack(tmp, dim=1)
    return att

def straight_through_trick(x,th=0.5):
    """
    This function allows gradients to flow through thresholding.
    """
    raise ValueError('This causes training instability, do not use it.')
    x= x - x.detach() + (x.detach()>th).float()
    return x


class _GradScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x

    @staticmethod
    def backward(ctx, grad_output):
        #print('GradScaler backward called with factor:', ctx.factor,flush=True)
        return grad_output * ctx.factor, None        # None = no grad wrt factor


class GradScaler(nn.Module):
    """ Identity layer whose backward gradient is multiplied by `factor`. """
    def __init__(self, factor: float):
        super().__init__()
        self.factor = float(factor)

    def forward(self, x):
        return _GradScaleFn.apply(x, self.factor)

import copy
class attribute_classifier(torch.nn.Module):
    """
    A MLP for attenuation values (mean and std) to classify lesions as hypo-attenuating, mixed/iso-attenuating, or hyper-attenuating.
    """
    def __init__(self, class_list, in_features=8, out_features=3,train_on_mask_only=False, model = 'MLP', calculate_HU=True,
                 mode = 'attenuation_classifier',loss_weight=1):
        super(attribute_classifier, self).__init__()
        if model == 'MLP':
            self.model = nn.Sequential(
                GradScaler(loss_weight),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, out_features),
                GradScaler(1/loss_weight)
            )
        elif model == 'neuron':
            self.model = nn.Sequential(
                GradScaler(loss_weight),
                nn.Linear(in_features, out_features,bias=False),
                GradScaler(1/loss_weight)
            )
        else:
            self.model = nn.Sequential(
                GradScaler(loss_weight),
                model,
                GradScaler(1/loss_weight)
            )
        
        self.class_list = class_list
        self.train_on_mask_only = train_on_mask_only  # if True, MLP is trained only on the mask and it is frozen when receiceiving the segmenter
        if train_on_mask_only:
            self.frozen_model = copy.deepcopy(self.model)
            #freeze
            for param in self.frozen_model.parameters():
                param.requires_grad = False
            self.frozen_model.eval()
        self.calculate_HU = calculate_HU
        self.mode = mode  # 'attenuation_classifier' or 'tumor_classifier'
        assert self.mode in ['attenuation_classifier', 'tumor_classifier'], f"Invalid mode: {self.mode}. Choose from ['attenuation_classifier', 'tumor_classifier']"
        if self.mode== 'tumor_classifier':
            if model == 'MLP':
                raise ValueError("Tumor classifier must use a different model than MLP.")
    def forward(self, ct, out, labels=None,cut_attenuation_grad=False):
        #ignore the malignant and benign classes in the class list
        out = out[:, :len(self.class_list)]  # (B, C, D, H, W)
        
        if cut_attenuation_grad:
            out = out.detach() #used to train the classifier before propagating the loss to the segmenter
        
        if labels is not None:
            #if the case is annotated by voxel (and has tumor) we train the MLP only, using mask as input
            mask_tumor, mask_organ, indices = get_mask_lesion_masks(labels, self.class_list) 
            from_mask = []
            tmp = []
            for b in list(range(mask_tumor.shape[0])):
                if mask_tumor[b].sum() != 0: #case annotated by voxel
                    tmp.append(labels[b])
                    from_mask.append(1)
                else:
                    #case not annotated by voxel, use the mask from the model
                    tmp.append(out[b])
                    from_mask.append(0)
            out = torch.stack(tmp, dim=0)  # (B, C, D, H, W)
        else:
            from_mask = torch.zeros(ct.shape[0]).int().tolist()
            
        num_lesion_classes = None
        if self.calculate_HU:
            if self.mode != 'attenuation_classifier':
                raise ValueError("calculate_HU is True but mode is not 'attenuation_classifier'.")
            inpt, indices = extract_hu(ct,out,self.class_list)
        else:
            mask_tumor, mask_organ, indices = get_mask_lesion_masks(out, self.class_list)
            num_lesion_classes = mask_tumor.shape[1]  # number of lesion classes in the mask
            if self.mode == 'attenuation_classifier':
                inpt = torch.cat((ct,mask_tumor, mask_organ), dim=1)  # (B, C, D, H, W)
            elif self.mode == 'tumor_classifier':
                inpt = torch.cat((mask_tumor, mask_organ), dim=1)  # (B, C, D, H, W)
                #no CT, we want the masks to be the bottleneck of information.
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Choose from ['attenuation_classifier', 'tumor_classifier']")
            
        if not self.train_on_mask_only:
            att = self.run_model(inpt, indices, frozen=False, num_lesion_classes=num_lesion_classes)  # run MLP on all HU features
        else:
            #if no mask was used, just use the standard MLP
            if np.mean(from_mask)==0:
                att = self.run_model(inpt, indices, frozen=False, num_lesion_classes=num_lesion_classes)
            elif np.mean(from_mask)==1: #only used images with per-voxel annotations
                att = self.run_model(inpt, indices, frozen=True, num_lesion_classes=num_lesion_classes)
                nothing = 0 * self.run_model(inpt[0].unsqueeze(0), indices, frozen=False, num_lesion_classes=num_lesion_classes).sum()
                att = att + nothing #just avoids the unused parameters problem
            else:
                #mixed case: some annotated, some not. Run both.
                #why we need this? 
                #here we want only the samples with masks to be used to train the MLP. Conversely, when training the segmenter, 
                #we only want the segmenter to be updated, not the MLP. This may avoid the segmenter to learn some shortcut solution, 
                #where it passes information to the MLP (about attenuation) without actually improving the tumor segmentation. 
                #But this may also increase the change of overfitting the MLP, as the number of masks may be small.
                frozen_out = self.run_model(inpt, indices, frozen=True, num_lesion_classes=num_lesion_classes) 
                unfrozen_out = self.run_model(inpt, indices, frozen=False, num_lesion_classes=num_lesion_classes)
                chosen = []
                for b in list(range(inpt.shape[0])):
                    if from_mask[b] == 1:
                        chosen.append(frozen_out[b])
                    elif from_mask[b] == 0:
                        chosen.append(unfrozen_out[b])
                    else:
                        raise ValueError('Invalid from_mask value: {}'.format(from_mask[b]))
                att = torch.stack(chosen, dim=0)  # (B, C, 3)
        return att
            
    def run_model(self, inpt, indices, frozen=False,num_lesion_classes=None):
        if not frozen:
            net = self.model
        else:
            for p_froz, p_live in zip(self.frozen_model.parameters(), self.model.parameters()):
                p_froz.data.copy_(p_live.data.detach())#detach may not be needed here. But it is ok to leave it.
            #keep it frozen
            for param in self.frozen_model.parameters():
                param.requires_grad = False
            net = self.frozen_model
            
        if self.calculate_HU:
            att = []
            for c in list(range(inpt.shape[1])):
                x = inpt[:, c, :]  # (B, 8)
                x = net(x)  # (B, 3)
                att.append(x)
            att = torch.stack(att, dim=1)
            att = pad_channels(att,self.class_list, indices)
            return att  # no sigmoid here; use BCEWithLogitsLoss for training
        else:
            x = net(inpt.float())  # (B, C*3)
            if self.mode == 'attenuation_classifier':
                assert num_lesion_classes == int(x.shape[1]/3), f"Number of lesion classes does not match the output of the classifier: expected {num_lesion_classes}, got {int(x.shape[1]/3)}"
                att = x.reshape((x.shape[0],num_lesion_classes,3)) #B,c',3, where c' is the number of lesion classes
                att = pad_channels(att,self.class_list, indices) #B,C,3
            elif self.mode == 'tumor_classifier':
                att = x.reshape((x.shape[0],num_lesion_classes,10,3))#B, lesion classes, tumor, (D,d,Objectness)
                att = pad_channels(att,self.class_list, indices) #B,C,3
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Choose from ['attenuation_classifier', 'tumor_classifier']")
            return att
            
        
        
        
class simple_classifier(torch.nn.Module):
    """
    Uses the difference between the mean tumor HU value of the tumor and the organ
    """
    def __init__(self, class_list, in_features=8, out_features=3):
        super(simple_classifier, self).__init__()
        self.class_list = class_list
        
    def forward(self, ct, out, labels=None,cut_attenuation_grad=False):
        x, indices = extract_hu(ct,out,self.class_list)
        # x[:, :, 4] is the difference in mean HU between tumor and organ
        diff_mean = x[:, :, 4].unsqueeze(-1)
        diff_mean = pad_channels(diff_mean,self.class_list, indices)
        return diff_mean  # shape (B, C, 1)
    
    