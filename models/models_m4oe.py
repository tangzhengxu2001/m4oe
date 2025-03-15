import math
from functools import partial
from typing import Callable
import torch
import torch.jit
import torch.nn as nn
import torch.utils.checkpoint
from timm.layers import Mlp, PatchDropout, PatchEmbed, trunc_normal_
from timm.models._manipulate import checkpoint_seq, named_apply
from my_timm.models.vision_transformer import (Block, BlockMet_all, BlockMet3, _load_weights,
                                               get_init_weights_vit,
                                               init_weights_vit_timm)
from typing import Callable
from transformers import ViTModel
import torch.nn.functional as F

def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    """
    Compute the softmax over specified dimensions, with support for multiple dimensions.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int or tuple[int]): The dimension(s) along which to compute softmax.

    Returns:
        torch.Tensor: Tensor after applying the softmax function.
    """
    # Subtract the max value along the specified dim(s) for numerical stability.
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    # Sum the exponentiated values along the specified dimensions.
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper module to implement a Soft Mixture-of-Experts (MoE) layer.
    
    This module wraps an expert layer (passed as `layer`) to perform a soft routing
    of the input tokens through a set of experts based on learned parameters.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        normalize: bool = True,
        **layer_kwargs,
    ) -> None:
        """
        Initialize the SoftMoELayerWrapper.

        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of expert networks.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): A callable that returns an expert layer instance.
            normalize (bool): Whether to normalize input tokens and learned parameters.
            **layer_kwargs: Additional keyword arguments to pass to the expert layer constructor.
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize the routing tensor phi with shape [dim, num_experts, slots_per_expert].
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        # If normalization is enabled, add a learnable scaling parameter.
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi with LeCun normal initialization.
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)

        # Create a list of expert network modules.
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Soft-MoE layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        # Validate input dimensions.
        assert x.shape[-1] == self.dim, f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert len(x.shape) == 3, f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Optionally normalize the input tokens and phi.
        if self.normalize:
            x = F.normalize(x, dim=2)  # Normalize each token along feature dimension.
            phi = self.scale * F.normalize(phi, dim=0)  # Normalize phi along its first dimension.

        # Compute routing logits using Einstein summation.
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        # Compute dispatch weights (across tokens).
        d = softmax(logits, dim=1)
        # Compute combine weights (across experts and slots).
        c = softmax(logits, dim=(2, 3))

        # Compute expert input slots as a weighted average of the input tokens.
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Pass each slot to the corresponding expert network.
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)],
            dim=1
        )

        # Combine expert outputs using the combine weights.
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y


class SoftMoELayerWrapperMET(nn.Module):
    """
    A variant of SoftMoELayerWrapper that incorporates task-specific embeddings
    and a mutual information loss component into the Soft-MoE layer.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        normalize: bool = True,
        **layer_kwargs,
    ) -> None:
        """
        Initialize the SoftMoELayerWrapperMET.

        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of expert networks.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): A callable to create an expert network.
            normalize (bool): Whether to normalize the input tokens and routing tensor.
            **layer_kwargs: Additional keyword arguments for the expert layer.
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize task-specific embeddings (one per task/slot).
        self.task_embedding1 = nn.Parameter(torch.zeros(dim))
        self.task_embedding2 = nn.Parameter(torch.zeros(dim))
        self.task_embedding3 = nn.Parameter(torch.zeros(dim))
        self.task_embedding4 = nn.Parameter(torch.zeros(dim))
        self.task_embedding5 = nn.Parameter(torch.zeros(dim))
        self.task_embedding6 = nn.Parameter(torch.zeros(dim))
        self.task_embedding7 = nn.Parameter(torch.zeros(dim))

        # Initialize the routing tensor phi.
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization.
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)
        # Initialize the mutual information loss module.
        self.criterion = MutualInformationLoss()

        # Create the list of expert network modules.
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SoftMoELayerWrapperMET.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            tuple: Returns three output tasks and the mutual information loss.
                   - y_task1: Output for task 1.
                   - y_task2: Output for task 2.
                   - y_task3: Output for task 3.
                   - tc_loss: Mutual information loss computed from the routing logits.
        """
        # Validate input dimensions.
        assert x.shape[-1] == self.dim, f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert len(x.shape) == 3, f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Optionally normalize the inputs and phi.
        if self.normalize:
            x = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(phi, dim=0)

        # Compute routing logits using Einstein summation.
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        # Compute mutual information loss based on logits.
        tc_loss = self.criterion(logits)
        # Compute dispatch and combine weights.
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))

        # Compute weighted input slots.
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Prepare task-specific embeddings by unsqueezing dimensions.
        task_emb1 = self.task_embedding1.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        task_emb2 = self.task_embedding2.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        task_emb3 = self.task_embedding3.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Note: Task embeddings for slots 4 to 7 are commented out.
        # Add task-specific embeddings to the corresponding slots.
        xs[:, :, 0, :] = xs[:, :, 0, :] + task_emb1  # First slot
        xs[:, :, 1, :] = xs[:, :, 1, :] + task_emb2  # Second slot
        xs[:, :, 2, :] = xs[:, :, 2, :] + task_emb3  # Third slot

        # Apply each expert to its corresponding slot.
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)],
            dim=1
        )

        # Combine expert outputs using the combine weights.
        y = torch.einsum("bnpd,bmnp->bmpd", ys, c)

        # Extract outputs corresponding to each task.
        y_task1 = y[:, :, 0, :]  # Task 1 output.
        y_task2 = y[:, :, 1, :]  # Task 2 output.
        y_task3 = y[:, :, 2, :]  # Task 3 output.

        return y_task1, y_task2, y_task3, tc_loss


class MutualInformationLoss(nn.Module):
    """
    Compute the mutual information loss to encourage diversity in the routing.

    This loss penalizes low mutual information between the expert routing
    probabilities and the aggregated distributions.
    """

    def __init__(self, epsilon=1e-4):
        """
        Initialize the MutualInformationLoss module.

        Args:
            epsilon (float): A small value added to denominators to avoid division by zero.
        """
        super(MutualInformationLoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent division by zero.

    def check_nan(self, var, var_name: str):
        """
        Check if the given tensor contains NaNs and print a warning.

        Args:
            var (torch.Tensor): Tensor to check.
            var_name (str): Name of the variable for logging purposes.
        """
        if torch.isnan(var).any():
            print(f"NaN detected in {var_name}:\n{var}")

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the mutual information loss based on the routing tensor phi.

        Args:
            phi (torch.Tensor): Routing tensor of shape [batch_size, m, n, p].

        Returns:
            torch.Tensor: A scalar tensor representing the negative mutual information loss.
        """
        batch_size, m, n, p = phi.shape
        # Flatten spatial dimensions for softmax computation.
        phi = phi.reshape(phi.shape[0], phi.shape[1] * phi.shape[2] * phi.shape[3])
        phi = torch.softmax(phi, dim=1)
        phi = phi.reshape(phi.shape[0], m, n, p)
        # Compute marginal distributions.
        p_m = phi.sum(dim=(2, 3))  # Marginal over experts and slots: shape [batch_size, m]
        p_t = phi.sum(dim=(1, 2))  # Marginal over tokens and experts: shape [batch_size, p]
        p_mt = phi.sum(dim=2)      # Joint distribution over tokens and slots: shape [batch_size, m, p]
        # Calculate denominator and numerator for the mutual information formula.
        denumerator = p_m.unsqueeze(2) * p_t.unsqueeze(1)
        numerator = p_mt
        # Compute log term with a small constant added for numerical stability.
        log_term = torch.log(numerator / denumerator + 1e-10)
        # Sum over dimensions to obtain the mutual information.
        mutual_info = torch.sum(p_mt * log_term, dim=(0, 1, 2))
      
        return -mutual_info  # Return negative mutual information as loss.


class SoftMoEVisionTransformer(nn.Module):
    """
    Vision Transformer architecture enhanced with Soft Mixture-of-Experts (MoE) MLP layers.

    This model is modified to process four images concurrently and routes the features
    through a series of expert MLP blocks. It incorporates multiple forward heads for different
    tasks and leverages pre-trained ViT models for initial feature extraction.
    """

    def __init__(
        self,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        moe_layer_index=6,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        num_classes_1: int = 1000,
        num_classes_2: int = 1000,
        num_classes_3: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values=None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm=None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        block_fn2=BlockMet3,
        mlp_layer: Callable = Mlp,
    ):
        """
        Initialize the SoftMoEVisionTransformer.

        Args:
            num_experts (int): Number of experts for the MoE layers.
            slots_per_expert (int): Number of token slots per expert.
            moe_layer_index (int or list[int]): Index (or list of indices) at which to apply MoE layers.
            img_size (int): Input image size.
            patch_size (int): Size of each patch.
            in_chans (int): Number of input channels.
            num_classes (int): Number of classes for classification.
            num_classes_1, num_classes_2, num_classes_3 (int): Number of classes for different tasks.
            global_pool (str): Type of global pooling ('token' or 'avg').
            embed_dim (int): Embedding dimensionality.
            depth (int): Depth (number of transformer blocks).
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio for MLP hidden dimension.
            qkv_bias (bool): Whether to include bias in QKV layers.
            qk_norm (bool): Whether to normalize QK.
            init_values: Initial values for layer scaling.
            class_token (bool): Whether to use a class token.
            no_embed_class (bool): If True, do not add class embedding.
            pre_norm (bool): Whether to apply pre-normalization.
            fc_norm: Normalization for the classifier head.
            drop_rate (float): Dropout rate for the classifier head.
            pos_drop_rate (float): Dropout rate for positional embeddings.
            patch_drop_rate (float): Dropout rate for patch dropout.
            proj_drop_rate (float): Dropout rate for projection layers.
            attn_drop_rate (float): Dropout rate for attention layers.
            drop_path_rate (float): Stochastic depth rate.
            weight_init (str): Weight initialization scheme.
            embed_layer (Callable): Patch embedding layer constructor.
            norm_layer: Normalization layer constructor.
            act_layer: Activation layer constructor.
            block_fn: Transformer block function.
            block_fn2: Alternate transformer block function for MoE.
            mlp_layer (Callable): MLP layer constructor.
        """
        super().__init__()
        # Validate global_pool settings.
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        # Create four separate patch embedding layers (one per image).
        self.patch_embed1 = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        self.patch_embed2 = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        self.patch_embed3 = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        self.patch_embed4 = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )

        # Total number of patches from all four images.
        num_patches = self.patch_embed1.num_patches * 4

        # Initialize class token if enabled.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        # Compute embedding length (including class token if needed).
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        # Initialize positional embeddings.
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        # Apply patch dropout if a rate is specified.
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Set up parameters for the MoE layers.
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert

        # Create partial functions for MoE layers with different configurations.
        moe_mlp_layer = partial(
            SoftMoELayerWrapperMET,
            layer=mlp_layer,
            dim=embed_dim,
            num_experts=self.num_experts,
            slots_per_expert=3,
        )
        moe_mlp_layer1 = partial(
            SoftMoELayerWrapper,
            layer=mlp_layer,
            dim=embed_dim,
            num_experts=32,
            slots_per_expert=1,
        )
        moe_mlp_layer2 = partial(
            SoftMoELayerWrapper,
            layer=mlp_layer,
            dim=embed_dim,
            num_experts=32,
            slots_per_expert=1,
        )
        moe_mlp_layer3 = partial(
            SoftMoELayerWrapper,
            layer=mlp_layer,
            dim=embed_dim,
            num_experts=32,
            slots_per_expert=1,
        )
        moe_mlp_layer4 = partial(
            SoftMoELayerWrapper,
            layer=mlp_layer,
            dim=embed_dim,
            num_experts=32,
            slots_per_expert=1,
        )
        moe_mlp_layer5 = partial(
            SoftMoELayerWrapper,
            layer=mlp_layer,
            dim=embed_dim,
            num_experts=16,
            slots_per_expert=1,
        )
        # Determine which layers should be MoE based on moe_layer_index.
        self.moe_layer_index = moe_layer_index
        if isinstance(moe_layer_index, list):
            # Only use MoE layers at the specified indices.
            assert len(moe_layer_index) > 0
            assert all([0 <= l < depth for l in moe_layer_index])

            mlp_layers_list = [
                moe_mlp_layer if i in moe_layer_index else mlp_layer
                for i in range(1)
            ]
            mlp_layers_list1 = [
                moe_mlp_layer1 if i in moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list2 = [
                moe_mlp_layer2 if i in moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list3 = [
                moe_mlp_layer3 if i in moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list4 = [
                moe_mlp_layer4 if i in moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list5 = [
                moe_mlp_layer5 if i in moe_layer_index else mlp_layer
                for i in range(depth)
            ]
        else:
            # Use MoE layers for all blocks at and after the given index.
            assert 0 <= moe_layer_index < depth
            mlp_layers_list = [
                moe_mlp_layer if i >= moe_layer_index else mlp_layer
                for i in range(1)
            ]
            mlp_layers_list1 = [
                moe_mlp_layer1 if i >= moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list2 = [
                moe_mlp_layer2 if i >= moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list3 = [
                moe_mlp_layer3 if i >= moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list4 = [
                moe_mlp_layer4 if i >= moe_layer_index else mlp_layer
                for i in range(depth)
            ]
            mlp_layers_list5 = [
                moe_mlp_layer5 if i >= moe_layer_index else mlp_layer
                for i in range(depth)
            ]
        # Compute stochastic depth probabilities.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # Create transformer blocks using block_fn2 for the MoE block.
        self.blocks = nn.Sequential(
            *[
                block_fn2(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list[i],
                )
                for i in range(1)
            ]
        )
        # Create additional transformer block sequences.
        self.blocks1 = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list1[i],
                )
                for i in range(depth)
            ]
        )
        self.blocks2 = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list2[i],
                )
                for i in range(depth)
            ]
        )
        self.blocks3 = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list3[i],
                )
                for i in range(depth)
            ]
        )
        self.blocks4 = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list4[i],
                )
                for i in range(depth)
            ]
        )
        self.blocks5 = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layers_list5[i],
                )
                for i in range(depth)
            ]
        )
        # Normalization layers after transformer blocks.
        self.norm1 = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.norm2 = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.norm3 = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier head normalization layers.
        self.fc_norm1 = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.fc_norm2 = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.fc_norm3 = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.fc_norm4 = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.fc_norm5 = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.fc_norm6 = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.fc_norm7 = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        
        self.head_drop = nn.Dropout(drop_rate)
        # Define classifier heads for different tasks.
        self.head1 = nn.Linear(self.embed_dim, num_classes_1) if num_classes > 0 else nn.Identity()
        self.head2 = nn.Linear(self.embed_dim, num_classes_3) if num_classes > 0 else nn.Identity()
        self.head3 = nn.Linear(self.embed_dim, num_classes_3) if num_classes > 0 else nn.Identity()
        self.head7 = nn.Linear(self.embed_dim, num_classes_2) if num_classes > 0 else nn.Identity()
        
        # Initialize weights if specified.
        if weight_init != "skip":
            self.init_weights(weight_init)
        
        # Load pre-trained ViT models for each view.
        self.vit_cc = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_mlo = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_2dcc = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_2dmlo = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        self.depth = depth
        # Projection layers to map ViT outputs to embed_dim.
        self.proj1 = nn.Linear(768, embed_dim)
        self.proj2 = nn.Linear(768, embed_dim)
        self.proj7 = nn.Linear(768, embed_dim)
        # Additional projection layers can be added similarly if needed.

    def init_weights(self, mode=""):
        """
        Initialize weights for the Vision Transformer using the specified scheme.

        Args:
            mode (str): Specifies the initialization mode. Options include "jax", "jax_nlhb", "moco", or "".
        """
        # Set head bias for certain initialization schemes.
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        # Apply weight initialization function to all submodules.
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        """
        Compatibility wrapper for weight initialization.

        Args:
            m: Module to initialize.
        """
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        """
        Load pretrained weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint.
            prefix (str): Optional prefix to match keys in the checkpoint.
        """
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore()
    def no_weight_decay(self):
        """
        Specify parameters that should not be decayed.

        Returns:
            set: Set of parameter names that should not be subject to weight decay.
        """
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def group_matcher(self, coarse=False):
        """
        Define groups for parameter-specific optimization.

        Args:
            coarse (bool): Whether to use coarse grouping.

        Returns:
            dict: A dictionary mapping group names to regex patterns.
        """
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed parameters.
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore()
    def set_grad_checkpointing(self, enable=True):
        """
        Enable or disable gradient checkpointing to save memory.

        Args:
            enable (bool): Whether to enable gradient checkpointing.
        """
        self.grad_checkpointing = enable

    @torch.jit.ignore()
    def get_classifier(self):
        """
        Return the classifier head.

        Returns:
            nn.Module: The classifier head module.
        """
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        """
        Reset the classifier head with a new number of classes.

        Args:
            num_classes (int): New number of classes.
            global_pool (str, optional): Global pooling type; if provided, updates the pooling method.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to the input tokens.

        Args:
            x (torch.Tensor): Input tokens of shape [batch_size, seq_len, embed_dim].

        Returns:
            torch.Tensor: Tokens after adding positional embeddings and applying dropout.
        """
        if self.no_embed_class:
            # For models like deit-3 where positional embedding does not overlap with class token.
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # Standard approach: concatenate class token then add positional embeddings.
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x: tuple) -> tuple:
        """
        Extract features from four input images.

        Args:
            x (tuple): Tuple of four images.

        Returns:
            tuple: A tuple containing task-specific feature outputs and the mutual information loss.
        """
        # Process each image with its corresponding pre-trained ViT model.
        x1 = self.vit_cc(x[0]).last_hidden_state
        x2 = self.vit_mlo(x[1]).last_hidden_state
        x3 = self.vit_2dcc(x[2]).last_hidden_state
        x4 = self.vit_2dmlo(x[3]).last_hidden_state
        
        # Concatenate features from all views along the sequence dimension.
        x_all = torch.cat((x1, x2, x3, x4), dim=1)

        # Apply gradient checkpointing if enabled.
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x_all)
        else:
            # Process each view through its dedicated transformer blocks.
            x1_s = self.blocks1(x1)
            x2_s = self.blocks2(x2)
            x3_s = self.blocks3(x3)
            x4_s = self.blocks4(x4)

            # Concatenate all views for the MoE block.
            x10 = torch.cat([x1, x2, x3, x4], dim=1)
            # Obtain task-specific outputs and mutual information loss.
            x_task1, x_task2, x_task3, tc_loss = self.blocks(x10)
            
            # For each task, concatenate the MoE output with the average-pooled outputs from each view.
            x_task1 = torch.concat([
                x_task1.mean(dim=1, keepdim=True),
                x1_s.mean(dim=1, keepdim=True),
                x2_s.mean(dim=1, keepdim=True),
                x3_s.mean(dim=1, keepdim=True),
                x4_s.mean(dim=1, keepdim=True)
            ], dim=1)
            x_task2 = torch.concat([
                x_task2.mean(dim=1, keepdim=True),
                x1_s.mean(dim=1, keepdim=True),
                x2_s.mean(dim=1, keepdim=True),
                x3_s.mean(dim=1, keepdim=True),
                x4_s.mean(dim=1, keepdim=True)
            ], dim=1)
            x_task3 = torch.concat([
                x_task3.mean(dim=1, keepdim=True),
                x1_s.mean(dim=1, keepdim=True),
                x2_s.mean(dim=1, keepdim=True),
                x3_s.mean(dim=1, keepdim=True),
                x4_s.mean(dim=1, keepdim=True)
            ], dim=1)

            # Further process the concatenated features through additional transformer blocks.
            x_task1 = self.blocks5(x_task1)
            x_task2 = self.blocks5(x_task2)
            x_task3 = self.blocks5(x_task3)
        
        # Normalize the task-specific features.
        x_task1 = self.norm1(x_task1)
        x_task2 = self.norm2(x_task2)
        x_task3 = self.norm3(x_task3)

        return x_task1, x_task2, x_task3, tc_loss

    def forward_head_1(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Forward pass for classifier head 1.

        Args:
            x (torch.Tensor): Input features.
            pre_logits (bool): If True, return features before applying the final linear layer.

        Returns:
            torch.Tensor: Classification output or pre-logits.
        """
        x = self.proj1(x)
        # Apply global pooling.
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm1(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head1(x)

    def forward_head_2(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Forward pass for classifier head 2.

        Args:
            x (torch.Tensor): Input features.
            pre_logits (bool): If True, return features before applying the final linear layer.

        Returns:
            torch.Tensor: Classification output or pre-logits.
        """
        x = self.proj2(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm2(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head2(x)

    def forward_head_3(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Forward pass for classifier head 3.

        Args:
            x (torch.Tensor): Input features.
            pre_logits (bool): If True, return features before applying the final linear layer.

        Returns:
            torch.Tensor: Classification output or pre-logits.
        """
        # Assuming a projection layer 'proj3' exists; if not, modify accordingly.
        x = self.proj3(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm3(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head3(x)

    def forward_head_4(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Forward pass for classifier head 4.

        Args:
            x (torch.Tensor): Input features.
            pre_logits (bool): If True, return features before applying the final linear layer.

        Returns:
            torch.Tensor: Classification output or pre-logits.
        """
        x = self.proj4(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm4(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head4(x)

    def forward_head_5(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Forward pass for classifier head 5.

        Args:
            x (torch.Tensor): Input features.
            pre_logits (bool): If True, return features before applying the final linear layer.

        Returns:
            torch.Tensor: Classification output or pre-logits.
        """
        x = self.proj5(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm5(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head5(x)

    def forward_head_6(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Forward pass for classifier head 6.

        Args:
            x (torch.Tensor): Input features.
            pre_logits (bool): If True, return features before applying the final linear layer.

        Returns:
            torch.Tensor: Classification output or pre-logits.
        """
        x = self.proj6(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm6(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head6(x)

    def forward_head_7(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Forward pass for classifier head 7.

        Args:
            x (torch.Tensor): Input features.
            pre_logits (bool): If True, return features before applying the final linear layer.

        Returns:
            torch.Tensor: Classification output or pre-logits.
        """
        x = self.proj7(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm7(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head7(x)

    def forward(self, x: tuple) -> tuple:
        """
        Forward pass for the entire Vision Transformer.

        Args:
            x (tuple): Tuple containing four images.

        Returns:
            tuple: A tuple containing outputs from classifier heads for task 1, task 2, task 3, and the mutual information loss.
        """
        # Extract features for each task.
        x_task1, x_task2, x_task3, tc_loss = self.forward_features(x)
        # Obtain classifier outputs for each task.
        x_task1 = self.forward_head_1(x_task1)
        x_task2 = self.forward_head_2(x_task2)
        x_task3 = self.forward_head_7(x_task3)
        return x_task1, x_task2, x_task3, tc_loss
