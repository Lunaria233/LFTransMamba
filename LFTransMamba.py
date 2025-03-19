import math
import time
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from mamba import selective_scan_cuda_core_forward
from mamba import selective_scan_cuda_core_backward


__SIZE__ = 32


class TransDWConv(nn.Module):
  def __init__(self, dim=64):
    super(TransDWConv, self).__init__()
    self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

  def forward(self, x):
    B, N, C = x.shape
    x = x.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
    x = self.dwconv(x)
    x = x.flatten(2).transpose(1, 2)
    return x


class TransMlp(nn.Module):
  """ MLP as used in Vision Transformer, MLP-Mixer and related networks
  """

  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    self.fc1 = nn.Linear(in_features, hidden_features)
    self.dwconv = TransDWConv(hidden_features)
    self.act = act_layer()
    self.drop1 = nn.Dropout(drop)
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop2 = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    x = self.dwconv(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


class TransAttention(nn.Module):
  def __init__(self,
               dim,
               num_heads=8,
               qkv_bias=False,
               attn_drop=0.,
               proj_drop=0.,
               sr_ratio=1,
               use_flashatten=False):
    super().__init__()
    assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim ** -0.5
    self.use_flashatten = use_flashatten

    self.q = nn.Linear(dim, dim, bias=qkv_bias)
    self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

    self.sr_ratio = sr_ratio
    if sr_ratio > 1:
      self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
      self.norm = nn.LayerNorm(dim)

  def forward(self, x):
    B, N, C = x.shape
    q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    if self.sr_ratio > 1:
      x_ = x.permute(0, 2, 1).reshape(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
      x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
      x_ = self.norm(x_)
      kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    else:
      kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    k, v = kv[0], kv[1]

    if self.use_flashatten:
      x = F.scaled_dot_product_attention(query=q, key=k, value=v, scale=self.scale)
      x = x.transpose(1, 2).reshape(B, N, C)

    else:
      attn = (q @ k.transpose(-2, -1)) * self.scale
      attn = attn.softmax(dim=-1)
      attn = self.attn_drop(attn)
      x = (attn @ v).transpose(1, 2).reshape(B, N, C)

    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class TransBlock(nn.Module):
  def __init__(self,
               dim,
               num_heads,
               mlp_ratio=4.,
               qkv_bias=False,
               drop=0.,
               attn_drop=0.,
               drop_path=0.,
               sr_ratio=1,
               act_layer=nn.GELU,
               norm_layer=nn.LayerNorm,
               use_flashatten=False):
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = TransAttention(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        attn_drop=attn_drop,
        proj_drop=drop,
        sr_ratio=sr_ratio,
        use_flashatten=use_flashatten)
    # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = TransMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

  def forward(self, x):
    b, h, w, c = x.shape
    x = rearrange(x, 'b h w c-> b (h w) c')
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    x = rearrange(x, 'b (h w) c-> b h w c', h=h, w=w)
    return x


# 这些函数有什么区别吗？小狼挠头

class CrossScan(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor):
    B, C, H, W = x.shape
    ctx.shape = (B, C, H, W)
    xs = x.new_empty((B, 4, C, H * W))
    xs[:, 0] = x.flatten(2, 3)
    xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
    return xs

  @staticmethod
  def backward(ctx, ys: torch.Tensor):
    # out: (b, k, d, l)
    B, C, H, W = ctx.shape
    L = H * W
    ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
    y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
    return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
  @staticmethod
  def forward(ctx, ys: torch.Tensor):
    B, K, D, H, W = ys.shape
    ctx.shape = (H, W)
    ys = ys.view(B, K, D, -1)
    ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
    y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
    return y

  @staticmethod
  def backward(ctx, x: torch.Tensor):
    # B, D, L = x.shape
    # out: (b, k, d, l)
    H, W = ctx.shape
    B, C, L = x.shape
    xs = x.new_empty((B, 4, C, L))
    xs[:, 0] = x
    xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
    xs = xs.view(B, 4, C, H, W)
    return xs, None, None


class CrossScan3D(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor):

    # 后续三个函数的标准

    B, C, H, W = x.shape
    ctx.shape = (B, C, H, W)
    xs = x.new_empty((B, 4, C, H * W))

    # 输入形态 b c (u h) (v w)
    # 第一行变化为: b c (h w) (u v) 这样变化就是 空间横向(光场横向)
    # 第二行变化为: b c (w h) (v u) 这样变化就是 空间纵向(光场纵向)

    xs[:, 0] = rearrange(x, 'b c (h u) (w v) -> b c (h w) (u v)', b=B, c=C, u=5,
                         v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 1] = rearrange(x, 'b c (h u) (w v) -> b c (w h) (v u)', b=B, c=C, u=5,
                         v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

    return xs

  @staticmethod
  def backward(ctx, ys: torch.Tensor):

    # out: (b, k, d, l)

    # 输入形态等于该函数forward输出形态，输出应该是forward输入形态
    B, C, H, W = ctx.shape
    L = H * W

    ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
    y = rearrange(ys[:, 0].view(B, -1, L // 25, 5 * 5), 'b c (h w) (u v) -> b c (h u) (w v)',
                  b=B, c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous()
    y += rearrange(ys[:, 1].view(B, -1, L // 25, 5 * 5), 'b c (w h) (v u) -> b c (h u) (w v)',
                   b=B, c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous()

    return y.view(B, -1, H, W)


class CrossMerge3D(torch.autograd.Function):
  @staticmethod
  def forward(ctx, ys: torch.Tensor):

    # 后面没有形态变换函数，所以这里要输出能够变成原本CrossScan3D的输入形态
    B, K, D, H, W = ys.shape
    ctx.shape = (H, W)
    ys = ys.view(B, K, D, -1)

    ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
    y = rearrange(ys[:, 0].view(B, -1, H * W // 25, 5 * 5), 'b c (h w) (u v) -> b c (h u) (w v)',
                  b=B, c=D, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    y += rearrange(ys[:, 1].view(B, -1, H * W // 25, 5 * 5), 'b c (w h) (v u) -> b c (h u) (w v)',
                   b=B, c=D, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)

    return y

  @staticmethod
  def backward(ctx, x: torch.Tensor):
    # B, D, L = x.shape
    # out: (b, k, d, l)

    # 梯度回传的是CrossScan3D输入形态，这里需要改成CrossScan3D的输出形态
    H, W = ctx.shape
    B, C, L = x.shape
    xs = x.new_empty((B, 4, C, L))

    xs[:, 0] = rearrange(x.view(B, -1, H, W), 'b c (h u) (w v) -> b c (h w) (u v)', b=B,
                         c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 1] = rearrange(x.view(B, -1, H, W), 'b c (h u) (w v) -> b c (w h) (v u)', b=B,
                         c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
    xs = xs.view(B, 4, C, H, W)
    return xs, None, None


class CrossScan4D(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor):

    B, C, H, W = x.shape
    ctx.shape = (B, C, H, W)
    xs = x.new_empty((B, 4, C, H * W))

    xs[:, 0] = rearrange(x, 'b c (u h) (v w) -> b c (u v) (h w)', b=B, c=C, u=5,
                         v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 1] = rearrange(x, 'b c (u h) (v w) -> b c (v u) (w h)', b=B, c=C, u=5,
                         v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

    return xs

  @staticmethod
  def backward(ctx, ys: torch.Tensor):

    # out: (b, k, d, l)

    B, C, H, W = ctx.shape
    L = H * W

    ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
    y = rearrange(ys[:, 0].view(B, -1, 5 * 5, L // 25), 'b c (u v) (h w) -> b c (u h) (v w)',
                  b=B, c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous()
    y += rearrange(ys[:, 1].view(B, -1, 5 * 5, L // 25), 'b c (v u) (w h) -> b c (u h) (v w)',
                   b=B, c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous()

    return y.view(B, -1, H, W)


class CrossMerge4D(torch.autograd.Function):
  @staticmethod
  def forward(ctx, ys: torch.Tensor):

    B, K, D, H, W = ys.shape
    ctx.shape = (H, W)
    ys = ys.view(B, K, D, -1)

    ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
    y = rearrange(ys[:, 0].view(B, -1, 5 * 5, H * W // 25), 'b c (u v) (h w) -> b c (u h) (v w)',
                  b=B, c=D, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    y += rearrange(ys[:, 1].view(B, -1, 5 * 5, H * W // 25), 'b c (v u) (w h) -> b c (u h) (v w)',
                   b=B, c=D, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)

    return y

  @staticmethod
  def backward(ctx, x: torch.Tensor):
    # B, D, L = x.shape
    # out: (b, k, d, l)

    H, W = ctx.shape
    B, C, L = x.shape
    xs = x.new_empty((B, 4, C, L))

    xs[:, 0] = rearrange(x.view(B, -1, H, W), 'b c (u h) (v w) -> b c (u v) (h w)', b=B,
                         c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 1] = rearrange(x.view(B, -1, H, W), 'b c (u h) (v w) -> b c (v u) (w h)', b=B,
                         c=C, u=5, v=5, h=H // 5, w=W // 5).contiguous().flatten(2, 3)
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
    xs = xs.view(B, 4, C, H, W)

    return xs, None, None


def cross_selective_scan(
    x: torch.Tensor = None,
    x_proj_weight: torch.Tensor = None,
    x_proj_bias: torch.Tensor = None,
    dt_projs_weight: torch.Tensor = None,
    dt_projs_bias: torch.Tensor = None,
    A_logs: torch.Tensor = None,
    Ds: torch.Tensor = None,
    out_norm: torch.nn.Module = None,
    out_norm_shape="v0",
    nrows=-1,  # for SelectiveScanNRow
    backnrows=-1,  # for SelectiveScanNRow
    delta_softplus=True,
    to_dtype=True,
    force_fp32=False,  # False if ssoflex
    ssoflex=True,
    SelectiveScan=None,
):
  # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

  B, D, H, W = x.shape
  D, N = A_logs.shape
  K, D, R = dt_projs_weight.shape
  L = H * W

  if SelectiveScan == SelectiveScanNRow:
    if nrows < 1:
      if D % 4 == 0:
        nrows = 4
      elif D % 3 == 0:
        nrows = 3
      elif D % 2 == 0:
        nrows = 2
      else:
        nrows = 1

    if backnrows < 1:
      if D % 4 == 0:
        backnrows = 4
      elif D % 3 == 0:
        backnrows = 3
      elif D % 2 == 0:
        backnrows = 2
      else:
        backnrows = 1

  def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

  xs = CrossScan.apply(x)
  x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
  if x_proj_bias is not None:
    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
  dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
  dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
  xs = xs.view(B, -1, L)
  dts = dts.contiguous().view(B, -1, L)
  As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
  Bs = Bs.contiguous()
  Cs = Cs.contiguous()
  Ds = Ds.to(torch.float)  # (K * c)
  delta_bias = dt_projs_bias.view(-1).to(torch.float)

  if force_fp32:
    xs = xs.to(torch.float)
    dts = dts.to(torch.float)
    Bs = Bs.to(torch.float)
    Cs = Cs.to(torch.float)

  ys: torch.Tensor = selective_scan(
      xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
  ).view(B, K, -1, H, W)

  y: torch.Tensor = CrossMerge.apply(ys)

  if out_norm_shape in ["v1"]:  # (B, C, H, W)
    y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
  else:  # (B, L, C)
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

  return (y.to(x.dtype) if to_dtype else y)


def cross_selective_scan3D(
    x: torch.Tensor = None,
    x_proj_weight: torch.Tensor = None,
    x_proj_bias: torch.Tensor = None,
    dt_projs_weight: torch.Tensor = None,
    dt_projs_bias: torch.Tensor = None,
    A_logs: torch.Tensor = None,
    Ds: torch.Tensor = None,
    out_norm: torch.nn.Module = None,
    out_norm_shape="v0",
    nrows=-1,  # for SelectiveScanNRow
    backnrows=-1,  # for SelectiveScanNRow
    delta_softplus=True,
    to_dtype=True,
    force_fp32=False,  # False if ssoflex
    ssoflex=True,
    SelectiveScan=None,
):
  # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

  B, D, H, W = x.shape
  D, N = A_logs.shape
  K, D, R = dt_projs_weight.shape
  L = H * W

  if SelectiveScan == SelectiveScanNRow:
    if nrows < 1:
      if D % 4 == 0:
        nrows = 4
      elif D % 3 == 0:
        nrows = 3
      elif D % 2 == 0:
        nrows = 2
      else:
        nrows = 1

    if backnrows < 1:
      if D % 4 == 0:
        backnrows = 4
      elif D % 3 == 0:
        backnrows = 3
      elif D % 2 == 0:
        backnrows = 2
      else:
        backnrows = 1

  def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

  xs = CrossScan3D.apply(x)
  x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
  if x_proj_bias is not None:
    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
  dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
  dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
  xs = xs.view(B, -1, L)
  dts = dts.contiguous().view(B, -1, L)
  As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
  Bs = Bs.contiguous()
  Cs = Cs.contiguous()
  Ds = Ds.to(torch.float)  # (K * c)
  delta_bias = dt_projs_bias.view(-1).to(torch.float)

  if force_fp32:
    xs = xs.to(torch.float)
    dts = dts.to(torch.float)
    Bs = Bs.to(torch.float)
    Cs = Cs.to(torch.float)

  ys: torch.Tensor = selective_scan(
      xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
  ).view(B, K, -1, H, W)

  y: torch.Tensor = CrossMerge3D.apply(ys)

  if out_norm_shape in ["v1"]:  # (B, C, H, W)
    y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
  else:  # (B, L, C)
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

  return (y.to(x.dtype) if to_dtype else y)


def cross_selective_scan4D(
    x: torch.Tensor = None,
    x_proj_weight: torch.Tensor = None,
    x_proj_bias: torch.Tensor = None,
    dt_projs_weight: torch.Tensor = None,
    dt_projs_bias: torch.Tensor = None,
    A_logs: torch.Tensor = None,
    Ds: torch.Tensor = None,
    out_norm: torch.nn.Module = None,
    out_norm_shape="v0",
    nrows=-1,  # for SelectiveScanNRow
    backnrows=-1,  # for SelectiveScanNRow
    delta_softplus=True,
    to_dtype=True,
    force_fp32=False,  # False if ssoflex
    ssoflex=True,
    SelectiveScan=None,
):
  # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

  B, D, H, W = x.shape
  D, N = A_logs.shape
  K, D, R = dt_projs_weight.shape
  L = H * W

  if SelectiveScan == SelectiveScanNRow:
    if nrows < 1:
      if D % 4 == 0:
        nrows = 4
      elif D % 3 == 0:
        nrows = 3
      elif D % 2 == 0:
        nrows = 2
      else:
        nrows = 1

    if backnrows < 1:
      if D % 4 == 0:
        backnrows = 4
      elif D % 3 == 0:
        backnrows = 3
      elif D % 2 == 0:
        backnrows = 2
      else:
        backnrows = 1

  def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

  xs = CrossScan4D.apply(x)
  x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
  if x_proj_bias is not None:
    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
  dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
  dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
  xs = xs.view(B, -1, L)
  dts = dts.contiguous().view(B, -1, L)
  As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
  Bs = Bs.contiguous()
  Cs = Cs.contiguous()
  Ds = Ds.to(torch.float)  # (K * c)
  delta_bias = dt_projs_bias.view(-1).to(torch.float)

  if force_fp32:
    xs = xs.to(torch.float)
    dts = dts.to(torch.float)
    Bs = Bs.to(torch.float)
    Cs = Cs.to(torch.float)

  ys: torch.Tensor = selective_scan(
      xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
  ).view(B, K, -1, H, W)

  y: torch.Tensor = CrossMerge4D.apply(ys)

  if out_norm_shape in ["v1"]:  # (B, C, H, W)
    y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
  else:  # (B, L, C)
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

  return (y.to(x.dtype) if to_dtype else y)


def cross_selective_scanv2(
    x: torch.Tensor = None,
    x_proj_weight: torch.Tensor = None,
    x_proj_bias: torch.Tensor = None,
    dt_projs_weight: torch.Tensor = None,
    dt_projs_bias: torch.Tensor = None,
    A_logs: torch.Tensor = None,
    Ds: torch.Tensor = None,
    out_norm: torch.nn.Module = None,
    out_norm_shape="v0",
    nrows=-1,  # for SelectiveScanNRow
    backnrows=-1,  # for SelectiveScanNRow
    delta_softplus=True,
    to_dtype=True,
    force_fp32=False,  # False if ssoflex
    ssoflex=True,
    SelectiveScan=None,
):
  # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

  B, D, H, W = x.shape
  D, N = A_logs.shape
  K, D, R = dt_projs_weight.shape
  L = H * W

  if SelectiveScan == SelectiveScanNRow:
    if nrows < 1:
      if D % 4 == 0:
        nrows = 4
      elif D % 3 == 0:
        nrows = 3
      elif D % 2 == 0:
        nrows = 2
      else:
        nrows = 1

    if backnrows < 1:
      if D % 4 == 0:
        backnrows = 4
      elif D % 3 == 0:
        backnrows = 3
      elif D % 2 == 0:
        backnrows = 2
      else:
        backnrows = 1

  def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

  # =========================
  # tmp
  x_proj_weight = x_proj_weight.sum(dim=0)
  x_proj_bias = x_proj_bias.view(K, -1, 1).sum(dim=0) if x_proj_bias is not None else None
  dt_projs_weight = dt_projs_weight.sum(dim=0)
  # =========================
  x = x.view(B, D, L)
  x_dbl = torch.einsum("b d l, c d -> b c l", x, x_proj_weight)
  if x_proj_bias is not None:
    x_dbl = x_dbl + x_proj_bias.view(1, -1, 1)
  dts = torch.einsum("b r l, d r -> b d l", x_dbl[:, :R, :], dt_projs_weight)
  ps = torch.cat([x, dts, x_dbl[:, R:, :]], dim=1).view(B, -1, H, W)  # (B, D+D+N+N, L)
  ps = CrossScan.apply(ps)  # (B, 4, D+D+N+N, L)

  xs, dts, Bs, Cs = torch.split(ps, [D, D, N, N], dim=2)

  xs = xs.contiguous().view(B, -1, L)
  dts = dts.contiguous().view(B, -1, L)
  As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
  Bs = Bs.contiguous()
  Cs = Cs.contiguous()
  Ds = Ds.to(torch.float)  # (K * c)
  delta_bias = dt_projs_bias.view(-1).to(torch.float)

  if force_fp32:
    xs = xs.to(torch.float)
    dts = dts.to(torch.float)
    Bs = Bs.to(torch.float)
    Cs = Cs.to(torch.float)

  ys: torch.Tensor = selective_scan(
      xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
  ).view(B, K, -1, H, W)

  y: torch.Tensor = CrossMerge.apply(ys)

  if out_norm_shape in ["v1"]:  # (B, C, H, W)
    y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
  else:  # (B, L, C)
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

  return (y.to(x.dtype) if to_dtype else y)


class SelectiveScanFake(torch.autograd.Function):
  # comment all checks if inside cross_selective_scan
  @staticmethod
  @torch.amp.custom_fwd(device_type='cuda')
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
    ctx.delta_softplus = delta_softplus
    ctx.backnrows = backnrows
    x = delta
    out = u
    ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
    return out

  @staticmethod
  @torch.amp.custom_bwd(device_type='cuda')
  def backward(ctx, dout, *args):
    u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
    if dout.stride(-1) != 1:
      dout = dout.contiguous()
    du, ddelta, dA, dB, dC, dD, ddelta_bias = u * 0, delta * 0, A * 0, B * 0, C * \
        0, C * 0, (D * 0 if D else None), (delta_bias * 0 if delta_bias else None)
    return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
  # comment all checks if inside cross_selective_scan
  @staticmethod
  @torch.amp.custom_fwd(device_type='cuda')
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
    ctx.delta_softplus = delta_softplus
    ctx.backnrows = backnrows
    out, x, *rest = selective_scan_cuda_core_forward(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
    ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
    return out

  @staticmethod
  @torch.amp.custom_bwd(device_type='cuda')
  def backward(ctx, dout, *args):
    u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
    if dout.stride(-1) != 1:
      dout = dout.contiguous()
    du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core_backward(
        u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
    )
    return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanNRow(torch.autograd.Function):
  # comment all checks if inside cross_selective_scan
  @staticmethod
  @torch.amp.custom_fwd(device_type='cuda')
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
    ctx.delta_softplus = delta_softplus
    ctx.backnrows = backnrows
    out, x, *rest = selective_scan_cuda_nrow.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
    return out

  @staticmethod
  @torch.amp.custom_bwd(device_type='cuda')
  def backward(ctx, dout, *args):
    u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
    if dout.stride(-1) != 1:
      dout = dout.contiguous()
    du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_nrow.bwd(
        u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.backnrows
    )
    return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
  # comment all checks if inside cross_selective_scan
  @staticmethod
  @torch.amp.custom_fwd(device_type='cuda')
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
    ctx.delta_softplus = delta_softplus
    ctx.backnrows = backnrows
    out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, oflex)
    ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
    return out

  @staticmethod
  @torch.amp.custom_bwd(device_type='cuda')
  def backward(ctx, dout, *args):
    u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
    if dout.stride(-1) != 1:
      dout = dout.contiguous()
    du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
        u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.backnrows
    )
    return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanMamba(torch.autograd.Function):
  # comment all checks if inside cross_selective_scan
  @staticmethod
  @torch.amp.custom_fwd(device_type='cuda')
  def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
    # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
    # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
    ctx.delta_softplus = delta_softplus
    ctx.backnrows = backnrows

    out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
    ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
    return out

  @staticmethod
  @torch.amp.custom_bwd(device_type='cuda')
  def backward(ctx, dout, *args):
    u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
    if dout.stride(-1) != 1:
      dout = dout.contiguous()

    du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
        u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
        False
    )

    return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None,
               act_layer=nn.GELU, drop=0., channels_first=False):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
    self.fc1 = Linear(in_features, hidden_features)
    self.act = act_layer()
    self.fc2 = Linear(hidden_features, out_features)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x


class SS2D(nn.Module):
  def __init__(
      self,
      # basic dims ===========
      d_model=96,
      d_state=16,
      ssm_ratio=2.0,
      ssm_rank_ratio=2.0,
      dt_rank="auto",
      act_layer=nn.SiLU,
      # dwconv ===============
      d_conv=3,  # < 2 means no conv
      conv_bias=True,
      # ======================
      dropout=0.0,
      bias=False,
      # dt init ==============
      dt_min=0.001,
      dt_max=0.1,
      dt_init="random",
      dt_scale=1.0,
      dt_init_floor=1e-4,
      initialize="v0",
      # ======================
      forward_type="v2",
      # ======================
      **kwargs,
  ):
    """
    ssm_rank_ratio would be used in the future...
    """
    factory_kwargs = {"device": None, "dtype": None}
    super().__init__()
    d_expand = int(ssm_ratio * d_model)
    d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
    self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
    self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
    self.d_conv = d_conv

    # tags for forward_type ==============================
    def checkpostfix(tag, value):
      ret = value[-len(tag):] == tag
      if ret:
        value = value[:-len(tag)]
      return ret, value

    self.disable_force32, forward_type = checkpostfix("no32", forward_type)
    self.disable_z, forward_type = checkpostfix("noz", forward_type)
    self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

    # softmax | sigmoid | dwconv | norm ===========================
    if forward_type[-len("none"):] == "none":
      forward_type = forward_type[:-len("none")]
      self.out_norm = nn.Identity()
    elif forward_type[-len("dwconv3"):] == "dwconv3":
      forward_type = forward_type[:-len("dwconv3")]
      self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
      self.out_norm_shape = "v1"
    elif forward_type[-len("softmax"):] == "softmax":
      forward_type = forward_type[:-len("softmax")]
      self.out_norm = nn.Softmax(dim=1)
    elif forward_type[-len("sigmoid"):] == "sigmoid":
      forward_type = forward_type[:-len("sigmoid")]
      self.out_norm = nn.Sigmoid()
    else:
      self.out_norm = nn.LayerNorm(d_inner)

    # forward_type debug =======================================
    FORWARD_TYPES = dict(
        v0=self.forward_corev0,
        fake=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanFake),
        v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        v2_3d=partial(self.forward_corev2_3d, force_fp32=None, SelectiveScan=SelectiveScanCore),
        v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
        v4=partial(
            self.forward_corev2,
            force_fp32=False,
            SelectiveScan=SelectiveScanOflex,
            cross_selective_scan=cross_selective_scanv2),
        v1=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanOflex),
        v01=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanMamba),
        share_ssm=self.forward_corev0_share_ssm,
        share_a=self.forward_corev0_share_a,
    )
    if forward_type.startswith("debug"):
      from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations
      FORWARD_TYPES.update(dict(
          debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
          debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
          debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16, self),
          debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs, self),
          debugforward_core_mambassm_fusecscm=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
          debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm, self),
          debugforward_core_sscore_fusecscm_fwdnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_fwdnrow, self),
          debugforward_core_sscore_fusecscm_bwdnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_bwdnrow, self),
          debugforward_core_sscore_fusecscm_fbnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_fbnrow, self),
          debugforward_core_ssoflex_fusecscm=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm, self),
          debugforward_core_ssoflex_fusecscm_i16o32=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
      ))
    self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))
    self.K = 4 if forward_type not in ["share_ssm"] else 1
    self.K2 = self.K if forward_type not in ["share_a"] else 1

    # in proj =======================================
    d_proj = d_expand if self.disable_z else (d_expand * 2)
    self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
    self.act: nn.Module = act_layer()

    # conv =======================================
    if self.d_conv > 1:
      self.conv2d = nn.Conv2d(
          in_channels=d_expand,
          out_channels=d_expand,
          groups=d_expand,
          bias=conv_bias,
          kernel_size=d_conv,
          padding=(d_conv - 1) // 2,
          **factory_kwargs,
      )

    # rank ratio =====================================
    self.ssm_low_rank = False
    if d_inner < d_expand:
      self.ssm_low_rank = True
      self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
      self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

    # x proj ============================
    self.x_proj = [
        nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        for _ in range(self.K)
    ]
    self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
    del self.x_proj

    # out proj =======================================
    self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
    self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    if initialize in ["v0"]:
      # dt proj ============================
      self.dt_projs = [
          self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
          for _ in range(self.K)
      ]
      self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
      self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
      del self.dt_projs

      # A, D =======================================
      self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
      self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)
    elif initialize in ["v1"]:
      # simple init dt_projs, A_logs, Ds
      self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
      self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))
                                 )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
      self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
      self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))
    elif initialize in ["v2"]:
      # simple init dt_projs, A_logs, Ds
      self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
      self.A_logs = nn.Parameter(torch.zeros((self.K2 * d_inner, self.d_state))
                                 )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
      self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
      self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

  @staticmethod
  def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001,
              dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
    dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
      nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
      nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    else:
      raise NotImplementedError

    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    dt = torch.exp(
        torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
      dt_proj.bias.copy_(inv_dt)
    # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
    # dt_proj.bias._no_reinit = True

    return dt_proj

  @staticmethod
  def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
    # S4D real initialization
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
        "n -> d n",
        d=d_inner,
    ).contiguous()
    A_log = torch.log(A)  # Keep A_log in fp32
    if copies > 0:
      A_log = repeat(A_log, "d n -> r d n", r=copies)
      if merge:
        A_log = A_log.flatten(0, 1)
    A_log = nn.Parameter(A_log)
    A_log._no_weight_decay = True
    return A_log

  @staticmethod
  def D_init(d_inner, copies=-1, device=None, merge=True):
    # D "skip" parameter
    D = torch.ones(d_inner, device=device)
    if copies > 0:
      D = repeat(D, "n1 -> r n1", r=copies)
      if merge:
        D = D.flatten(0, 1)
    D = nn.Parameter(D)  # Keep in fp32
    D._no_weight_decay = True
    return D

  # only used to run previous version
  def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
      return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    B, C, H, W = x.shape
    L = H * W
    K = 4

    x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2,
                         dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
    xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
    # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

    xs = xs.float().view(B, -1, L)  # (b, k * d, l)
    dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
    Bs = Bs.float()  # (b, k, d_state, l)
    Cs = Cs.float()  # (b, k, d_state, l)

    As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
    Ds = self.Ds.float()  # (k * d)
    dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

    # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
    # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

    out_y = selective_scan(
        xs, dts,
        As, Bs, Cs, Ds,
        delta_bias=dt_projs_bias,
        delta_softplus=True,
    ).view(B, K, -1, L)
    # assert out_y.dtype == torch.float

    inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
    wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = self.out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

  def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
    """
    we may conduct this ablation later, but not with v0.
    """
    ...

  def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
    """
    we may conduct this ablation later, but not with v0.
    """
    ...

  def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                     cross_selective_scan=cross_selective_scan, force_fp32=None):
    force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    if self.ssm_low_rank:
      x = self.in_rank(x)
    x = cross_selective_scan(
        x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
        self.A_logs, self.Ds,
        out_norm=getattr(self, "out_norm", None),
        out_norm_shape=getattr(self, "out_norm_shape", "v0"),
        delta_softplus=True, force_fp32=force_fp32,
        SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
    )
    if self.ssm_low_rank:
      x = self.out_rank(x)
    return x

  def forward_corev2_3d(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                        cross_selective_scan=cross_selective_scan, force_fp32=None):
    force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    if self.ssm_low_rank:
      x = self.in_rank(x)
    x = cross_selective_scan3D(
        x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
        self.A_logs, self.Ds,
        out_norm=getattr(self, "out_norm", None),
        out_norm_shape=getattr(self, "out_norm_shape", "v0"),
        delta_softplus=True, force_fp32=force_fp32,
        SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
    )
    if self.ssm_low_rank:
      x = self.out_rank(x)
    return x

  def forward(self, x: torch.Tensor, **kwargs):

    x = self.in_proj(x)
    if not self.disable_z:
      x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
      if not self.disable_z_act:
        z = self.act(z)
    if self.d_conv > 0:
      x = x.permute(0, 3, 1, 2).contiguous()
      x = self.conv2d(x)  # (b, d, h, w)
    x = self.act(x)
    y = self.forward_core(x, channel_first=(self.d_conv > 1))
    if not self.disable_z:
      y = y * z
    out = self.dropout(self.out_proj(y))
    return out


class SS2D3D(nn.Module):
  def __init__(
      self,
      # basic dims ===========
      d_model=96,
      d_state=16,
      ssm_ratio=2.0,
      ssm_rank_ratio=2.0,
      dt_rank="auto",
      act_layer=nn.SiLU,
      # dwconv ===============
      d_conv=3,  # < 2 means no conv
      conv_bias=True,
      # ======================
      dropout=0.0,
      bias=False,
      # dt init ==============
      dt_min=0.001,
      dt_max=0.1,
      dt_init="random",
      dt_scale=1.0,
      dt_init_floor=1e-4,
      initialize="v0",
      # ======================
      forward_type="v2",
      # ======================
      **kwargs,
  ):
    """
    ssm_rank_ratio would be used in the future...
    """
    factory_kwargs = {"device": None, "dtype": None}
    super().__init__()
    d_expand = int(ssm_ratio * d_model)
    d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
    self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
    self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
    self.d_conv = d_conv

    # tags for forward_type ==============================
    def checkpostfix(tag, value):
      ret = value[-len(tag):] == tag
      if ret:
        value = value[:-len(tag)]
      return ret, value

    self.disable_force32, forward_type = checkpostfix("no32", forward_type)
    self.disable_z, forward_type = checkpostfix("noz", forward_type)
    self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

    # softmax | sigmoid | dwconv | norm ===========================
    if forward_type[-len("none"):] == "none":
      forward_type = forward_type[:-len("none")]
      self.out_norm = nn.Identity()
    elif forward_type[-len("dwconv3"):] == "dwconv3":
      forward_type = forward_type[:-len("dwconv3")]
      self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
      self.out_norm_shape = "v1"
    elif forward_type[-len("softmax"):] == "softmax":
      forward_type = forward_type[:-len("softmax")]
      self.out_norm = nn.Softmax(dim=1)
    elif forward_type[-len("sigmoid"):] == "sigmoid":
      forward_type = forward_type[:-len("sigmoid")]
      self.out_norm = nn.Sigmoid()
    else:
      self.out_norm = nn.LayerNorm(d_inner)

    # forward_type debug =======================================
    FORWARD_TYPES = dict(
        v0=self.forward_corev0,
        fake=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanFake),
        v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        v2_3d=partial(self.forward_corev2_3d, force_fp32=None, SelectiveScan=SelectiveScanCore),
        v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
        v4=partial(
            self.forward_corev2,
            force_fp32=False,
            SelectiveScan=SelectiveScanOflex,
            cross_selective_scan=cross_selective_scanv2),
        v1=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanOflex),
        v01=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanMamba),
        share_ssm=self.forward_corev0_share_ssm,
        share_a=self.forward_corev0_share_a,
    )
    if forward_type.startswith("debug"):
      from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations
      FORWARD_TYPES.update(dict(
          debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
          debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
          debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16, self),
          debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs, self),
          debugforward_core_mambassm_fusecscm=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
          debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm, self),
          debugforward_core_sscore_fusecscm_fwdnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_fwdnrow, self),
          debugforward_core_sscore_fusecscm_bwdnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_bwdnrow, self),
          debugforward_core_sscore_fusecscm_fbnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_fbnrow, self),
          debugforward_core_ssoflex_fusecscm=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm, self),
          debugforward_core_ssoflex_fusecscm_i16o32=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
      ))
    self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))
    self.K = 4 if forward_type not in ["share_ssm"] else 1
    self.K2 = self.K if forward_type not in ["share_a"] else 1

    # in proj =======================================
    d_proj = d_expand if self.disable_z else (d_expand * 2)
    self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
    self.act: nn.Module = act_layer()

    # conv =======================================
    if self.d_conv > 1:
      self.conv2d = nn.Conv2d(
          in_channels=d_expand,
          out_channels=d_expand,
          groups=d_expand,
          bias=conv_bias,
          kernel_size=d_conv,
          padding=(d_conv - 1) // 2,
          **factory_kwargs,
      )

    # rank ratio =====================================
    self.ssm_low_rank = False
    if d_inner < d_expand:
      self.ssm_low_rank = True
      self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
      self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

    # x proj ============================
    self.x_proj = [
        nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        for _ in range(self.K)
    ]
    self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
    del self.x_proj

    # out proj =======================================
    self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
    self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    if initialize in ["v0"]:
      # dt proj ============================
      self.dt_projs = [
          self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
          for _ in range(self.K)
      ]
      self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
      self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
      del self.dt_projs

      # A, D =======================================
      self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
      self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)
    elif initialize in ["v1"]:
      # simple init dt_projs, A_logs, Ds
      self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
      self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))
                                 )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
      self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
      self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))
    elif initialize in ["v2"]:
      # simple init dt_projs, A_logs, Ds
      self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
      self.A_logs = nn.Parameter(torch.zeros((self.K2 * d_inner, self.d_state))
                                 )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
      self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
      self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

  @staticmethod
  def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001,
              dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
    dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
      nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
      nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    else:
      raise NotImplementedError

    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    dt = torch.exp(
        torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
      dt_proj.bias.copy_(inv_dt)
    # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
    # dt_proj.bias._no_reinit = True

    return dt_proj

  @staticmethod
  def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
    # S4D real initialization
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
        "n -> d n",
        d=d_inner,
    ).contiguous()
    A_log = torch.log(A)  # Keep A_log in fp32
    if copies > 0:
      A_log = repeat(A_log, "d n -> r d n", r=copies)
      if merge:
        A_log = A_log.flatten(0, 1)
    A_log = nn.Parameter(A_log)
    A_log._no_weight_decay = True
    return A_log

  @staticmethod
  def D_init(d_inner, copies=-1, device=None, merge=True):
    # D "skip" parameter
    D = torch.ones(d_inner, device=device)
    if copies > 0:
      D = repeat(D, "n1 -> r n1", r=copies)
      if merge:
        D = D.flatten(0, 1)
    D = nn.Parameter(D)  # Keep in fp32
    D._no_weight_decay = True
    return D

  # only used to run previous version
  def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
      return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    B, C, H, W = x.shape
    L = H * W
    K = 4

    x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2,
                         dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
    xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
    # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

    xs = xs.float().view(B, -1, L)  # (b, k * d, l)
    dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
    Bs = Bs.float()  # (b, k, d_state, l)
    Cs = Cs.float()  # (b, k, d_state, l)

    As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
    Ds = self.Ds.float()  # (k * d)
    dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

    # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
    # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

    out_y = selective_scan(
        xs, dts,
        As, Bs, Cs, Ds,
        delta_bias=dt_projs_bias,
        delta_softplus=True,
    ).view(B, K, -1, L)
    # assert out_y.dtype == torch.float

    inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
    wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = self.out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

  def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
    """
    we may conduct this ablation later, but not with v0.
    """
    ...

  def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
    """
    we may conduct this ablation later, but not with v0.
    """
    ...

  def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                     cross_selective_scan=cross_selective_scan, force_fp32=None):
    force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    if self.ssm_low_rank:
      x = self.in_rank(x)
    x = cross_selective_scan(
        x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
        self.A_logs, self.Ds,
        out_norm=getattr(self, "out_norm", None),
        out_norm_shape=getattr(self, "out_norm_shape", "v0"),
        delta_softplus=True, force_fp32=force_fp32,
        SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
    )
    if self.ssm_low_rank:
      x = self.out_rank(x)
    return x

  def forward_corev2_3d(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                        cross_selective_scan=cross_selective_scan, force_fp32=None):
    force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    if self.ssm_low_rank:
      x = self.in_rank(x)
    x = cross_selective_scan3D(
        x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
        self.A_logs, self.Ds,
        out_norm=getattr(self, "out_norm", None),
        out_norm_shape=getattr(self, "out_norm_shape", "v0"),
        delta_softplus=True, force_fp32=force_fp32,
        SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
    )
    if self.ssm_low_rank:
      x = self.out_rank(x)
    return x

  def forward(self, x: torch.Tensor, **kwargs):

    x = self.in_proj(x)
    if not self.disable_z:
      x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
      if not self.disable_z_act:
        z = self.act(z)
    if self.d_conv > 0:
      x = x.permute(0, 3, 1, 2).contiguous()
      x = self.conv2d(x)  # (b, d, h, w)

    x = self.act(x)
    y = self.forward_core(x, channel_first=(self.d_conv > 1))
    if not self.disable_z:
      y = y * z
    out = self.dropout(self.out_proj(y))
    return out


class SS2D4D(nn.Module):
  def __init__(
      self,
      # basic dims ===========
      d_model=96,
      d_state=16,
      ssm_ratio=2.0,
      ssm_rank_ratio=2.0,
      dt_rank="auto",
      act_layer=nn.SiLU,
      # dwconv ===============
      d_conv=3,  # < 2 means no conv
      conv_bias=True,
      # ======================
      dropout=0.0,
      bias=False,
      # dt init ==============
      dt_min=0.001,
      dt_max=0.1,
      dt_init="random",
      dt_scale=1.0,
      dt_init_floor=1e-4,
      initialize="v0",
      # ======================
      forward_type="v2",
      # ======================
      **kwargs,
  ):
    """
    ssm_rank_ratio would be used in the future...
    """
    factory_kwargs = {"device": None, "dtype": None}
    super().__init__()
    d_expand = int(ssm_ratio * d_model)
    d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
    self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
    self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
    self.d_conv = d_conv

    # tags for forward_type ==============================
    def checkpostfix(tag, value):
      ret = value[-len(tag):] == tag
      if ret:
        value = value[:-len(tag)]
      return ret, value

    self.disable_force32, forward_type = checkpostfix("no32", forward_type)
    self.disable_z, forward_type = checkpostfix("noz", forward_type)
    self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

    # softmax | sigmoid | dwconv | norm ===========================
    if forward_type[-len("none"):] == "none":
      forward_type = forward_type[:-len("none")]
      self.out_norm = nn.Identity()
    elif forward_type[-len("dwconv3"):] == "dwconv3":
      forward_type = forward_type[:-len("dwconv3")]
      self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
      self.out_norm_shape = "v1"
    elif forward_type[-len("softmax"):] == "softmax":
      forward_type = forward_type[:-len("softmax")]
      self.out_norm = nn.Softmax(dim=1)
    elif forward_type[-len("sigmoid"):] == "sigmoid":
      forward_type = forward_type[:-len("sigmoid")]
      self.out_norm = nn.Sigmoid()
    else:
      self.out_norm = nn.LayerNorm(d_inner)

    # forward_type debug =======================================
    FORWARD_TYPES = dict(
        v0=self.forward_corev0,
        fake=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanFake),
        v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        v2_4d=partial(self.forward_corev2_4d, force_fp32=None, SelectiveScan=SelectiveScanCore),
        v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
        v4=partial(
            self.forward_corev2,
            force_fp32=False,
            SelectiveScan=SelectiveScanOflex,
            cross_selective_scan=cross_selective_scanv2),
        v1=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanOflex),
        v01=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanMamba),
        share_ssm=self.forward_corev0_share_ssm,
        share_a=self.forward_corev0_share_a,
    )
    if forward_type.startswith("debug"):
      from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations
      FORWARD_TYPES.update(dict(
          debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
          debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
          debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16, self),
          debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs, self),
          debugforward_core_mambassm_fusecscm=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
          debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm, self),
          debugforward_core_sscore_fusecscm_fwdnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_fwdnrow, self),
          debugforward_core_sscore_fusecscm_bwdnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_bwdnrow, self),
          debugforward_core_sscore_fusecscm_fbnrow=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm_fbnrow, self),
          debugforward_core_ssoflex_fusecscm=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm, self),
          debugforward_core_ssoflex_fusecscm_i16o32=partial(
              SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
      ))
    self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))
    self.K = 4 if forward_type not in ["share_ssm"] else 1
    self.K2 = self.K if forward_type not in ["share_a"] else 1

    # in proj =======================================
    d_proj = d_expand if self.disable_z else (d_expand * 2)
    self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
    self.act: nn.Module = act_layer()

    # conv =======================================
    if self.d_conv > 1:
      self.conv2d = nn.Conv2d(
          in_channels=d_expand,
          out_channels=d_expand,
          groups=d_expand,
          bias=conv_bias,
          kernel_size=d_conv,
          padding=(d_conv - 1) // 2,
          **factory_kwargs,
      )

    # rank ratio =====================================
    self.ssm_low_rank = False
    if d_inner < d_expand:
      self.ssm_low_rank = True
      self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
      self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

    # x proj ============================
    self.x_proj = [
        nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        for _ in range(self.K)
    ]
    self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
    del self.x_proj

    # out proj =======================================
    self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
    self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    if initialize in ["v0"]:
      # dt proj ============================
      self.dt_projs = [
          self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
          for _ in range(self.K)
      ]
      self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
      self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
      del self.dt_projs

      # A, D =======================================
      self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
      self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)
    elif initialize in ["v1"]:
      # simple init dt_projs, A_logs, Ds
      self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
      self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))
                                 )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
      self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
      self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))
    elif initialize in ["v2"]:
      # simple init dt_projs, A_logs, Ds
      self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
      self.A_logs = nn.Parameter(torch.zeros((self.K2 * d_inner, self.d_state))
                                 )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
      self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
      self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

  @staticmethod
  def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001,
              dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
    dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = dt_rank**-0.5 * dt_scale
    if dt_init == "constant":
      nn.init.constant_(dt_proj.weight, dt_init_std)
    elif dt_init == "random":
      nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    else:
      raise NotImplementedError

    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    dt = torch.exp(
        torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
      dt_proj.bias.copy_(inv_dt)
    # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
    # dt_proj.bias._no_reinit = True

    return dt_proj

  @staticmethod
  def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
    # S4D real initialization
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
        "n -> d n",
        d=d_inner,
    ).contiguous()
    A_log = torch.log(A)  # Keep A_log in fp32
    if copies > 0:
      A_log = repeat(A_log, "d n -> r d n", r=copies)
      if merge:
        A_log = A_log.flatten(0, 1)
    A_log = nn.Parameter(A_log)
    A_log._no_weight_decay = True
    return A_log

  @staticmethod
  def D_init(d_inner, copies=-1, device=None, merge=True):
    # D "skip" parameter
    D = torch.ones(d_inner, device=device)
    if copies > 0:
      D = repeat(D, "n1 -> r n1", r=copies)
      if merge:
        D = D.flatten(0, 1)
    D = nn.Parameter(D)  # Keep in fp32
    D._no_weight_decay = True
    return D

  # only used to run previous version
  def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
      return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    B, C, H, W = x.shape
    L = H * W
    K = 4

    x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2,
                         dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
    xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
    # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

    xs = xs.float().view(B, -1, L)  # (b, k * d, l)
    dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
    Bs = Bs.float()  # (b, k, d_state, l)
    Cs = Cs.float()  # (b, k, d_state, l)

    As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
    Ds = self.Ds.float()  # (k * d)
    dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

    # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
    # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

    out_y = selective_scan(
        xs, dts,
        As, Bs, Cs, Ds,
        delta_bias=dt_projs_bias,
        delta_softplus=True,
    ).view(B, K, -1, L)
    # assert out_y.dtype == torch.float

    inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
    wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = self.out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

  def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
    """
    we may conduct this ablation later, but not with v0.
    """
    ...

  def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
    """
    we may conduct this ablation later, but not with v0.
    """
    ...

  def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                     cross_selective_scan=cross_selective_scan, force_fp32=None):
    force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    if self.ssm_low_rank:
      x = self.in_rank(x)
    x = cross_selective_scan(
        x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
        self.A_logs, self.Ds,
        out_norm=getattr(self, "out_norm", None),
        out_norm_shape=getattr(self, "out_norm_shape", "v0"),
        delta_softplus=True, force_fp32=force_fp32,
        SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
    )
    if self.ssm_low_rank:
      x = self.out_rank(x)
    return x

  def forward_corev2_4d(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                        cross_selective_scan=cross_selective_scan, force_fp32=None):
    force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
    if not channel_first:
      x = x.permute(0, 3, 1, 2).contiguous()
    if self.ssm_low_rank:
      x = self.in_rank(x)
    x = cross_selective_scan4D(
        x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
        self.A_logs, self.Ds,
        out_norm=getattr(self, "out_norm", None),
        out_norm_shape=getattr(self, "out_norm_shape", "v0"),
        delta_softplus=True, force_fp32=force_fp32,
        SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
    )
    if self.ssm_low_rank:
      x = self.out_rank(x)
    return x

  def forward(self, x: torch.Tensor, **kwargs):

    x = self.in_proj(x)
    if not self.disable_z:
      x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
      if not self.disable_z_act:
        z = self.act(z)
    if self.d_conv > 0:
      x = x.permute(0, 3, 1, 2).contiguous()
      x = self.conv2d(x)  # (b, d, h, w)

    x = self.act(x)
    y = self.forward_core(x, channel_first=(self.d_conv > 1))
    if not self.disable_z:
      y = y * z
    out = self.dropout(self.out_proj(y))
    return out


class DWConv(nn.Module):
  def __init__(self, dim=64):
    super(DWConv, self).__init__()
    self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

  def forward(self, x, h, w):
    # B, N, C = x.shape
    x = x.permute(0, 3, 1, 2).contiguous()
    x = self.dwconv(x)
    x = x.permute(0, 2, 3, 1).contiguous()
    # x = x.flatten(2).transpose(1, 2).contiguous()
    return x


class Conv_Mlp(nn.Module):
  """ MLP as used in Vision Transformer, MLP-Mixer and related networks
  """

  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features

    self.fc1 = nn.Linear(in_features, hidden_features)
    self.dwconv = DWConv(hidden_features)
    self.act = act_layer()
    self.drop1 = nn.Dropout(drop)
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop2 = nn.Dropout(drop)

  def forward(self, x, h, w):
    x = self.fc1(x)
    x = self.dwconv(x, h, w)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


class LFSSBlock(nn.Module):
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      # =============================
      ssm_d_state: int = 16,
      ssm_ratio=2.0,
      ssm_rank_ratio=2.0,
      ssm_dt_rank: Any = "auto",
      ssm_act_layer=nn.SiLU,
      ssm_conv: int = 3,
      ssm_conv_bias=True,
      ssm_drop_rate: float = 0,
      ssm_init="v0",
      forward_type="v2",
      # =============================
      mlp_ratio=4.0,
      mlp_act_layer=nn.GELU,
      mlp_drop_rate: float = 0.0,
      # =============================
      use_checkpoint: bool = False,
      post_norm: bool = False,
      **kwargs,
  ):
    super().__init__()
    self.ssm_branch = ssm_ratio > 0
    self.mlp_branch = mlp_ratio > 0
    self.use_checkpoint = use_checkpoint
    self.post_norm = post_norm

    if self.ssm_branch:
      self.norm = norm_layer(hidden_dim)

      self.op_mac = SS2D3D(
          d_model=hidden_dim,
          d_state=ssm_d_state,
          ssm_ratio=ssm_ratio,
          ssm_rank_ratio=ssm_rank_ratio,
          dt_rank=ssm_dt_rank,
          act_layer=ssm_act_layer,
          d_conv=0,
          conv_bias=ssm_conv_bias,
          dropout=ssm_drop_rate,
          initialize=ssm_init,
          forward_type="v2_3d",
      )

      self.ssm_skip_scale = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

    self.drop_path = DropPath(drop_path)

    if self.mlp_branch:
      self.norm2 = norm_layer(hidden_dim)
      mlp_hidden_dim = int(hidden_dim * mlp_ratio)
      self.mlp = Conv_Mlp(
          in_features=hidden_dim,
          hidden_features=mlp_hidden_dim,
          act_layer=mlp_act_layer,
          drop=mlp_drop_rate)
      self.mlp_skip_scale = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

  def _forward(self, input: torch.Tensor):

    stride = __SIZE__
    _, H, W, _ = input.shape

    if self.ssm_branch:

      x = self.norm(input)

      x_mac = rearrange(x, '(b u v) h w c -> b (h u) (w v) c', u=5, v=5, h=H, w=W).contiguous()
      x_mac = rearrange(self.op_mac(x_mac), 'b (h u) (w v) c -> (b u v) h w c', u=5, v=5, h=H, w=W).contiguous()
      x = self.ssm_skip_scale * input + self.drop_path(x_mac)

    if self.mlp_branch:
      x = self.mlp_skip_scale * x + self.drop_path(self.mlp(self.norm2(x), H // 5, W // 5))  # FFN

    return x

  def forward(self, input: torch.Tensor):

    if self.use_checkpoint and self.training:
      return checkpoint.checkpoint(self._forward, input, use_reentrant=True)
    else:
      return self._forward(input)


class VLSSBlock(nn.Module):
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      # =============================
      ssm_d_state: int = 16,
      ssm_ratio=2.0,
      ssm_rank_ratio=2.0,
      ssm_dt_rank: Any = "auto",
      ssm_act_layer=nn.SiLU,
      ssm_conv: int = 3,
      ssm_conv_bias=True,
      ssm_drop_rate: float = 0,
      ssm_init="v0",
      forward_type="v2",
      # =============================
      mlp_ratio=4.0,
      mlp_act_layer=nn.GELU,
      mlp_drop_rate: float = 0.0,
      # =============================
      use_checkpoint: bool = False,
      post_norm: bool = False,
      **kwargs,
  ):
    super().__init__()
    self.ssm_branch = ssm_ratio > 0
    self.mlp_branch = mlp_ratio > 0
    self.use_checkpoint = use_checkpoint
    self.post_norm = post_norm

    if self.ssm_branch:
      self.norm = norm_layer(hidden_dim)

      self.op_mac = SS2D4D(
          d_model=hidden_dim,
          d_state=ssm_d_state,
          ssm_ratio=ssm_ratio,
          ssm_rank_ratio=ssm_rank_ratio,
          dt_rank=ssm_dt_rank,
          act_layer=ssm_act_layer,
          d_conv=0,
          conv_bias=ssm_conv_bias,
          dropout=ssm_drop_rate,
          initialize=ssm_init,
          forward_type="v2_4d",
      )

      self.ssm_skip_scale = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

    self.drop_path = DropPath(drop_path)

    if self.mlp_branch:
      self.norm2 = norm_layer(hidden_dim)
      mlp_hidden_dim = int(hidden_dim * mlp_ratio)
      self.mlp = Conv_Mlp(
          in_features=hidden_dim,
          hidden_features=mlp_hidden_dim,
          act_layer=mlp_act_layer,
          drop=mlp_drop_rate)
      self.mlp_skip_scale = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

  def _forward(self, input: torch.Tensor):

    stride = __SIZE__
    _, H, W, _ = input.shape

    if self.ssm_branch:

      x = self.norm(input)
      x = rearrange(x, '(b u v) h w c -> b (u h) (v w) c', u=5, v=5, h=H, w=W)
      x = rearrange(self.op_mac(x), 'b (u h) (v w) c -> (b u v) h w c', u=5, v=5, h=H, w=W)
      x = self.ssm_skip_scale * input + self.drop_path(x)

    if self.mlp_branch:
      x = self.mlp_skip_scale * x + self.drop_path(self.mlp(self.norm2(x), H // 5, W // 5))  # FFN

    return x

  def forward(self, input: torch.Tensor):

    if self.use_checkpoint and self.training:
      return checkpoint.checkpoint(self._forward, input, use_reentrant=True)
    else:
      return self._forward(input)


class VSSBlock(nn.Module):
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      # =============================
      ssm_d_state: int = 16,
      ssm_ratio=2.0,
      ssm_rank_ratio=2.0,
      ssm_dt_rank: Any = "auto",
      ssm_act_layer=nn.SiLU,
      ssm_conv: int = 3,
      ssm_conv_bias=True,
      ssm_drop_rate: float = 0,
      ssm_init="v0",
      forward_type="v2",
      # =============================
      mlp_ratio=4.0,
      mlp_act_layer=nn.GELU,
      mlp_drop_rate: float = 0.0,
      # =============================
      use_checkpoint: bool = False,
      post_norm: bool = False,
      num_spa_trans=2,
      **kwargs,
  ):
    super().__init__()
    self.ssm_branch = ssm_ratio > 0
    self.mlp_branch = mlp_ratio > 0
    self.use_checkpoint = use_checkpoint
    self.post_norm = post_norm

    if self.ssm_branch:
      self.norm = norm_layer(hidden_dim)

      self.op_sai = SS2D(
          d_model=hidden_dim,
          d_state=ssm_d_state,
          ssm_ratio=ssm_ratio,
          ssm_rank_ratio=ssm_rank_ratio,
          dt_rank=ssm_dt_rank,
          act_layer=ssm_act_layer,
          d_conv=ssm_conv,
          conv_bias=ssm_conv_bias,
          dropout=ssm_drop_rate,
          initialize=ssm_init,
          forward_type=forward_type,
      )

      self.ssm_skip_scale = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

    self.drop_path = DropPath(drop_path)

    if self.mlp_branch:
      self.norm2 = norm_layer(hidden_dim)
      mlp_hidden_dim = int(hidden_dim * mlp_ratio)
      self.mlp = Conv_Mlp(
          in_features=hidden_dim,
          hidden_features=mlp_hidden_dim,
          act_layer=mlp_act_layer,
          drop=mlp_drop_rate)
      self.mlp_skip_scale = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

    self.spa_trans = []
    for i in range(num_spa_trans):
      self.spa_trans.append(TransBlock(
          dim=hidden_dim,
          num_heads=4,
          mlp_ratio=4,
          qkv_bias=True,
          drop=0.0,
          attn_drop=0.0,
          sr_ratio=1,
          act_layer=nn.GELU,
          norm_layer=nn.LayerNorm,
          use_flashatten=True))
    self.spa_trans = nn.ModuleList(self.spa_trans)

  def _forward(self, input: torch.Tensor):

    stride = __SIZE__
    _, H, W, _ = input.shape

    if self.ssm_branch:

      x = input
      for trans_layer in self.spa_trans:
        x = trans_layer(x)
      x = self.norm(x)
      x = self.ssm_skip_scale * input + self.drop_path(self.op_sai(x))

    if self.mlp_branch:
      x = self.mlp_skip_scale * x + self.drop_path(self.mlp(self.norm2(x), H // 5, W // 5))  # FFN

    return x

  def forward(self, input: torch.Tensor):

    if self.use_checkpoint and self.training:
      return checkpoint.checkpoint(self._forward, input, use_reentrant=True)
    else:
      return self._forward(input)


class LFTransMamba(nn.Module):
  """Tunable Parameters

      - D: d_state
      - R: ssm_ratio
      - K: num_blocks
      - C: channels

  """

  def __init__(self, angRes_in, scale_factor, D=16, R=1, K=4, C=64, T=3):
    super(LFTransMamba, self).__init__()
    channels = C
    num_blocks = K
    ssm_d_state = D
    ssm_ratio = R
    num_spa_trans = T

    self.hierarchy_list = [*range(2, 2 + 3 * num_blocks, 3)]

    self.angRes = angRes_in
    self.scale = scale_factor

    self.ang_embed = nn.Parameter(torch.zeros(1, channels, self.angRes**2, 1, 1), requires_grad=True)
    self.mask_token = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
    
    #################### Initial Feature Extraction #####################
    self.conv_init0 = nn.Sequential(nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
    self.conv_init = nn.Sequential(
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        nn.LeakyReLU(0.1, inplace=True),
    )

    ############# Deep Spatial-Angular Correlation Learning #############

    self.layers = nn.ModuleList()
    self.depths = [1, 1, 1] * num_blocks
    self.type_layers = [VSSBlock, VLSSBlock, LFSSBlock] * num_blocks
    self.num_layers = len(self.depths)
    self.dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.depths))]

    for i_layer in range(self.num_layers):
      self.layers.append(self._make_layer(
          self.type_layers[i_layer],
          dim=channels,
          drop_path=self.dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
          use_checkpoint=False,
          norm_layer=nn.LayerNorm,
          downsample=nn.Identity(),
          # =================
          ssm_d_state=ssm_d_state,
          ssm_ratio=ssm_ratio,
          ssm_rank_ratio=2.0,
          ssm_dt_rank='auto',
          ssm_act_layer=nn.SiLU,
          ssm_conv=3,
          ssm_conv_bias=True,
          ssm_drop_rate=0.0,
          ssm_init='v0',
          forward_type='v2',
          # =================
          mlp_ratio=4.0,
          mlp_act_layer=nn.GELU,
          mlp_drop_rate=0.0,
          num_spa_trans=num_spa_trans,
      ))

    self.mla = nn.Sequential(
        nn.Conv2d(channels * num_blocks, channels, kernel_size=3, padding=1, bias=False),
        nn.LeakyReLU(0.1),
    )

    ########################### UP-Sampling #############################
    self.upsampling = nn.Sequential(
        nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
        nn.PixelShuffle(self.scale),
        nn.LeakyReLU(0.1),
        nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
    )

  @staticmethod
  def _make_layer(
      block,
      dim=96,
      drop_path=[0.1, 0.1],
      use_checkpoint=True,
      norm_layer=nn.LayerNorm,
      downsample=nn.Identity(),
      # ===========================
      ssm_d_state=16,
      ssm_ratio=2.0,
      ssm_rank_ratio=2.0,
      ssm_dt_rank="auto",
      ssm_act_layer=nn.SiLU,
      ssm_conv=3,
      ssm_conv_bias=True,
      ssm_drop_rate=0.0,
      ssm_init="v0",
      forward_type="v2",
      # ===========================
      mlp_ratio=4.0,
      mlp_act_layer=nn.GELU,
      mlp_drop_rate=0.0,
      num_spa_trans=2,
      **kwargs,
  ):
    depth = len(drop_path)
    blocks = []
    for d in range(depth):
      blocks.append(block(
          hidden_dim=dim,
          drop_path=drop_path[d],
          norm_layer=norm_layer,
          ssm_d_state=ssm_d_state,
          ssm_ratio=ssm_ratio,
          ssm_rank_ratio=ssm_rank_ratio,
          ssm_dt_rank=ssm_dt_rank,
          ssm_act_layer=ssm_act_layer,
          ssm_conv=ssm_conv,
          ssm_conv_bias=ssm_conv_bias,
          ssm_drop_rate=ssm_drop_rate,
          ssm_init=ssm_init,
          forward_type=forward_type,
          mlp_ratio=mlp_ratio,
          mlp_act_layer=mlp_act_layer,
          mlp_drop_rate=mlp_drop_rate,
          use_checkpoint=use_checkpoint,
          num_spa_trans=num_spa_trans,
      ))

    return nn.Sequential(OrderedDict(
        blocks=nn.Sequential(*blocks,),
        downsample=downsample,
    ))

  def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    unmasked_x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - unmasked_x.shape[1], 1)
    masked_x = torch.cat([unmasked_x, mask_tokens], dim=1)  # no cls token
    masked_x = torch.gather(masked_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

    return masked_x
  
  def forward(self, lr, info=None):
    """
    """
    b, c, uh, vw = lr.shape
    u, v = self.angRes, self.angRes
    h, w = uh // u, vw // v

    #!< NOTE(kj): LFSR interpolation should be on SAI view instead of full image!!
    sr = rearrange(lr, 'b c (u h) (v w) -> (b u v) c h w', u=u, v=v, h=h, w=w)
    sr = F.interpolate(sr, scale_factor=self.scale, mode='bicubic', align_corners=False)
    sr = rearrange(sr, '(b u v) c h w -> b c (u h) (v w)', u=u, v=v).contiguous()

    # Initial Feature Extraction
    x = rearrange(lr, 'b c (u h) (v w) -> b c (u v) h w', u=self.angRes, v=self.angRes).contiguous()
    buffer = self.conv_init0(x)
    buffer = self.conv_init(buffer) + buffer

    if self.training:
      mask_buffer = rearrange(buffer, 'b c (u v) h w -> (b u v) (h w) c', u=5, v=5)
      mask_buffer = self.random_masking(mask_buffer, mask_ratio=0.35)
      buffer = rearrange(mask_buffer, '(b u v) (h w) c -> b c (u v) h w', b=b, u=u, v=v, h=h, w=w)

    # Deep Spatial-Angular Correlation Learning
    # start Mamba (input require: [B, C, H, W])
    buffer_cache = rearrange(
        buffer + self.ang_embed,
        'b c (u v) h w -> (b u v) h w c',
        u=self.angRes,
        v=self.angRes).contiguous()

    hierarchical_feature = []
    for index, layer in enumerate(self.layers):
      buffer_cache = layer(buffer_cache)
      if index in self.hierarchy_list:
        hierarchical_feature.append(buffer_cache)

    buffer_cache = torch.cat(hierarchical_feature, dim=-1)
    buffer_cache = self.mla(rearrange(buffer_cache, '(b u v) h w c -> (b u v) c h w', h=h, w=w, u=u, v=v).contiguous())
    buffer_cache = rearrange(buffer_cache, '(b u v) c h w -> b (u h) (v w) c', h=h, w=w, u=u, v=v).contiguous()

    # UP-Sampling
    # end Mamba (input require: [B, C, H, W])
    buffer = rearrange(buffer_cache, 'b (u h) (v w) c -> b c (u h) (v w)', h=h, w=w, u=u, v=v).contiguous() + \
        rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', h=h, w=w, u=u, v=v).contiguous()

    y = self.upsampling(buffer)
    return y + sr


class get_loss(nn.Module):
  def __init__(self, args):
    super(get_loss, self).__init__()
    self.criterion_Loss = torch.nn.L1Loss()

  def forward(self, out, HR, degrade_info=None):
    loss = self.criterion_Loss(out['SR'], HR)

    return loss


def weights_init(m: nn.Module):

  if isinstance(m, nn.Linear):
    trunc_normal_(m.weight, std=.02)
    if isinstance(m, nn.Linear) and m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.LayerNorm):
    nn.init.constant_(m.bias, 0)
    nn.init.constant_(m.weight, 1.0)


def _selective_scan_flop_jit(inputs, outputs):
  def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
      flops += B * D * L
    if with_Z:
      flops += B * D * L
    return flops

  B, D, L = inputs[0].type().sizes()
  N = inputs[2].type().sizes()[1]
  flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
  return flops

if __name__ == '__main__':
  from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count

  supported_ops = {
      "aten::silu": None,  # as relu is in _IGNORED_OPS
      "aten::neg": None,  # as relu is in _IGNORED_OPS
      "aten::exp": None,  # as relu is in _IGNORED_OPS
      "aten::flip": None,  # as permute is in _IGNORED_OPS
      "prim::PythonOp.SelectiveScanMamba": _selective_scan_flop_jit,
      "prim::PythonOp.SelectiveScanOflex": _selective_scan_flop_jit,
      "prim::PythonOp.SelectiveScanCore": _selective_scan_flop_jit,
      "prim::PythonOp.SelectiveScanNRow": _selective_scan_flop_jit,
  }

  model = LFTransMamba(5, 4, D=16, R=1, K=6, C=32, T=0)
  model.eval().cuda()
  inputs = (torch.rand(1, 1, 32 * 5, 32 * 5).cuda(),)
  counter = FlopCountAnalysis(model, inputs)
  counter.set_op_handle(**supported_ops)
  # print(counter.total())
  
  for name, flops in counter.by_module().items():
    print(f"{name}: {flops}")