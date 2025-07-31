from lib import *


def round_width(width, multiplier, min_width = 8, divisor = 8, ceil = False):
  if not multiplier:
    return width
  
  width *= multiplier
  min_width = min_width or divisor
  if ceil:
      width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
  else:
      width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
  if width_out < 0.9 * width:
      width_out += divisor
  return int(width_out)


def create_x3d_stem(
    # Conv configs.
    in_channels: int,
    out_channels: int,
    conv_kernel_size: Tuple[int] = (5, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    conv_padding: Tuple[int] = (2, 1, 1),
    # BN configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
) -> nn.Module:


  conv_xy_module = nn.Conv3d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = (1, conv_kernel_size[1], conv_kernel_size[2]),
      stride=(1, conv_stride[1], conv_stride[2]),
      padding=(0, conv_padding[1], conv_padding[2]),
      bias=False,
  )

  
  conv_t_module = nn.Conv3d(
      in_channels = out_channels,
      out_channels = out_channels,
      kernel_size=(conv_kernel_size[0], 1, 1),
      stride=(conv_stride[0], 1, 1),
      padding=(conv_padding[0], 0, 0),
      bias=False,
      groups=out_channels, 
  )


  stacked_conv_module = Conv2plus1d(
      conv_t=conv_xy_module,
      norm=None,
      activation=None,
      conv_xy=conv_t_module,
  )


  norm_module = (
      None
      if norm is None
      else norm(num_features=out_channels, eps=norm_eps, momentum=norm_momentum)
  )


  activation_module = None if activation is None else activation()

  return ResNetBasicStem(
      conv = stacked_conv_module,
      norm = norm_module,
      activation = activation_module,
      pool = None
  )



class Conv2plus1d(nn.Module):
  def __init__(
      self,
      conv_t: nn.Module = None,
      norm: nn.Module = None,
      activation: nn.Module = None,
      conv_xy: nn.Module = None,
      conv_xy_first: bool = False,
  ) -> None:
    super(Conv2plus1d, self).__init__()
    self.conv_t = conv_t
    self.norm = norm
    self.activation = activation
    self.conv_xy = conv_xy
    self.conv_xy_first = conv_xy_first
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv_xy(x) if self.conv_xy_first else self.conv_t(x)
    x = self.norm(x) if self.norm else x
    x = self.activation(x) if self.activation else x
    x = self.conv_t(x) if self.conv_xy_first else self.conv_xy(x)
    return x

class ResNetBasicStem(nn.Module):
  def __init__(self, 
               conv: nn.Module = None,
               norm: nn.Module = None,
               activation: nn.Module = None,
               pool: nn.Module = None
  ):
    super().__init__()
    self.conv = conv
    self.norm = norm
    self.activation = activation
    self.pool = pool
  
  def forward(self, x):
    x = self.conv(x)
    if self.norm is not None:
      x = self.norm(x)
    if self.activation is not None:
      x = self.activation(x)
    if self.pool is not None:
      x = self.pool(x)
    
    return x

class Swish(nn.Module):
    def forward(self, x):
        return SwishFunction.apply(x)


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

def create_x3d_bottleneck_block(
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:

  conv_a = nn.Conv3d(
      in_channels = dim_in,
      out_channels = dim_inner,
      kernel_size = (1, 1, 1),
      bias = False
  )
  norm_a = (
      None 
      if norm is None 
      else norm(num_features = dim_inner, eps = norm_eps, momentum = norm_momentum)
  )
  act_a = None if activation is None else activation()

  # 3x3x3 Conv (Separable Convolution) 
  conv_b = nn.Conv3d(
      in_channels = dim_inner,
      out_channels = dim_inner,
      kernel_size = conv_kernel_size,
      stride = conv_stride,
      padding = [size // 2 for size in conv_kernel_size],
      bias = False,
      groups = dim_inner, 
      dilation = (1, 1, 1)
  )

  se = (
      SqueezeExcitation(
          num_channels = dim_inner,
          num_channels_reduced = round_width(dim_inner, se_ratio),
          is_3d = True
      )
      if se_ratio > 0.0
      else nn.Identity()
  )
  
  norm_b = nn.Sequential(
      (
          nn.Identity()
          if norm is None
          else norm(num_features = dim_inner, eps = norm_eps, momentum = norm_momentum)    
      ),
      se
  )
  act_b = None if inner_act is None else inner_act()

  # 1x1x1 Conv (Separable Convolution)
  conv_c = nn.Conv3d(
      in_channels = dim_inner,
      out_channels = dim_out,
      kernel_size = (1, 1, 1),
      bias = False
  )
  norm_c = (
      None
      if norm is None
      else norm(num_features = dim_out, eps = norm_eps, momentum = norm_momentum)
  )

  return BottleneckBlock(
      conv_a=conv_a,
      norm_a=norm_a,
      act_a=act_a,
      conv_b=conv_b,
      norm_b=norm_b,
      act_b=act_b,
      conv_c=conv_c,
      norm_c=norm_c
  )

class BottleneckBlock(nn.Module):

  def __init__(
      self,
      conv_a: nn.Module = None,
      norm_a: nn.Module = None,
      act_a: nn.Module = None,
      conv_b: nn.Module = None,
      norm_b: nn.Module = None,
      act_b: nn.Module = None,
      conv_c: nn.Module = None,
      norm_c: nn.Module = None,
  ):
    super(BottleneckBlock, self).__init__()
    self.conv_a = conv_a
    self.norm_a = norm_a
    self.act_a = act_a

    self.conv_b = conv_b
    self.norm_b = norm_b
    self.act_b = act_b

    self.conv_c = conv_c
    self.norm_c = norm_c
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv_a(x)
    x = self.norm_a(x) if self.norm_a is not None else x
    x = self.act_a(x) if self.act_a is not None else x

    x = self.conv_b(x)
    x = self.norm_b(x) if self.norm_b is not None else x
    x = self.act_b(x) if self.act_b is not None else x

    x = self.conv_c(x)
    x = self.norm_c(x) if self.norm_c is not None else x

    return x

class ResBlock(nn.Module):

  def __init__(
      self,
      branch1_conv: nn.Module = None,
      branch1_norm: nn.Module = None,
      branch2: nn.Module = None,
      activation: nn.Module = None,
      branch_fusion: Callable = None
  ) -> nn.Module:
    super(ResBlock, self).__init__()
    self.branch1_conv = branch1_conv
    self.branch1_norm = branch1_norm
    self.branch2 = branch2
    self.activation = activation
    self.branch_fusion = branch_fusion
  
  def forward(self, x) -> torch.Tensor:
    if self.branch1_conv is None:
      x = self.branch_fusion(x, self.branch2(x))
    else:
      shortcut = self.branch1_conv(x)
      if self.branch1_norm is not None:
        shortcut = self.branch1_norm(shortcut)
      x = self.branch_fusion(shortcut, self.branch2(x))
    
    if self.activation is not None:
      x = self.activation(x)
    return x

def create_x3d_res_block(
    # Bottleneck Block configs.
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    use_shortcut: bool = True,
    # Conv configs
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish
) -> nn.Module:

  norm_model = None
  if norm is not None and dim_in != dim_out:
    norm_model = norm(num_features = dim_out)
  

  return ResBlock(

      branch1_conv = nn.Conv3d(dim_in, dim_out, kernel_size = (1, 1, 1), stride = conv_stride, bias = False)
      if (dim_in != dim_out or np.prod(conv_stride) > 1) and use_shortcut
      else None,
      branch1_norm = norm_model if dim_in != dim_out and use_shortcut else None,

      branch2 = bottleneck(
          dim_in = dim_in,
          dim_inner = dim_inner,
          dim_out = dim_out,
          conv_kernel_size=conv_kernel_size,
          conv_stride=conv_stride,
          norm=norm,
          norm_eps=norm_eps,
          norm_momentum=norm_momentum,
          se_ratio=se_ratio,
          activation=activation,
          inner_act=inner_act
      ),
      activation = None if activation is None else activation(),

      branch_fusion = lambda x, y: x + y
  )


class ResStage(nn.Module):

  def __init__(self, res_blocks: nn.ModuleList) -> nn.Module:
    super(ResStage, self).__init__()
    self.res_blocks = res_blocks
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for _, res_block in enumerate(self.res_blocks):
      x = res_block(x)
      
    return x


def round_repeats(repeats, multiplier):

  if not multiplier:
    return repeats
  return int(math.ceil(repeats * multiplier))


def create_x3d_res_stage(
    # Stage configs
    depth: int,
    # Bottle Block Configs
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    bottleneck: Callable = create_x3d_bottleneck_block,
    # Conv Configs
    conv_kernel_size: Tuple[int] = (3, 3, 3),
    conv_stride: Tuple[int] = (1, 2, 2),
    # Norm Configs
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    se_ratio: float = 0.0625,
    # Activation configs.
    activation: Callable = nn.ReLU,
    inner_act: Callable = Swish,
) -> nn.Module:

  res_blocks = []
  for idx in range(depth):
    block = create_x3d_res_block(
        dim_in = dim_in if idx == 0 else dim_out,
        dim_inner = dim_inner,
        dim_out = dim_out,
        bottleneck = bottleneck,
        conv_kernel_size=conv_kernel_size,
        conv_stride=conv_stride if idx == 0 else (1, 1, 1),
        norm = norm,
        norm_eps = norm_eps,
        norm_momentum = norm_momentum,
        se_ratio=(se_ratio if (idx + 1) % 2 else 0.0),
        activation=activation,
        inner_act=inner_act,
    )

    res_blocks.append(block)
  
  return ResStage(res_blocks=nn.ModuleList(res_blocks))


class Net(nn.Module):
  def __init__(self, blocks: nn.ModuleList):
    super(Net, self).__init__()
    self.blocks = blocks
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for idx in range(len(self.blocks)):
      x = self.blocks[idx](x)

    return x

def create_x3d(
    input_channel: int = 3,
    input_clip_length: int = 13,
    input_crop_size: int = 160,
    # Model Configs
    model_num_class: int = 400,
    dropout_rate: float = 0.5,
    width_factor: float = 2.0,  
    depth_factor: float = 2.2,  
    # Normalization configs.
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 0.1,
    norm_momentum: float = 0.1,
    # Activation configs.
    activation: Callable = nn.ReLU,
    # Stem Configs
    stem_dim_in: int = 12,
    stem_conv_kernel_size: Tuple[int] = (5, 3, 3),
    stem_conv_stride: Tuple[int] = (1, 2, 2),
    # Stage configs.
    stage_conv_kernel_size: Tuple[Tuple[int]] = (
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
        (3, 3, 3),
    ),
    stage_spatial_stride: Tuple[int] = (2, 2, 2, 2),
    stage_temporal_stride: Tuple[int] = (1, 1, 1, 1),
    bottleneck: Callable = create_x3d_bottleneck_block,
    bottleneck_factor: float = 2.25,
    se_ratio: float = 0.0625,
    inner_act: Callable = Swish,
    # Head configs.
    #head_dim_out: int = 2048,
    head_dim_out: int = 1024,
    head_pool_act: Callable = nn.ReLU,
    head_bn_lin5_on: bool = False,
    head_activation: Callable = nn.Softmax,
    head_output_with_global_average: bool = True,
) -> nn.Module:
  blocks = []
  
  
 
  stem_dim_out = round_width(stem_dim_in, width_factor) # 24
  stem = create_x3d_stem(
      in_channels = input_channel,
      out_channels = stem_dim_out,
      conv_kernel_size = stem_conv_kernel_size,
      conv_stride = stem_conv_stride,
      conv_padding=[size // 2 for size in stem_conv_kernel_size],
      norm=norm,
      norm_eps=norm_eps,
      norm_momentum=norm_momentum,
      activation=activation,
  )

  blocks.append(stem)

  # Compute the depth and dimension for each stage  
  stage_depths = [1, 2, 5, 3]

  exp_stage = 2.0
  stage_dim1 = stem_dim_in # 12
  stage_dim2 = round_width(stage_dim1, exp_stage, divisor = 8) # 24
  stage_dim3 = round_width(stage_dim2, exp_stage, divisor = 8) # 48
  stage_dim4 = round_width(stage_dim3, exp_stage, divisor=8) # 96
  stage_dims = [stage_dim1, stage_dim2, stage_dim3, stage_dim4] # 12, 24, 48, 96

  # print(stage_dim1, stage_dim2, stage_dim3, stage_dim4)

  dim_in = stem_dim_out

  for idx in range(len(stage_dims)):
    dim_out = round_width(stage_dims[idx], width_factor) # 24, 48, 96, 192
    # print(dim_out)
    dim_inner = int(bottleneck_factor * dim_out) # 54, 108, 216, 432
    # print(dim_inner)
    depth = round_repeats(stage_depths[idx], depth_factor) # 3, 5, 11, 7
    # print(depth)

    stage_conv_stride = (
        stage_temporal_stride[idx],
        stage_spatial_stride[idx],
        stage_spatial_stride[idx],
    ) # (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)
    # print(stage_conv_stride)

    stage = create_x3d_res_stage(
        depth=depth,
        dim_in=dim_in,
        dim_inner=dim_inner,
        dim_out=dim_out,
        bottleneck=bottleneck,
        conv_kernel_size=stage_conv_kernel_size[idx],
        conv_stride=stage_conv_stride,
        norm=norm,
        norm_eps=norm_eps,
        norm_momentum=norm_momentum,
        se_ratio=se_ratio,
        activation=activation,
        inner_act=inner_act,
    )

    blocks.append(stage)
    dim_in = dim_out
  
  # return nn.ModuleList(blocks)

  # Create head for X3D.
  total_spatial_stride = stem_conv_stride[1] * np.prod(stage_spatial_stride) # 32
  total_temporal_stride = stem_conv_stride[0] * np.prod(stage_temporal_stride) # 1
  
  assert (
      input_clip_length >= total_temporal_stride
  ), "Clip length doesn't match temporal stride!"
  
  assert (
      input_crop_size >= total_spatial_stride
  ), "Crop size doesn't match spatial stride!"

  head_pool_kernel_size = (
      input_clip_length // total_temporal_stride,
      int(math.ceil(input_crop_size / total_spatial_stride)),
      int(math.ceil(input_crop_size / total_spatial_stride))
  ) # (13, 5, 5)
  
  head = create_x3d_head(
      dim_in = dim_out,
      dim_inner = dim_inner,
      dim_out = head_dim_out,
      num_classes = model_num_class,
      pool_act = head_pool_act,
      pool_kernel_size = head_pool_kernel_size,
      norm = norm,
      norm_eps = norm_eps,
      norm_momentum = norm_momentum,
      bn_lin5_on = head_bn_lin5_on,
      dropout_rate = dropout_rate,
      activation = head_activation,
      output_with_global_average = head_output_with_global_average
  )

  # blocks.append(head)
  # block_head = []
  # block_head.append(head)

  # return nn.ModuleList(block_head)

  blocks.append(head)

  return Net(blocks = nn.ModuleList(blocks))


def create_x3d_head(
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    num_classes: int,
    # Pooling Configs
    pool_act: Callable = nn.ReLU,
    pool_kernel_size: Tuple[int] = (13, 5, 5),
    # BN Configs
    norm: Callable = nn.BatchNorm3d,
    norm_eps: float = 1e-5,
    norm_momentum: float = 0.1,
    bn_lin5_on = False,
    # Dropout configs.
    dropout_rate: float = 0.5,
    # Activation configs.
    activation: Callable = nn.Softmax,
    # Output configs.
    output_with_global_average: bool = True,
) -> nn.Module:

  pre_conv_module = nn.Conv3d(
      in_channels = dim_in, out_channels = dim_inner, kernel_size = (1, 1, 1), bias = False
  )
  pre_norm_module = norm(num_features = dim_inner, eps = norm_eps, momentum = norm_momentum)
  pre_act_module = None if pool_act is None else pool_act()


  if pool_kernel_size is None:
    pool_module = nn.AdaptiveAvgPool3d((1, 1, 1))
  else:
    pool_module = nn.AvgPool3d(pool_kernel_size, stride = 1)


  post_conv_module = nn.Conv3d(
      in_channels = dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False
  ) 
  if bn_lin5_on:
    post_norm_module = norm(
      num_features = dim_out, eps = norm_eps, momentum = norm_momentum
    )
  else:
    post_norm_module = None
  # post_act_module = None if pool_act is None else pool_act() # Sửa ở đây
  post_act_module = None

  projected_pool_module = ProjectedPool(
    pre_conv = pre_conv_module,
    pre_norm = pre_norm_module,
    pre_act = pre_act_module,
    pool = pool_module,
    post_conv = post_conv_module,
    post_norm = post_norm_module,
    post_act = post_act_module,
  )

  if activation is None:
    activation_module = None
  elif activation == nn.Softmax:
    activation_module = activation(dim=1)
  elif activation == nn.Sigmoid:
    activation_module = activation()
  else:
    raise NotImplementedError(
        "{} is not supported as an activation" "function.".format(activation)
    )
    
  if output_with_global_average:
    output_pool = nn.AdaptiveAvgPool3d(1)
  else:
    output_pool = None
  
  # return ResNetBasicHead(
  #     proj = nn.Linear(dim_out, num_classes, bias=True),
  #     activation = activation_module,
  #     pool = projected_pool_module,
  #     dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None,
  #     output_pool = output_pool,
  # ) # Sửa ở đây
  

  return ResNetBasicHead(
      pool = projected_pool_module,
  )


class ProjectedPool(nn.Module):

  def __init__(
      self,
      pre_conv: nn.Module = None,
      pre_norm: nn.Module = None,
      pre_act: nn.Module = None,
      pool: nn.Module = None,
      post_conv: nn.Module = None,
      post_norm: nn.Module = None,
      post_act: nn.Module = None,
  ):

    super(ProjectedPool, self).__init__()
    self.pre_conv = pre_conv
    self.pre_norm = pre_norm
    self.pre_act = pre_act

    self.pool = pool

    self.post_conv = post_conv
    self.post_norm = post_norm
    self.post_act = post_act

  def forward(self, x):

    x = self.pre_conv(x)
    if self.pre_norm is not None:
      x = self.pre_norm(x)
    if self.pre_act is not None:
      x = self.pre_act(x)
    

    x = self.pool(x)
    

    x = self.post_conv(x)
    if self.post_norm is not None:
      x = self.post_norm(x)
    if self.post_act is not None:
      x = self.post_act(x)
    
    return x


class ResNetBasicHead(nn.Module):

  def __init__(
    self,
    pool: nn.Module = None,
    dropout: nn.Module = None,
    proj: nn.Module = None,
    activation: nn.Module = None,
    output_pool: nn.Module = None,
  ):

    super(ResNetBasicHead, self).__init__()
    self.pool = pool
    self.dropout = dropout
    self.proj = proj
    self.activation = activation
    self.output_pool = output_pool
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:

    if self.pool is not None:
      x = self.pool(x)
    

    if self.dropout is not None:
      x = self.dropout(x)

    if self.proj is not None:
      x = x.permute((0, 2, 3, 4, 1))
      x = self.proj(x)
      x = x.permute((0, 4, 1, 2, 3))
    

    if self.activation is not None:
      x = self.activation(x)


    if self.output_pool is not None:
      x = self.output_pool(x)

    x = x.squeeze(-1).squeeze(-1).squeeze(-1)
    
    return x

if __name__ == "__main__":
    model = create_x3d(
        input_clip_length=16, 
        input_crop_size=224, 
        width_factor=2.0,
        depth_factor=2.2,
        head_dim_out=2048,
        model_num_class=400, 
    ) # X3D_M
    
    pretrained_path = "X3D_M_extract_features.pth"
    
    try:
        # 尝试加载模型
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
        print("加载模型成功！")
    except RuntimeError as e:
        print(f"加载模型时出错: {e}")
        print("尝试部分加载状态字典...")
        
        # 部分加载 - 忽略缺失键
        state_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        
        # 过滤掉不匹配的键
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
        if missing_keys:
            print(f"模型中以下键在预训练状态字典中缺失: {missing_keys}")
            
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"成功加载了 {len(pretrained_dict)}/{len(state_dict)} 个参数")
    
    # 测试不同输入
    x = torch.randn(2, 3, 16, 224, 224)  # (B, C, T, H, W)
    out = model(x)
    print(f"输出张量形状: {out.shape}")  # 应该是(B, feature_dim)
    
    print("\n不同X3D模型配置：")
    print("X3D_S: width_factor=1.0, depth_factor=1.0, input_size=160")
    print("X3D_M: width_factor=2.0, depth_factor=2.2, input_size=224")
    print("X3D_L: width_factor=2.0, depth_factor=5.0, input_size=312")
    print("X3D_XL: width_factor=2.0, depth_factor=5.0, input_size=312, bottleneck_factor=2.25")
    
