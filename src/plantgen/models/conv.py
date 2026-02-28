import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    pass


class StandardConv(ConvBlock):
    """
    Simple convolutional block with a convolution layer, and optional batchnorm and activation fuction
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            act_fn=None
        ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding='same'
        )
        self.bn = nn.BatchNorm2d(out_channels) if act_fn is not None else None
        self.act_fn = act_fn if act_fn is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.act_fn is not None:
            x = self.bn(x)
            x = self.act_fn(x)
        return x


class DepthwiseSeparableConv(ConvBlock):
    """
    Depthwise separable convolutional block with a convolution layer, and optional batchnorm and activation fuction
    https://arxiv.org/abs/1704.04861
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            act_fn=F.relu
        ):

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding='same'
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')

        self.batchnorm = nn.BatchNorm2d(in_channels)

        self.act_fn = act_fn

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.act_fn(x)

        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class ResnetBlock(ConvBlock):
    """
    Residual convolutional block
    https://arxiv.org/abs/1512.03385
    """

    def __init__(
            self,
            block: nn.Module,
        ):

        super().__init__()

        self.block = block

    def forward(self, x):
        res = x
        x = self.block(x)
        return res + x


class DownsampleBlock(ConvBlock):
    """
    Simple downsampling block with a maxpooling layer, and optional batchnorm and activation fuction
    """

    def __init__(
            self,
            block: nn.Module,
            out_channels,
            act_fn=F.relu
        ):

        super().__init__()

        self.block = block
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.downsample = nn.MaxPool2d(2)

        self.act_fn = act_fn

    def forward(self, x):
        x = self.block(x)
        x = self.batchnorm(x)
        x = self.act_fn(x)
        x = self.downsample(x)
        return x


class UpsampleBlock(ConvBlock):
    """
    Simple upsampling block with a bilinear upsampling layer, and optional batchnorm and activation fuction
    """
    def __init__(
            self,
            block: nn.Module,
            out_channels,
            act_fn=F.relu
        ):

        super().__init__()

        self.block = block
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.act_fn = act_fn

    def forward(self, x):
        x = self.upsample(x)
        x = self.block(x)
        x = self.batchnorm(x)
        x = self.act_fn(x)
        return x


class ChannelFirstLayerNorm(nn.Module):
    """
    Wrapper of torch's LayerNorm that operates on channel-first data (images).
    """

    def __init__(self, num_channels, eps):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


class ConvNextBlock(nn.Module):
    """
    ConvNext block as described in:
    https://arxiv.org/abs/2201.03545

    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding='same', groups=dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, padding='same')
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1, padding='same')

        self.ln = ChannelFirstLayerNorm(dim, eps=eps)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor):
        res = x

        x = self.dw(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act_fn(x)
        x = self.pw2(x)

        return res + x


class ConvNextEncoder(nn.Module):
    """
    Encoder for an autoencoder based on ConvNext blocks (https://arxiv.org/abs/2201.03545), with downsampling 
    layers (strided convolutions) between stages.
    """
    def __init__(
            self,
            in_channels: int,
            depths: list[int] = [3, 3, 9],
            dims: list[int] = [96, 192, 384],
            ln_eps: float = 1e-6):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=3, padding='same'),
                ChannelFirstLayerNorm(dims[0], eps=ln_eps)
            )
        )

        for i, dim in enumerate(dims):
            for _ in range(depths[i]):
                self.layers.append(ConvNextBlock(dim, eps=ln_eps))
            if i < len(dims) - 1:
                self.layers.append(
                    nn.Sequential(
                        ChannelFirstLayerNorm(dim, eps=ln_eps),
                        nn.Conv2d(dim, dims[i+1], kernel_size=2, stride=2)
                    )
                )

        self.in_channels = in_channels
        self.depths = depths
        self.dims = dims
        self.ln_eps = ln_eps

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNextDecoder(nn.Module):
    """
    Decoder for an autoencoder based on ConvNext blocks (https://arxiv.org/abs/2201.03545), with 
    upsampling layers (transposed convolutions) between stages.
    """
    def __init__(
            self,
            latent_dim: int,
            out_channels: int,
            depths: list[int] = [9, 3, 3],
            dims: list[int] = [384, 192, 96],
            ln_eps: float = 1e-6):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(latent_dim, dims[0], kernel_size=3, padding='same'),
                ChannelFirstLayerNorm(dims[0], eps=ln_eps)
            )
        )

        for i, dim in enumerate(dims):
            for _ in range(depths[i]):
                self.layers.append(ConvNextBlock(dim, eps=ln_eps))

            output_dim = dims[i+1] if i < len(dims) - 1 else dims[i]

            if i < len(dims) - 1:
                self.layers.append(
                    nn.Sequential(
                        ChannelFirstLayerNorm(dim, eps=ln_eps),
                        nn.ConvTranspose2d(dim, output_dim, kernel_size=2, stride=2)
                    )
                )

        self.layers.append(
            nn.Sequential(
                ChannelFirstLayerNorm(dims[-1], eps=ln_eps),
                nn.Conv2d(dims[-1], out_channels, kernel_size=3, padding='same')
            )
        )

        self.latent_dim = latent_dim
        self.depths = depths
        self.dims = dims
        self.ln_eps = ln_eps

        nn.init.zeros_(self.layers[-1][-1].weight)
        nn.init.zeros_(self.layers[-1][-1].bias)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
