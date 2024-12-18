import jax.numpy as jnp
from flax import linen as nn
from flax.serialization import from_bytes

class InstanceNormalization(nn.Module):
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        mean = jnp.mean(x, axis=(1,2), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(1,2), keepdims=True)
        gamma = self.param('gamma', nn.initializers.ones, (1,1,1,x.shape[-1]))
        beta = self.param('beta', nn.initializers.zeros, (1,1,1,x.shape[-1]))
        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)
        return gamma * x_norm + beta

class ResidualBlock(nn.Module):
  num_filters: int

  @nn.compact
  def __call__(self, x):
    residual = x
    y = nn.Conv(features = self.num_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x)
    y = InstanceNormalization()(y)
    y = nn.relu(y)
    y = nn.Conv(features = self.num_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(y)
    y = InstanceNormalization()(y)
    if residual.shape != y.shape:
      residual = nn.Conv(features = self.num_filters, kernel_size = (1, 1), strides = (1, 1), padding = 'SAME')(residual)
      residual = InstanceNormalization()(residual)

    return nn.relu(residual + y)

class Generator(nn.Module):
  num_filters: int = 64
  num_blocks: int = 9

  @nn.compact
  def __call__(self, x):
    # Initial convolution
    x = nn.Conv(features=self.num_filters, kernel_size=(7, 7), strides=(1, 1), padding='SAME')(x)
    x = InstanceNormalization()(x)
    x = nn.relu(x)

    # Downsampling
    x = nn.Conv(features=self.num_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
    x = InstanceNormalization()(x)
    x = nn.relu(x)

    x = nn.Conv(features=self.num_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
    x = InstanceNormalization()(x)
    x = nn.relu(x)

    # Residual blocks
    for _ in range(self.num_blocks):
      x = ResidualBlock(num_filters=self.num_filters*4)(x)

    # Upsampling (fractionally-strided convolutions)
    x = nn.ConvTranspose(features=self.num_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
    x = InstanceNormalization()(x)
    x = nn.relu(x)
    x = nn.ConvTranspose(features=self.num_filters, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
    x = InstanceNormalization()(x)
    x = nn.relu(x)

    # Final convolution to RGB
    x = nn.Conv(features=3, kernel_size=(7, 7), strides=(1, 1), padding='SAME')(x)
    x = jnp.tanh(x)

    return x

def load_model(params_path, key):
    model = Generator()
    dummy_input = jnp.ones((1, 256, 256, 3))
    params = model.init(key, dummy_input)
    with open(params_path, "rb") as f:
        params = from_bytes(params, f.read())
    return model, params