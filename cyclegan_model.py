import os
import jax
import glob
import flax
import optax
import random
import warnings
import numpy as np
import jax.numpy as jnp
import flax.serialization
import matplotlib.pyplot as plt
from jax import jit, grad
from PIL import Image
from tqdm import tqdm
from flax import linen as nn

warnings.filterwarnings('ignore')

print("Default device:", jax.devices()[0])
print("Default backend:", jax.default_backend())

BASE_PATH = 'PATH_DSET'

"""## Preparation"""

class CONFIG:
    lr = 2e-4 # learning rate
    lambda_cycle = 10.0 # weight for cycle consistency loss
    lambda_identity = 0.5 # weight for identity loss
    bs = 4 # batch_size
    num_epochs = 50
    imgsz = (256,256,3)

names_folder = os.listdir(BASE_PATH)
for i in range(len(names_folder)):
    if os.path.isdir(os.path.join(BASE_PATH, names_folder[i])):
        print(f'The folder {names_folder[i]} contains {len(os.listdir(os.path.join(BASE_PATH, names_folder[i])))} files')


TRAIN_MONET = os.path.join(BASE_PATH, 'trainA')
TRAIN_PHOTOS = os.path.join(BASE_PATH, 'trainB')

# Visualization
_, axes = plt.subplots(
    nrows = 2,
    ncols = 6,
    figsize = (12, 6),
    facecolor = 'white',
)

for i, ax in enumerate(axes[0]):
    files = os.listdir(TRAIN_MONET)
    path_monet = np.random.choice(files)
    img_monet = Image.open(os.path.join(TRAIN_MONET, path_monet))
    ax.imshow(img_monet)
    ax.set_title(f'Monet style_{i+1}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

for i, ax in enumerate(axes[1]):
    files = os.listdir(TRAIN_PHOTOS)
    path_photos = np.random.choice(files)
    img_photos = Image.open(os.path.join(TRAIN_PHOTOS, path_photos))
    ax.imshow(img_photos)
    ax.set_title(f'Real photos_{i+1}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('photo2monet.png')
plt.show()
print(f'SHAPE Monet photos: {np.array(img_monet).shape}')
print(f'SHAPE Real photos: {np.array(img_photos).shape}')

# Store data into a list
monet_paths = glob.glob(os.path.join(BASE_PATH, "trainA", "*.jpg"))
photo_paths = glob.glob(os.path.join(BASE_PATH, "trainB", "*.jpg"))
test_monet_paths = glob.glob(os.path.join(BASE_PATH, "testA", "*.jpg"))
test_photo_paths = glob.glob(os.path.join(BASE_PATH, "testB", "*.jpg"))

def denormalize(img):
  img = (img + 1.0)/2.0
  img = np.clip(img, 0, 1)
  return img

def load_and_preprocess_image(path):
  img = Image.open(path).convert('RGB')
  img = img.resize(CONFIG.imgsz[:2])
  img = np.array(img).astype(np.float32) / 127.5 - 1.0
  return img

def create_dataset(A_paths, B_paths, batch_size=4):
  random.shuffle(A_paths)
  random.shuffle(B_paths)

  lenA = len(A_paths)
  lenB = len(B_paths)
  max_len = max(lenA, lenB)

  for i in tqdm(range(0, max_len, batch_size), desc = 'Process dataset'):
    A_indices = range(i, i+batch_size)
    B_indices = range(i, i+batch_size)

    A_batch_paths = [A_paths[idx % lenA] for idx in A_indices]
    B_batch_paths = [B_paths[idx % lenB] for idx in B_indices]

    A_batch = [load_and_preprocess_image(p) for p in A_batch_paths]
    B_batch = [load_and_preprocess_image(p) for p in B_batch_paths]

    yield np.array(A_batch), np.array(B_batch)

"""## Modelisation"""

class InstanceNormalization(nn.Module):
  epsilon: float = 1e-5
  @nn.compact
  def __call__(self, x):
    mean = jnp.mean(x, axis=(1,2), keepdims = True)
    var = jnp.mean((x-mean)**2, axis=(1,2), keepdims=True)
    gamma = self.param('gamma', nn.initializers.ones, (1,1,1,x.shape[-1]))
    beta = self.param('beta', nn.initializers.zeros, (1,1,1,x.shape[-1]))
    x_norm = (x-mean) / jnp.sqrt(var + self.epsilon)
    return gamma * x_norm + beta

"""### Model building"""

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

class Discriminator(nn.Module):
  num_filters: int = 64

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=self.num_filters, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
    x = nn.leaky_relu(x, negative_slope=0.2)
    x = nn.Conv(features=self.num_filters*2, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
    x = InstanceNormalization()(x)
    x = nn.leaky_relu(x, negative_slope=0.2)
    x = nn.Conv(features=self.num_filters*4, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
    x = InstanceNormalization()(x)
    x = nn.leaky_relu(x, negative_slope=0.2)
    x = nn.Conv(features=1, kernel_size=(4, 4), strides=(1, 1), padding='SAME')(x)

    return x

# The img is in form of tuple so we convert to int
img_size = CONFIG.imgsz[:1]
img_size_int = int(img_size[0])

key = jax.random.PRNGKey(0)
x_dummy = jnp.ones((1, img_size_int, img_size_int, 3), dtype=jnp.float32)

G = Generator()  # A->B
F = Generator()  # B->A
Dx = Discriminator() # D->A
Dy = Discriminator() # D->B

params_G = G.init(key, x_dummy)
params_F = F.init(key, x_dummy)
params_Dx = Dx.init(key, x_dummy)
params_Dy = Dy.init(key, x_dummy)

lr = 2e-4
optim_G = optax.adam(lr, b1=0.5, b2=0.999)
optim_F = optax.adam(lr, b1=0.5, b2=0.999)
optim_Dx = optax.adam(lr, b1=0.5, b2=0.999)
optim_Dy = optax.adam(lr, b1=0.5, b2=0.999)

opt_state_G = optim_G.init(params_G)
opt_state_F = optim_F.init(params_F)
opt_state_Dx = optim_Dx.init(params_Dx)
opt_state_Dy = optim_Dy.init(params_Dy)

""" ### Loss"""

def mse_loss(pred, target):
  return jnp.mean((pred - target)**2)

def generator_loss(params_G, params_F, params_Dx, params_Dy, x_real, y_real):
  # x_real: A (Monet), y_real: B (Photo)
  y_fake = G.apply(params_G, x_real)
  x_recon = F.apply(params_F, y_fake)
  x_fake = F.apply(params_F, y_real)
  y_recon = G.apply(params_G, x_fake)

  Dy_pred_fake = Dy.apply(params_Dy, y_fake)
  Dx_pred_fake = Dx.apply(params_Dx, x_fake)

  loss_G_adv = mse_loss(Dy_pred_fake, jnp.ones_like(Dy_pred_fake))
  loss_F_adv = mse_loss(Dx_pred_fake, jnp.ones_like(Dx_pred_fake))
  loss_cycle = mse_loss(x_recon, x_real) + mse_loss(y_recon, y_real)

  total_loss = loss_G_adv + loss_F_adv + CONFIG.lambda_cycle * loss_cycle
  return total_loss

def discriminator_loss(params_D, apply_fn, real, fake):
    pred_real = apply_fn(params_D, real)
    pred_fake = apply_fn(params_D, fake)
    real_loss = mse_loss(pred_real, jnp.ones_like(pred_real))
    fake_loss = mse_loss(pred_fake, jnp.zeros_like(pred_fake))
    return (real_loss + fake_loss)*0.5

"""### Training stage"""

@jit
def train_step(params_G, params_F, params_Dx, params_Dy,
               opt_state_G, opt_state_F, opt_state_Dx, opt_state_Dy,
               x_real, y_real, key):
  # Compute generator loss
  total_g_f_loss = generator_loss(params_G, params_F, params_Dx, params_Dy, x_real, y_real)
  grads_G, grads_F = grad(generator_loss, argnums=(0,1))(params_G, params_F, params_Dx, params_Dy, x_real, y_real)
  updates_G, new_opt_state_G = optim_G.update(grads_G, opt_state_G)
  updates_F, new_opt_state_F = optim_F.update(grads_F, opt_state_F)
  new_params_G = optax.apply_updates(params_G, updates_G)
  new_params_F = optax.apply_updates(params_F, updates_F)

  # Update discriminators
  x_fake = F.apply(new_params_F, y_real)
  y_fake = G.apply(new_params_G, x_real)

  d_x_loss = discriminator_loss(params_Dx, Dx.apply, x_real, x_fake)
  grads_Dx = grad(discriminator_loss)(params_Dx, Dx.apply, x_real, x_fake)
  updates_Dx, new_opt_state_Dx = optim_Dx.update(grads_Dx, opt_state_Dx)
  new_params_Dx = optax.apply_updates(params_Dx, updates_Dx)

  d_y_loss = discriminator_loss(params_Dy, Dy.apply, y_real, y_fake)
  grads_Dy = grad(discriminator_loss)(params_Dy, Dy.apply, y_real, y_fake)
  updates_Dy, new_opt_state_Dy = optim_Dy.update(grads_Dy, opt_state_Dy)
  new_params_Dy = optax.apply_updates(params_Dy, updates_Dy)

  return (new_params_G, new_params_F, new_params_Dx, new_params_Dy,
            new_opt_state_G, new_opt_state_F, new_opt_state_Dx, new_opt_state_Dy,
            total_g_f_loss, d_x_loss, d_y_loss)


def predictions(epoch, params_G, params_F):
  # Choose images
  test_monet_path = random.choice(test_monet_paths)
  test_photo_path = random.choice(test_photo_paths)
  test_monet_img = load_and_preprocess_image(test_monet_path)
  test_photo_img = load_and_preprocess_image(test_photo_path)

  # add batches
  test_monet_jnp = jnp.expand_dims(test_monet_img, axis = 0)
  test_photo_jnp = jnp.expand_dims(test_photo_img, axis = 0)

  # Monet -> Photo
  y_fake = G.apply(params_G, test_monet_jnp)
  # Photo -> Monet
  x_fake = F.apply(params_F, test_photo_jnp)

  # denormalize
  test_monet_img = denormalize(test_monet_img)
  test_photo_img = denormalize(test_photo_img)
  y_fake_img = denormalize(np.array(y_fake[0]))
  x_fake_img = denormalize(np.array(x_fake[0]))

  # Vizualisation
  fig, axs = plt.subplots(2, 2, figsize=(8,5))
  axs[0, 0].imshow(test_monet_img)
  axs[0, 0].set_title('Monet style [Original]')
  axs[0, 1].imshow(test_photo_img)
  axs[0, 1].set_title('Real photo [Original]')
  axs[1, 0].imshow(y_fake_img)
  axs[1, 0].set_title('Monet -> Photo [Preds]')
  axs[1, 1].imshow(x_fake_img)
  axs[1, 1].set_title('Photo -> Monet [Preds]')

  for ax in axs.flat:
    ax.axis('off')

  output_dir = "results"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  fig.suptitle(f'Epoch {epoch}')
  plt.tight_layout()
  output_path = os.path.join(output_dir, f'epoch_{epoch}.png')
  plt.savefig(output_path)
  plt.show()

# History
history = {
    'g_f_losses': [],
    'dx_losses': [],
    'dy_losses': []
}

# training model
for epoch in tqdm(range(CONFIG.num_epochs)):
  dataset_iter = create_dataset(monet_paths, photo_paths, batch_size=CONFIG.bs)
  epoch_g_f_loss = []
  epoch_dx_loss = []
  epoch_dy_loss = []

  for x_real, y_real in dataset_iter:
    x_real = jnp.array(x_real)
    y_real = jnp.array(y_real)
    (params_G, params_F, params_Dx, params_Dy,
         opt_state_G, opt_state_F, opt_state_Dx, opt_state_Dy,
         total_g_f_loss, d_x_loss, d_y_loss) = train_step(
            params_G, params_F, params_Dx, params_Dy,
            opt_state_G, opt_state_F, opt_state_Dx, opt_state_Dy,
            x_real, y_real, key)

    epoch_g_f_loss.append(float(total_g_f_loss))
    epoch_dx_loss.append(float(d_x_loss))
    epoch_dy_loss.append(float(d_y_loss))

  # Calculate the mean of the loss
  mean_g_f_loss = np.mean(epoch_g_f_loss)
  mean_dx_loss = np.mean(epoch_dx_loss)
  mean_dy_loss = np.mean(epoch_dy_loss)

  # Store losses in history
  history['g_f_losses'].append(mean_g_f_loss)
  history['dx_losses'].append(mean_dx_loss)
  history['dy_losses'].append(mean_dy_loss)

  print(f"Epoch {epoch+1}, G+F Loss: {total_g_f_loss:.4f}, Dx Loss: {d_x_loss:.4f}, Dy Loss: {d_y_loss:.4f}")
  if (epoch+1) == 1 or (epoch+1) % 10 == 0:
    predictions(epoch+1, params_G, params_F)

"""### Checkpoint"""

save_dir = "checkpoints"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

params_G_path = os.path.join(save_dir, "params_G.msgpack")
params_G_bytes = flax.serialization.to_bytes(params_G)
with open(params_G_path, "wb") as f:
    f.write(params_G_bytes)

params_F_path = os.path.join(save_dir, "params_F.msgpack")
params_F_bytes = flax.serialization.to_bytes(params_F)
with open(params_F_path, "wb") as f:
    f.write(params_F_bytes)

print(f"Generators parameters saved at {save_dir}")

"""### History losses"""

g_f_losses = history['g_f_losses']
dx_losses = history['dx_losses']
dy_losses = history['dy_losses']

plt.figure(figsize=(7, 4))
plt.plot(g_f_losses, label = 'Generators G&F Loss', color = 'blue', linewidth = 2, marker = '*')
plt.title('CycleGAN Training Generators Losses Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('results/cyclegan gen_loss')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dx_losses, label='Dx Loss', color='red', linewidth=3)
plt.plot(dy_losses, label='Dy Loss', color='blue', linewidth=3)

plt.title('CycleGAN Training Discriminator Losses Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('results/cyclegan disc_loss')
plt.show()

"""## Prediction"""

def load_images_from_dir(directory, n=4):
  paths = glob.glob(os.path.join(directory, "*.jpg"))
  chosen_paths = random.sample(paths, min(n, len(paths)))
  images = [load_and_preprocess_image(p) for p in chosen_paths]
  return chosen_paths, np.array(images)

def predict_and_show(params_G = params_G, params_F = params_F,
                     monet_dir = TRAIN_MONET, photo_dir = TRAIN_PHOTOS,
                     mode="photo2monet", n=4, pair_mode=False):
  """
  mode can be:
  - "photo2monet": take n images from photo_dir, show original and translated (Photo->Monet).
  - "monet2photo": take n images from monet_dir, show original and translated (Monet->Photo).
  - "random": randomly choose images from either monet_dir or photo_dir and apply the corresponding translation.

  If pair_mode=True, we show pairs:
  - 1st row: Photo [Original] and Photo->Monet [Pred]
  - 2nd row: Monet [Original] and Monet->Photo [Pred]
  alternating domains per row.
  """

  if pair_mode:
    # In pair mode, we take n images total, half from photo_dir and half from monet_dir
    n_half = n // 2
    photo_paths, photo_imgs = load_images_from_dir(photo_dir, n_half)
    monet_paths, monet_imgs = load_images_from_dir(monet_dir, n_half)

    # Display in pairs:
    # For each photo, show Photo [Original] and Photo->Monet [Pred]
    # For each Monet, show Monet [Original] and Monet->Photo [Pred]
    fig, axs = plt.subplots(n_half*2, 2, figsize=(8, 4*n_half))
    for i in range(n_half):
      # Photo -> Monet (use F)
      ph = jnp.array(photo_imgs[i:i+1])
      monet_pred = F.apply(params_F, ph)
      axs[2*i,0].imshow(denormalize(photo_imgs[i]))
      axs[2*i,0].set_title("Photo [Original]")
      axs[2*i,1].imshow(denormalize(np.array(monet_pred[0])))
      axs[2*i,1].set_title("Photo -> Monet [Pred]")

      # Monet -> Photo (use G)
      mo = jnp.array(monet_imgs[i:i+1])
      photo_pred = G.apply(params_G, mo)
      axs[2*i+1,0].imshow(denormalize(monet_imgs[i]))
      axs[2*i+1,0].set_title("Monet [Original]")
      axs[2*i+1,1].imshow(denormalize(np.array(photo_pred[0])))
      axs[2*i+1,1].set_title("Monet -> Photo [Pred]")

    for ax in axs.flat:
      ax.axis('off')
    plt.tight_layout()
    plt.savefig('results/pair_mode.png')
    plt.show()

  else:
    # Normal mode (not pair)
    if mode == "photo2monet":
      paths, imgs = load_images_from_dir(photo_dir, n)
      preds = F.apply(params_F, jnp.array(imgs))
      original_title = "Photo [Original]"
      pred_title = "Photo -> Monet [Pred]"

    elif mode == "monet2photo":
      paths, imgs = load_images_from_dir(monet_dir, n)
      preds = G.apply(params_G, jnp.array(imgs))
      original_title = "Monet [Original]"
      pred_title = "Monet -> Photo [Pred]"

    elif mode == "random":
      # Randomly pick images and apply corresponding translations
      imgs_list = []
      preds_list = []
      original_titles = []
      pred_titles = []
      chosen_paths = []

      for _ in range(n):
        if random.random() < 0.5:
          # Take a photo and apply Photo->Monet (F)
          p_path, p_imgs = load_images_from_dir(photo_dir, 1)
          imgs_list.append(p_imgs[0])
          pred = F.apply(params_F, jnp.array(p_imgs))
          preds_list.append(np.array(pred[0]))
          original_titles.append("Photo [Original]")
          pred_titles.append("Photo -> Monet [Pred]")
          chosen_paths += p_path
        else:
          # Take a Monet and apply Monet->Photo (G)
          m_path, m_imgs = load_images_from_dir(monet_dir, 1)
          imgs_list.append(m_imgs[0])
          pred = G.apply(params_G, jnp.array(m_imgs))
          preds_list.append(np.array(pred[0]))
          original_titles.append("Monet [Original]")
          pred_titles.append("Monet -> Photo [Pred]")
          chosen_paths += m_path

      imgs = np.array(imgs_list)
      preds = np.array(preds_list)
    else:
      raise ValueError("Unknown mode. Choose among 'photo2monet', 'monet2photo', or 'random'.")

    # Display the results
    if mode != "random":
      fig, axs = plt.subplots(n, 2, figsize=(8,4*n))
      for i in range(n):
        axs[i,0].imshow(denormalize(imgs[i]))
        axs[i,0].set_title(original_title)
        axs[i,1].imshow(denormalize(np.array(preds[i])))
        axs[i,1].set_title(pred_title)
        for ax in axs[i,:]:
          ax.axis('off')
      plt.tight_layout()
      plt.savefig('results/vizu')
      plt.show()
    else:
      # Random mode
      fig, axs = plt.subplots(n, 2, figsize=(8,4*n))
      for i in range(n):
        axs[i,0].imshow(denormalize(imgs[i]))
        axs[i,0].set_title(original_titles[i])
        axs[i,1].imshow(denormalize(preds[i]))
        axs[i,1].set_title(pred_titles[i])
        for ax in axs[i,:]:
          ax.axis('off')
      plt.tight_layout()
      plt.savefig('results/vizu')
      plt.show()

predict_and_show()

predict_and_show(mode='monet2photo')

predict_and_show(n = 6, pair_mode=True)

