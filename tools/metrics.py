from PIL import Image
import torch
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

# Image paths
classical_path = r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\samples classical\x0_0.png"
fqconv_path = r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\samples vertex fqconv\x0_0.png"
hqconv_path = r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\samples vertex hqconv\x0_0.png"

classical_path_fmnist = r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\samples classical fmnist\x0_0.png"
fqconv_path_fmnist = r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\samples vertex fqconv fmnist\x0_0.png"
hqconv_path_fmnist = r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\samples vertex hqconv fmnist\x0_0.png"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

torch.manual_seed(42)

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
fmnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

real_batch = torch.stack([mnist[i][0] for i in range(9)])
real_batch_fmnist = torch.stack([fmnist[i][0] for i in range(9)])

def split_grid_image(grid_image, img_size=(28, 28), grid_size=(3, 3), padding=2):
    """
    Splits a PIL grid image back into individual images.

    Args:
        grid_image (PIL.Image): the composite image
        img_size (tuple): size (width, height) of each individual image
        grid_size (tuple): number of images per row and column (cols, rows)
        padding (int): pixels between images, if any

    Returns:
        torch.Tensor: Tensor of shape [N, 1, H, W] containing the recovered images
    """
    img_w, img_h = img_size
    cols, rows = grid_size
    images = []
    transform = transforms.ToTensor()

    for row in range(rows):
        for col in range(cols):
            left = col * (img_w + padding)
            upper = row * (img_h + padding)
            right = left + img_w
            lower = upper + img_h

            img = grid_image.crop((left, upper, right, lower))
            img_tensor = transform(img)
            images.append(img_tensor)

    return torch.stack(images)

# MNIST Classical
classical_img_grid = Image.open(classical_path).convert("L")
classical_img = split_grid_image(classical_img_grid, grid_size=(10, 10))
classical_batch = classical_img [:9]

# FashionMNIST Classical
classical_img_grid_fmnist = Image.open(classical_path_fmnist).convert("L")
classical_img_fmnist = split_grid_image(classical_img_grid_fmnist, grid_size=(10, 10))
classical_batch_fmnist = classical_img_fmnist[:9]

#FID Model for MNIST
fid = FrechetInceptionDistance(feature=2048)
real_batch = real_batch.repeat(1, 3, 1, 1).to(torch.uint8)
classical_batch = classical_batch.repeat(1, 3, 1, 1).to(torch.uint8)
fid.update(real_batch, real=True)
fid.update(classical_batch, real=False)
fid = fid.compute()

# KID for MNIST
metric = KernelInceptionDistance(subsets=3, subset_size=9)
metric.update(real_batch, real=True)
metric.update(classical_batch, real=False)
kid = metric.compute()

print("Classical Model Results:")
print("------------------------")
print(f"FID score classical MNIST: {fid}")
print(f"KID classical MNIST mean: {kid[0]:.9f}")
print(f"KID classical MNIST std: {kid[1]:.9f}")

#FID Model for FashionMNIST
fid = FrechetInceptionDistance(feature=2048)
real_batch_fmnist = real_batch_fmnist.repeat(1, 3, 1, 1).to(torch.uint8)
classical_batch_fmnist = classical_batch_fmnist.repeat(1, 3, 1, 1).to(torch.uint8)
fid.update(real_batch_fmnist, real=True)
fid.update(classical_batch_fmnist, real=False)
fid = fid.compute()

# KID for FashionMNIST
metric = KernelInceptionDistance(subsets=3, subset_size=9)
metric.update(real_batch_fmnist, real=True)
metric.update(classical_batch_fmnist, real=False)
kid = metric.compute()

print(f"FID score classical FashionMNIST: {fid}")
print(f"KID classical FashionMNIST mean: {kid[0]:.9f}")
print(f"KID classical FashionMNIST std: {kid[1]:.9f}")
print()

# MNIST FQConv
fqconv_img_grid = Image.open(fqconv_path).convert("L")
fqconv_batch = split_grid_image(fqconv_img_grid, grid_size=(3, 3))

# FashionMNIST FQConv
fqconv_img_grid_fmnist = Image.open(fqconv_path_fmnist).convert("L")
fqconv_batch_fmnist = split_grid_image(fqconv_img_grid_fmnist, grid_size=(3, 3))

#FID Model for MNIST
fid = FrechetInceptionDistance(feature=2048)
fqconv_batch = fqconv_batch.repeat(1, 3, 1, 1).to(torch.uint8)
fid.update(real_batch, real=True)
fid.update(fqconv_batch, real=False)
fid = fid.compute()

# KID for MNIST
metric = KernelInceptionDistance(subsets=3, subset_size=9)
metric.update(real_batch, real=True)
metric.update(fqconv_batch, real=False)
kid = metric.compute()

print("FQConv Model Results:")
print("--------------------")
print(f"FID score fqconv MNIST: {fid}")
print(f"KID fqconv MNIST mean: {kid[0]:.9f}")
print(f"KID fqconv MNIST std: {kid[1]:.9f}")

#FID Model for FashionMNIST
fid = FrechetInceptionDistance(feature=2048)
fqconv_batch_fmnist = fqconv_batch_fmnist.repeat(1, 3, 1, 1).to(torch.uint8)
fid.update(real_batch_fmnist, real=True)
fid.update(fqconv_batch_fmnist, real=False)
fid = fid.compute()

# KID for FashionMNIST
metric = KernelInceptionDistance(subsets=3, subset_size=9)
metric.update(real_batch_fmnist, real=True)
metric.update(fqconv_batch_fmnist, real=False)
kid = metric.compute()

print(f"FID score fqconv FashionMNIST: {fid}")
print(f"KID fqconv FashionMNIST mean: {kid[0]:.9f}")
print(f"KID fqconv FashionMNIST std: {kid[1]:.9f}")
print()

# MNIST HQConv
hqconv_img_grid = Image.open(hqconv_path).convert("L")
hqconv_batch = split_grid_image(hqconv_img_grid, grid_size=(3, 3))

# FashionMNIST HQConv
hqconv_img_grid_fmnist = Image.open(hqconv_path_fmnist).convert("L")
hqconv_batch_fmnist = split_grid_image(hqconv_img_grid_fmnist, grid_size=(3, 3))

#FID Model for MNIST
fid = FrechetInceptionDistance(feature=2048)
hqconv_batch = hqconv_batch.repeat(1, 3, 1, 1).to(torch.uint8)
fid.update(real_batch, real=True)
fid.update(hqconv_batch, real=False)
fid = fid.compute()

# KID for MNIST
metric = KernelInceptionDistance(subsets=3, subset_size=9)
metric.update(real_batch, real=True)
metric.update(hqconv_batch, real=False)
kid = metric.compute()

print("HQConv Model Results:")
print("--------------------")
print(f"FID score hqconv MNIST: {fid}")
print(f"KID hqconv MNIST mean: {kid[0]:.9f}")
print(f"KID hqconv MNIST std: {kid[1]:.9f}")

#FID Model for FashionMNIST
fid = FrechetInceptionDistance(feature=2048)
hqconv_batch_fmnist = hqconv_batch_fmnist.repeat(1, 3, 1, 1).to(torch.uint8)
fid.update(real_batch_fmnist, real=True)
fid.update(hqconv_batch_fmnist, real=False)
fid = fid.compute()

# KID for FashionMNIST
metric = KernelInceptionDistance(subsets=3, subset_size=9)
metric.update(real_batch_fmnist, real=True)
metric.update(hqconv_batch_fmnist, real=False)
kid = metric.compute()

print(f"FID score hqconv FashionMNIST: {fid}")
print(f"KID hqconv FashionMNIST mean: {kid[0]:.9f}")
print(f"KID hqconv FashionMNIST std: {kid[1]:.9f}")