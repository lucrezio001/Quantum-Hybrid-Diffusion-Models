import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader
from models.unet_base import Unet
from models.unet_vertex_fqconv import Unet_fqconv
from models.unet_vertex_hqconv import Unet_hqconv
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        #lambda x: (2 * x) - 1
    ])

    # Create the dataset

    mnist = MNIST(root="./data", train=True, transform=transform, download=True)
    fashion_mnist = FashionMNIST(root="./data", train=True, transform=transform, download=True)

    # undersample
    # # Set random seed for reproducibility
    # torch.manual_seed(42)
    # indices = torch.randperm(len(mnist))[:1000]
    # littele_mnist = torch.utils.data.Subset(mnist, indices)
    
    mnist_loader = DataLoader(fashion_mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=2)
    
    # Instantiate the model
    # model = Unet_fqconv(model_config).to(device)
    # model = Unet_hqconv(model_config).to(device)
    model = Unet(model_config).to(device)
    model.train()
    
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    
    # classic: Trainable parameters: 6,884,297
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))
    
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im, y in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))
    
    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
