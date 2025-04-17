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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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

    mnist =  MNIST(root="./data", train=True, transform=transform, download=True)
    fashion_mnist = FashionMNIST(root="./data", train=True, transform=transform, download=True)
    # Set random seed for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(len(mnist))[:1000]
    littele_mnist = torch.utils.data.Subset(mnist, indices)
    littele_fmnist = torch.utils.data.Subset(fashion_mnist, indices)
    mnist_loader = DataLoader(littele_fmnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=2)
    
    # Instantiate the model
    # Use different quantum Unet
    # model = Unet(model_config).to(device)
    # model = Unet_hqconv(model_config).to(device)
    model = Unet_fqconv(model_config).to(device)
    model.train()
    
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    
    # Freeze tutti i parametri non quantistici
    for name, param in model.named_parameters():
        if 'quantum' not in name:
            param.requires_grad = False
        else:
            print(f"Parametro quantistico trainabile: {name}")
    
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    
    # classic: Trainable parameters: 6,884,297
    # quantum: Trainable parameters: 6,884,561
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    
    #### Custom import for fine tuning
    
    # # Load checkpoint
    # old_state_dict = torch.load(r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\ddpm_ckpt.pth")

    # # map
    # new_state_dict = {}
    # for key in old_state_dict:
    #     new_key = key.replace("fqconv_ansatz_layer.weights", "fqconv_weights")
    #     new_state_dict[new_key] = old_state_dict[key]

    # # Carica nel nuovo modello
    # model.load_state_dict(new_state_dict, strict=False)
    
    # Load the checkpoint
    old_state_dict = torch.load(r"F:\UniversitE\Quantum test\DDPM-Pytorch\default\ddpm_ckpt.pth")

    # Remap keys and exclude quantum layer weights
    new_state_dict = {}
    for key in old_state_dict:
        # Skip the quantum layer weights
        if "fqconv_ansatz_layer.weights" in key or "fqconv_weights" in key:
            continue

        # Optionally rename keys if needed
        new_key = key.replace("fqconv_ansatz_layer.weights", "fqconv_weights")
        new_state_dict[new_key] = old_state_dict[key]

    # Load into model (strict=False so missing keys are allowed)
    model.load_state_dict(new_state_dict, strict=False)


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
