"""Main script for running train/val/test"""

import os
import re
import yaml
import argparse
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
from ConvCVAE import ConvCVAE
from pierce_Train import train_model
# from test import test_model
from data import ChromosomeDataset
from torch.utils.data import DataLoader
import logging 
from tqdm import tqdm

def setup_logger(output, exp_name):
    logger = logging.getLogger("Convolutional Conditional Variational Autoencoder")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    log_output = os.path.join(output, 'logs', exp_name + '.log')
    fh = logging.FileHandler(log_output)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def extract_yaml(yaml_path):
  with open(yaml_path, 'r') as f:
      return yaml.safe_load(f)
    
def get_exp_num(output_dir, exptype):
  exp_output_dir = os.path.join(output_dir, exptype)
  os.makedirs(exp_output_dir, exist_ok=True)
  max_num = 0
  dir_list = os.listdir(exp_output_dir)
  for d in dir_list:
    if 'exp' in d and os.path.isdir(os.path.join(exp_output_dir, d)):
      match = re.search(r'exp(\d+)', d)
      if match:
        curr_num = int(match.group(1))
        if max_num < curr_num:
          max_num = curr_num
  return max_num + 1

def create_dir(results_dir):
  if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok = True)
    print(f"Making directory {results_dir}...")
    
def print_hyps(args, parser):
  print("Summary of Specified Hyperparameters:")
  defaults = vars(parser.parse_args([]))  # Get default values
  changed = False
  for k, v in vars(args).items():
    if k in defaults and v != defaults[k]:
      print(f"  {k}: {v}")
      changed = True
  if not changed:
    print("  (no changes from defaults)")

def main(args):
  # Set seeds:
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  target_imgsize = (args.imgsize, args.imgsize)
  model = ConvCVAE(target_imgsize, latent_dim = args.latent, deeper = args.deeper).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=0.5)
  
  if args.weight is not None:
    if os.path.exists(args.weight):
        state = torch.load(args.weight, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from: {args.weight}")
    else:
        print(f"WARNING: weight path not found: {args.weight}")

  os.makedirs(args.output, exist_ok = True)
  
  # get a new exp name for the experiment:
  dir_name = get_exp_num(args.output, args.exptype)
  exp_name = 'exp' + str(dir_name)
  # logger = setup_logger(args.output, exp_name)
  
  config = extract_yaml(args.yaml)
  results_dir = os.path.join(args.output, args.exptype, exp_name)
  create_dir(results_dir)

  if args.exptype == 'train':
    train_img_paths = glob.glob(f"{config['train']}/*.jpg")
    val_img_paths = glob.glob(f"{config['val']}/*.jpg")
    
    train_data = ChromosomeDataset(train_img_paths, target_size = target_imgsize, transform = True)
    train_dataloader = DataLoader(train_data, args.bsize, shuffle=True)
  
    val_data = ChromosomeDataset(val_img_paths, target_size = target_imgsize, transform = False)
    val_dataloader = DataLoader(val_data, args.bsize, shuffle=False)
    
    training_loss, val_loss = train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, device, args.epochs, results_dir, args.patience, args.beta)
    print(f"Training results found in directory {exp_name}")
    
  elif args.exptype == 'val':
    print(f"{args.exptype} not implemented yet", flush = True)
  elif args.exptype == 'test':
    print(f"{args.exptype} not implemented yet", flush = True)
    # test_data = ChromosomeDataset(test_img_paths, target_size = target_imgsize, transform = False)
    # test_dataloader = DataLoader(test_data, args.bsize, shuffle=False)
    # 
    # test_results = test_model(model, test_dataloader, device, args.bce, args.ssim)
    # 
   # Use the trained encoder model part to get z_mean

  

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description="Convolutional CVAE")
  parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train")
  parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
  parser.add_argument("--seed", type=int, default=42, help="random seed")
  parser.add_argument("--imgsize", type=int, default=64, help="image resizing dimension (img_size x img_size)")
  parser.add_argument("--bsize", type=int, default=32, help="batch size for training")
  parser.add_argument("--yaml", type=str, default='config.yaml', help="path to config yaml file")
  parser.add_argument("--output", type=str, default='outputs/', help="path to the model outputs")
  parser.add_argument("--exptype", type=str, choices=["train", "val", "test"], required=True, help="choose between train, val, test")
  parser.add_argument("--beta", type=float, default=0.1, help="beta annealing value for training")
  parser.add_argument("--patience", type=float, default=20, help="patience epoch number")
  parser.add_argument("--deeper", action="store_true", help="Use deeper model architecture")
  parser.add_argument("--latent", type=int, default=64, help="latent space dimension")
  parser.add_argument("--stepsize", type=int, default=10, help="decide on the step size for scheduler")
  parser.add_argument("--weight", type=str, default="outputs/train/exp188/weights/best.pth", help="hyperparameters/weight")

  args = parser.parse_args()
  main(args)
  print_hyps(args, parser)

