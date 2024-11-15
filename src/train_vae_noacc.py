import time
import argparse
import json
import logging
import math
import os
# from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
import os

from model import VAE
from Trashmyself.vae_modules.discriminator.model import NLayerDiscriminator, weights_init

import torch


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def calculate_adaptive_weight(nll_loss, g_loss, discriminator_weight, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight

def make_kl(mean, var, logvar):
    kl_loss = 0.5 * torch.mean(
                torch.pow(mean, 2) + var - 1.0 - logvar,
                dim=[1, 2,],
            )
    
    return kl_loss

def main():

    args={}
    args.learning_rate = 1e-3
    args.num_train_epochs = 100
    
    input_dim = 384
    hidden_dim = 128
    latent_dim = 13


    
    train_dataloader = DataLoader(train_multi_generator, shuffle=True, batch_size=32)
    eval_dataloader = DataLoader(eval_multi_generator, shuffle=False, batch_size=32)
    
    model_G = VAE(input_dim, hidden_dim, latent_dim).cuda()
    model_D = NLayerDiscriminator(input_nc=3,
                                n_layers=3,
                                use_actnorm=False,
                                ndf=64
                                ).apply(weights_init).cuda()
    
    
    optimizer_G_parameters = model_G.parameters()
    optimizer_G = torch.optim.Adam(optimizer_G_parameters, lr=args.learning_rate)
    
    optimizer_D_parameters = model_D.parameters()
    optimizer_D = torch.optim.Adam(optimizer_D_parameters, lr=args.learning_rate)
    
    lr_scheduler_G = ExponentialLR(optimizer_G, gamma=0.95)
    lr_scheduler_D = ExponentialLR(optimizer_D, gamma=0.95)
    
    writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard_log"))
    
    progress_bar = tqdm(range(len(train_dataloader)*args.num_train_epochs))
    
    global_steps = 0
    best_loss = np.inf
    save_checkpoint = False
    
    alpha = 1e-6
    discriminator_iter_start = 10000
    
    for epoch in range(args.num_train_epochs):
        model_G.train()
        model_D.train()
        total_loss_recon, total_loss_gan, total_loss_d, total_val_loss_recon, total_val_loss_gan, total_val_loss_d = 0, 0, 0, 0, 0, 0
        for step, batch in enumerate(train_dataloader):
            device = torch.device('cuda')
            input_amp = batch['X_amp'][:, 0].to(device, dtype=torch.float32)
            
            reconstruction, mean, logvar, var = model_G(input_amp)
            
            kl_loss = make_kl(mean, var, logvar)
            
            rec_loss = torch.abs(input_amp.contiguous() - reconstruction.contiguous())**2

            nll_loss = rec_loss
            nll_loss = torch.mean(nll_loss)
            
            logits_fake = model_D(reconstruction.contiguous())
            
            logits_fake_D = model_D(reconstruction.contiguous().detach())
            logits_real = model_D(input_amp.contiguous().detach())
            
            g_loss = -torch.mean(logits_fake)
            
            d_weight = calculate_adaptive_weight(nll_loss, g_loss, 0.5, last_layer=model_G.get_last_layer())
            disc_factor = adopt_weight(1.0, global_steps, threshold=discriminator_iter_start)
            loss_D = disc_factor * hinge_d_loss(logits_real, logits_fake_D)
            loss_G = nll_loss + d_weight * disc_factor * g_loss + alpha * kl_loss.mean()
            
            total_loss_recon += nll_loss.detach().float()
            total_loss_gan += loss_G.detach().float()
            
            loss_G.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()
            
            total_loss_d += loss_D.detach().float()
            
            loss_D.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()
            
            lr_scheduler_G.step()
            lr_scheduler_D.step()
            
            progress_bar.update(1)
            global_steps += 1
        
        writer.add_scalar('Recon Loss/Train (epoch)', round(total_loss_recon.item()/len(train_dataloader), 4), epoch)
        writer.add_scalar('GAN Loss/Train (epoch)', round(total_loss_gan.item()/len(train_dataloader), 4), epoch)
        writer.add_scalar('Disc Loss/Train (epoch)', round(total_loss_d.item()/len(train_dataloader), 4), epoch)
        
        model_G.eval()
        model_D.eval()
        eval_progress_bar = tqdm(range(len(eval_dataloader)))
        
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):

                device = torch.device('cuda')
                input_amp = batch['X_amp'][:,0].to(device, dtype=torch.float32)
                
                reconstruction, mean, logvar, var = model_G(input_amp)
            
                kl_loss = make_kl(mean, var, logvar)
            
                rec_loss = torch.abs(input_amp.contiguous() - reconstruction.contiguous())**2
                
                nll_loss = rec_loss
                nll_loss = torch.mean(nll_loss)
                
                #logits_fake = model_D(reconstruction.contiguous())
                #g_loss = -torch.mean(logits_fake)
                
                #d_weight = calculate_adaptive_weight(nll_loss, g_loss, 0.5, last_layer=model_G.get_last_layer())
                #disc_factor = adopt_weight(1.0, global_step, threshold=discriminator_iter_start)
                #loss_G = nll_loss + d_weight * disc_factor * g_loss + alpha * kl_loss.mean()
                
                total_val_loss_recon += nll_loss.detach().float()
                #total_val_loss_gan += loss_G.detach().float()
                
                #logits_real = model_D(input_amp.contiguous().detach())
                #loss_D = disc_factor * hinge_d_loss(logits_real, logits_fake)
                
                #total_val_loss_d += loss_D.detach().float()
                
                eval_progress_bar.update(1)
        
        now_val_loss = total_val_loss_recon.item()/len(eval_dataloader)
        
        writer.add_scalar('Recon Loss/Test (epoch)', round(now_val_loss, 4), epoch)
        #writer.add_scalar('GAN Loss/Test (epoch)', round(total_val_loss_gan.item()/len(eval_dataloader), 4), epoch)
        #writer.add_scalar('Disc Loss/Test (epoch)', round(total_val_loss_d.item()/len(eval_dataloader), 4), epoch)
        
        if now_val_loss < best_loss:
            best_loss = now_val_loss
            save_checkpoint = True
        else:
            save_checkpoint = False
            
            
        if save_checkpoint:
            state_model_G = model_G.state_dict()
            state_opt_G =  optimizer_G.state_dict()
            
            state_model_D = model_D.state_dict()
            state_opt_D =  optimizer_D.state_dict()
            
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, "best")
                os.makedirs(output_dir, exist_ok=True)
            torch.save(state_model_G, os.path.join(output_dir, 'best_model_G.bin'))
            torch.save(state_opt_G, os.path.join(output_dir, 'best_optimizer_G.bin'))
            torch.save(state_model_D, os.path.join(output_dir, 'best_model_D.bin'))
            torch.save(state_opt_D, os.path.join(output_dir, 'best_optimizer_D.bin'))
        
        if epoch % 50 == 0:    
            state_model_G = model_G.state_dict()
            state_opt_G =  optimizer_G.state_dict()
            
            state_model_D = model_D.state_dict()
            state_opt_D =  optimizer_D.state_dict()
            output_dir = f"epoch_{epoch+1 }"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                os.makedirs(output_dir, exist_ok=True)
            torch.save(state_model_G, os.path.join(output_dir, f'model_G_{epoch+1}.bin'))
            torch.save(state_opt_G, os.path.join(output_dir, f'optimizer_G_{epoch+1}.bin'))
            torch.save(state_model_D, os.path.join(output_dir, f'model_D_{epoch+1}.bin'))
            torch.save(state_opt_D, os.path.join(output_dir, f'optimizer_D_{epoch+1}.bin'))
        
        
if __name__ == '__main__':
    main()
    
