import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_juan import Discriminator
from generator_juan import Generator


########################
########################Juan's part begins here
########################

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        #DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
        #DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD Training Discriminator begins
        #DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
        with torch.cuda.amp.autocast():
            horse=horse
            zebra=zebra
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)


            #loss for real horse
            D_H_real = disc_H(horse)
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            
            #loss for fake horse
            D_H_fake = disc_H(fake_horse.detach())
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))

            #loss for both horses
            D_H_loss = D_H_real_loss + D_H_fake_loss

            #loss for real zebra
            D_Z_real = disc_Z(zebra)
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            
            #loss for fake zebra
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))

            #loss for both zebras
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            #loss for discriminator for zebra and horse 
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
        #DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD Training Discriminator ends
        #DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD

        #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
        #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG Training Generator begins
        #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG

        with torch.cuda.amp.autocast():

            #adversarial loss for fake horse
            D_H_fake = disc_H(fake_horse)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            
            #adversarial loss for fake zebra
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss for zebra
            cycle_zebra = gen_Z(fake_horse)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            
            # cycle loss for horse
            cycle_horse = gen_H(fake_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)


            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
        #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG Training Generator ends
        #GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG

        if idx % 200 == 0:
            save_image(fake_horse*0.5+0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"saved_images/zebra_{idx}.png")
            save_image(cycle_horse*0.5+0.5, f"saved_images/cycle_horse_{idx}.png")
            save_image(cycle_zebra*0.5+0.5, f"saved_images/cycle_zebra_{idx}.png")
            save_image(zebra*0.5+0.5, f"saved_images/original_zebra_{idx}.png")
            save_image(horse*0.5+0.5, f"saved_images/original_horse_{idx}.png")





########################
########################Juan part ends here
########################


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3).to(config.DEVICE)
    gen_H = Generator(img_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()





    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+"/horses", root_zebra=config.TRAIN_DIR+"/zebras", transform=config.transforms
    )
    val_dataset = HorseZebraDataset(
       root_horse=config.VAL_DIR+"/horses", root_zebra=config.VAL_DIR+"/zebras", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()
