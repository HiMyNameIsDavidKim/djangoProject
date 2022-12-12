from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm


'''
DCGAN 논문
https://arxiv.org/abs/1511.06434
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. 
Comparatively, unsupervised learning with CNNs has received less attention. 
In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. 
We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), 
that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. 
Training on various image datasets, 
we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from 
object parts to scenes in both the generator and discriminator. 
Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.
'''


class DcGan(object):
    def __init__(self):
        self.dataroot = '/Users/davidkim/PycharmProjects/djangoProject/moviee/data' # 사진은 넣어야 한다.
        self.workers = 2
        self.batch_size = 128
        self.image_size = 64
        self.nc = 3
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        self.num_epochs = 1
        self.lr = 0.0002
        self.beta1 = 0.5
        self.ngpu = 1
        self.dataloader = None
        self.netG = None
        self.netD = None

    def img_show(self):
        dataroot = self.dataroot
        image_size = self.image_size
        batch_size = self.batch_size
        workers = self.workers
        ngpu = self.ngpu
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers, multiprocessing_context='fork')
        self.dataloader = dataloader

        ngpu, workers = 0, 0 # M1 MPS에서 dataloader 오류. CPU로 전환.
        device = torch.device("mps" if (torch.backends.mps.is_available() and ngpu > 0) else "cpu")

        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def print_NetG(self):
        ngpu = self.ngpu
        device = torch.device("mps" if (torch.backends.mps.is_available() and ngpu > 0) else "cpu")

        netG = Generator(ngpu).to(device)

        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        netG.apply(self.weights_init)

        self.netG = netG
        print(netG)

    def print_NetD(self):
        ngpu = self.ngpu
        device = torch.device("mps" if (torch.backends.mps.is_available() and ngpu > 0) else "cpu")

        netD = Discriminator(ngpu).to(device)

        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        netD.apply(self.weights_init)

        self.netD = netD
        print(netD)

    def hook_dcgan(self):
        self.img_show()
        self.print_NetG()
        self.print_NetD()
        self.train_dcgan()

    def train_dcgan(self):
        nz = self.nz
        ngpu = self.ngpu
        netD = self.netD
        netG = self.netG
        lr = self.lr
        beta1 = self.beta1
        num_epochs = self.num_epochs
        dataloader = self.dataloader
        device = torch.device("mps" if (torch.backends.mps.is_available() and ngpu > 0) else "cpu")

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        real_label = 1.
        fake_label = 0.

        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            for i, data in enumerate(tqdm(dataloader)):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1
            # Output training stats
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        real_batch = next(iter(dataloader))

        # Plot the real images
        ngpu, workers = 0, 0  # M1 MPS에서 dataloader 오류. CPU로 전환.
        device = torch.device("mps" if (torch.backends.mps.is_available() and ngpu > 0) else "cpu")

        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.show()

    def zoom_in(self):
        mydlib = Mydlib()
        mydlib.zoom_in()


class Generator(nn.Module):
    def __init__(self, ngpu):
        that = DcGan()
        nz = that.nz
        ngf = that.ngf
        nc = that.nc
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        that = DcGan()
        nc = that.nc
        ndf = that.ndf
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


import sys
import dlib
import cv2
import openface


class Mydlib(object):
    def __init__(self):
        pass

    def zoom_in(self):
        predictor_model = "../files/shape_predictor_68_face_landmarks.dat"

        face_detector = dlib.get_frontal_face_detector()
        face_pose_predictor = dlib.shape_predictor(predictor_model)
        face_aligner = openface.AlignDlib(predictor_model)

        image = cv2.imread('../files/Lenna.png')

        detected_faces = face_detector(image, 1)
        for i, face_rect in enumerate(detected_faces):
            print(
                "- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                   face_rect.right(),
                                                                                   face_rect.bottom()))

            pose_landmarks = face_pose_predictor(image, face_rect)
            alignedFace = face_aligner.align(532, image, face_rect,
                                             landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            cv2.imwrite(f"../data/aligned_face_{i}.jpg", alignedFace)


def menu_show(ls):
    [print(f"{i}.{j}") for i, j in enumerate(ls)]
    return input("Choose menu : ")


dc_menu = ["Exit", #0
                "/moviee/movies/faces", # 1. Load dataset
                "/moviee/movies/fake-faces", # 2. Train and Show fake image
                "/moviee/movies/face-zoom-in", # 3. Find face
]
dc_lambda = {
    "1": lambda x: x.img_show(),
    "2": lambda x: x.hook_dcgan(),
    "3": lambda x: x.zoom_in(),
}


if __name__ == '__main__':
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    api = DcGan()
    while True:
        menu = menu_show(dc_menu)
        if menu == '0':
            print("Exit")
            break
        else:
            try:
                dc_lambda[menu](api)
            except KeyError as e:
                if 'Some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")