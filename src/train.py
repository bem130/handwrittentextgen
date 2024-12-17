# 学習する

import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

project_dir = "./"

# データセットのパス（Google Drive内のパスに変更）
input_dir = project_dir+'font_data/train/input/'
target_dir = project_dir+'font_data/train/target/'
epoch_dir = project_dir+'output/log/epoch/'
model_dir = project_dir+'output/'

# Generator (U-Net)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        return self.decoder(x2)


# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# Custom Dataset for loading images
class FontDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        assert len(self.input_images) == len(self.target_images), "入力画像とターゲット画像の数が一致していません"

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = os.path.join(self.input_dir, self.input_images[idx])
        target_image = os.path.join(self.target_dir, self.target_images[idx])

        # 画像をPIL Imageとして開く
        input_image = Image.open(input_image)
        target_image = Image.open(target_image)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def tensorDataset(input_dir, target_dir, transform):
    """
    Datasetを全部VRAMに乗せる
    """
    input_images = sorted(os.listdir(input_dir))
    target_images = sorted(os.listdir(target_dir))

    x = []
    y = []
    for i, t in zip(input_images, target_images):
        i, t = os.path.join(input_dir, i), os.path.join(target_dir, t)
        i, t = Image.open(i), Image.open(t)
        if transform:
            i, t = transform(i), transform(t)
        i, t = i.cuda(), t.cuda()
        x.append(i)
        y.append(t)
    x, y = torch.stack(x), torch.stack(y)
    return torch.utils.data.TensorDataset(x, y)


def show_image(tensor_image, max_images=16):
    """
    tensor_image : バッチサイズを持つ Tensor (B, C, H, W)
    max_images   : 表示する画像の最大数
    """
    # サブセットを選択
    if tensor_image.size(0) > max_images:
        tensor_image = tensor_image[:max_images]  # 最初の max_images 枚を取得
    # Tensor を CPU に移動し、[0, 1] の範囲に正規化
    tensor_image = tensor_image.detach().cpu()
    tensor_image = (tensor_image + 1) / 2  # Normalize to [0, 1]
    # グリッド画像を作成
    grid = torchvision.utils.make_grid(tensor_image, nrow=4, padding=2, normalize=True)
    # 画像を表示
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()

def main():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 96  # 8192
    num_epochs = 5000
    os.makedirs(os.path.join(epoch_dir), exist_ok=True)

    # Transforms for resizing and grayscale images
    transform = transforms.Compose([transforms.Resize((128, 512)), transforms.Grayscale(), transforms.ToTensor()])

    # DataLoader setup
    train_dataset = tensorDataset(input_dir, target_dir, transform)
    #train_dataset = FontDataset(input_dir, target_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, loss functions, and optimizers setup
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    # generator = Generator()
    # discriminator = Discriminator()
    criterion_GAN = nn.BCEWithLogitsLoss().cuda()
    criterion_L1 = nn.L1Loss().cuda()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for i, (input_image, target_image) in enumerate(train_loader):
            # input_image, target_image = input_image.cuda(), target_image.cuda()

            # Generator training
            optimizer_G.zero_grad()
            gen_output = generator(input_image)

            # Discriminator loss
            pred_fake = discriminator(torch.cat((input_image, gen_output), 1))
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_L1 = criterion_L1(gen_output, target_image) * 100
            loss_G = loss_GAN + loss_L1
            loss_G.backward()
            optimizer_G.step()

            # Discriminator training
            optimizer_D.zero_grad()
            pred_real = discriminator(torch.cat((input_image, target_image), 1))
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real, device='cuda'))
            pred_fake = discriminator(torch.cat((input_image, gen_output.detach()), 1))
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device='cuda'))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # Print logs every 100 steps
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss_G: {loss_G.item()}, Loss_D: {loss_D.item()}")

        # Save generated image every 10 epochs
        if (epoch + 1) % 10 == 0:
            show_image(gen_output)
        if (epoch + 1) % 100 == 0:
            torch.save(generator.state_dict(), model_dir+'/generator.'+str(epoch)+'.pth')

    # Save the models after training
    torch.save(generator.state_dict(), model_dir+'/generator.pth')
    torch.save(discriminator.state_dict(), model_dir+'/discriminator.pth')

if __name__ == '__main__':
    main()
