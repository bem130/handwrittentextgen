# 学習する

import datetime
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

train_input_dir = project_dir+'font_data/train/input/'
train_target_dir = project_dir+'font_data/train/target/'

val_input_dir = project_dir+'font_data/val/input/'
val_target_dir = project_dir+'font_data/val/target/'

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

def tensorDataset(input_dir, target_dir, transform, device):
    """Datasetを全部読み込む"""
    input_images = sorted(os.listdir(input_dir))
    target_images = sorted(os.listdir(target_dir))

    x = []
    y = []
    for i, t in zip(input_images, target_images):
        i, t = os.path.join(input_dir, i), os.path.join(target_dir, t)
        i, t = Image.open(i), Image.open(t)
        if transform:
            i, t = transform(i), transform(t)
        i, t = i.to(device), t.to(device)
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

# Validation loop
def calculate_val_loss(generator, discriminator, val_loader, criterion_GAN, criterion_L1, device):
    generator.eval()  # モデルを評価モードに切り替え
    discriminator.eval()  # モデルを評価モードに切り替え

    val_loss_G = 0
    val_loss_D = 0
    val_steps = len(val_loader)
    gen_output = 0

    with torch.no_grad():  # 勾配計算を行わない
        for input_image, target_image in val_loader:
            input_image, target_image = input_image.to(device), target_image.to(device)

            # Generatorの出力を取得
            gen_output = generator(input_image)

            # Discriminatorの損失
            pred_real = discriminator(torch.cat((input_image, target_image), 1))
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            pred_fake = discriminator(torch.cat((input_image, gen_output), 1))
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            # Generatorの損失
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_L1 = criterion_L1(gen_output, target_image) * 100
            loss_G = loss_GAN + loss_L1

            val_loss_G += loss_G.item()
            val_loss_D += loss_D.item()

    return val_loss_G / val_steps, val_loss_D / val_steps, gen_output


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 96  #32
    num_epochs = 5000
    os.makedirs(os.path.join(epoch_dir), exist_ok=True)

    # Transforms
    transform = transforms.Compose([transforms.Resize((128, 512)), transforms.Grayscale(), transforms.ToTensor()])

    # DataLoader setup (訓練データと検証データ)
    #train_dataset = FontDataset(train_input_dir, train_target_dir, transform)
    train_dataset = tensorDataset(train_input_dir, train_target_dir, transform, device)  # データをcuda上に
    #val_dataset = FontDataset(val_input_dir, val_target_dir, transform)  # 同じディレクトリで分割する場合
    val_dataset = tensorDataset(val_input_dir, val_target_dir, transform, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 既にcudaにあるので0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_L1 = nn.L1Loss().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss tracking lists
    loss_G_list = []
    loss_D_list = []
    val_loss_G_list = []
    val_loss_D_list = []

    start_time = datetime.datetime.now()
    # Training loop
    for epoch in range(num_epochs):
        generator.train()  # モデルを学習モードに切り替え
        discriminator.train()  # モデルを学習モードに切り替え

        for i, (input_image, target_image) in enumerate(train_loader):
            input_image, target_image = input_image.to(device), target_image.to(device)

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
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(torch.cat((input_image, gen_output.detach()), 1))
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # Print logs every 100 steps
            if i % 100 == 0:
                # print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss_G: {loss_G.item()}, Loss_D: {loss_D.item()}")
                loss_G_list.append(loss_G.item())
                loss_D_list.append(loss_D.item())

        # Validate and print validation loss
        val_loss_G, val_loss_D, val_output = calculate_val_loss(generator, discriminator, val_loader, criterion_GAN, criterion_L1, device)
        # print(f"Epoch [{epoch}/{num_epochs}], Val Loss_G: {val_loss_G}, Val Loss_D: {val_loss_D}")
        val_loss_G_list.append(val_loss_G)
        val_loss_D_list.append(val_loss_D)

        # Save generated image every 10 epochs
        if (epoch) % 5 == 0:
            elapsed = datetime.datetime.now() - start_time
            remain = elapsed / (epoch+0.001) * num_epochs
            print(f"Epoch [{epoch}/{num_epochs}], Val Loss_G: {val_loss_G}, Val Loss_D: {val_loss_D}, estimate: {remain}")
        if (epoch) % 20 == 0:
            show_image(gen_output)
            show_image(val_output)
            # Plotting the loss curves
            plt.figure(figsize=(6, 3))
            plt.plot(loss_G_list, label='t G Loss')
            plt.plot(loss_D_list, label='t D Loss')
            plt.plot(val_loss_G_list, label='v G Loss', linestyle='dashed')
            plt.plot(val_loss_D_list, label='v D Loss', linestyle='dashed')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Generator and Discriminator Losses')
            plt.show()


        # Save model every 100 epochs
        if (epoch) % 100 == 0:
            torch.save(generator.state_dict(), model_dir+'/generator.'+str(epoch)+'.pth')

    # Final model save
    torch.save(generator.state_dict(), model_dir+'/generator.pth')
    torch.save(discriminator.state_dict(), model_dir+'/discriminator.pth')

if __name__ == '__main__':
    main()
