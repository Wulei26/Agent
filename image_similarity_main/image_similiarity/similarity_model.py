import torch
import torch.nn as nn

__all__ = ["ConvEncoder", "ConvDecoder"]


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))

        # (batch_size, 512, 1, 1)
        x = x.squeeze(-1).squeeze(-1)

        # (batch_size, 512)
        return x


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv5 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.deconv6 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        # (batch_size, 512) -> (batch_size, 512, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)

        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv4(x))
        x = torch.relu(self.deconv5(x))

        # 输出图像归一化到 [0, 1]
        x = torch.sigmoid(self.deconv6(x))

        return x


if __name__ == "__main__":
    input_tensor = torch.randn(32, 3, 64, 64)

    encoder = ConvEncoder()
    decoder = ConvDecoder()

    encoded_tensor = encoder(input_tensor)
    decoded_tensor = decoder(encoded_tensor)

    print("输入形状:", input_tensor.shape)
    print("编码形状:", encoded_tensor.shape)
    print("解码形状:", decoded_tensor.shape)
