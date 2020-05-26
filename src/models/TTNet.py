import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchnorm(self.conv(x))))
        return x


class ConvBlock_without_Pooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_without_Pooling, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv(x)))
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        middle_channels = int(in_channels / 4)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU()
        self.tconv = nn.ConvTranspose2d(middle_channels, middle_channels, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm1(self.tconv(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))

        return x


class BallDetection(nn.Module):
    def __init__(self, dropout_p):
        super(BallDetection, self).__init__()
        self.conv1 = nn.Conv2d(27, 64, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.convblock1 = ConvBlock(in_channels=64, out_channels=64)
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.convblock3 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock4 = ConvBlock(in_channels=128, out_channels=128)
        self.convblock5 = ConvBlock(in_channels=128, out_channels=256)
        self.convblock6 = ConvBlock(in_channels=256, out_channels=256)
        self.fc1 = nn.Linear(in_features=2560, out_features=1792)
        self.fc2 = nn.Linear(in_features=1792, out_features=896)
        self.fc3 = nn.Linear(in_features=896, out_features=448)
        self.dropout1d = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        out_block2 = self.convblock1(self.convblock1(x))
        x = self.dropout2d(out_block2)
        out_block3 = self.convblock3(x)
        out_block4 = self.convblock4(out_block3)
        x = self.dropout2d(out_block4)
        out_block5 = self.convblock5(out_block4)
        features = self.convblock6(out_block5)

        x = self.dropout2d(features)
        x = x.view(x.size(0), -1)

        x = self.dropout1d(self.relu(self.fc1(x)))
        x = self.dropout1d(self.relu(self.fc2(x)))
        out = self.sigmoid(self.fc3(x))

        return out, features, out_block2, out_block3, out_block4, out_block5


class EventsSpotting(nn.Module):
    def __init__(self, dropout_p):
        super(EventsSpotting, self).__init__()
        self.conv1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.convblock = ConvBlock_without_Pooling(in_channels=64, out_channels=64)
        self.fc1 = nn.Linear(in_features=640, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_features, local_features):
        input_eventspotting = torch.cat((global_features, local_features), dim=1)
        x = self.relu(self.batchnorm(self.conv1(input_eventspotting)))
        x = self.dropout2d(x)
        x = self.convblock(x)
        x = self.dropout2d(x)
        x = self.convblock(x)
        x = self.dropout2d(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(x))

        return out


class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.deconvblock5 = DeconvBlock(in_channels=256, out_channels=128)
        self.deconvblock4 = DeconvBlock(in_channels=128, out_channels=128)
        self.deconvblock3 = DeconvBlock(in_channels=128, out_channels=64)
        self.deconvblock2 = DeconvBlock(in_channels=64, out_channels=64)
        self.tconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0,
                                        output_padding=0)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=2, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, out_block2, out_block3, out_block4, out_block5):
        x = self.deconvblock5(out_block5)
        x = x + out_block4
        x = self.deconvblock4(x)
        x = x + out_block3
        x = self.deconvblock3(x)

        x = x + out_block2
        x = self.deconvblock2(x)

        x = self.relu(self.tconv(x))

        x = self.relu(self.conv1(x))

        out = self.sigmoid(self.conv2(x))

        return out


class TTNet(nn.Module):
    def __init__(self, dropout_p, input_size=(320, 128)):
        super(TTNet, self).__init__()
        self.ball_global_stage = BallDetection(dropout_p=dropout_p)
        self.ball_local_stage = BallDetection(dropout_p=dropout_p)
        self.events_spotting = EventsSpotting(dropout_p=dropout_p)
        self.segmentation = Segmentation()
        self.w_resize = input_size[0]
        self.h_resize = input_size[1]

    def forward(self, original_batch_input, resize_batch_input):
        pred_ball_global, global_features, out_block2, out_block3, out_block4, out_block5 = self.ball_global_stage(
            resize_batch_input)

        input_ball_local = self.crop_original_batch(original_batch_input, resize_batch_input, pred_ball_global)
        # input_ball_local = input_ball_local.cuda()
        pred_ball_local, local_features, _, _, _, _ = self.ball_local_stage(input_ball_local)

        pred_eventspotting = self.events_spotting(global_features, local_features)

        pred_segmentation = self.segmentation(out_block2, out_block3, out_block4, out_block5)


        return pred_ball_global, pred_ball_local, pred_eventspotting, pred_segmentation

    def crop_original_batch(self, original_batch_input, resize_batch_input, pred_ball_global):
        # Process input for local stage based on output of the global one
        b_size, _, h_original, w_original = original_batch_input.size()
        ball_pos_x = torch.argmax(pred_ball_global[:, :self.w_resize], dim=-1)  # Upper part
        ball_pos_y = torch.argmax(pred_ball_global[:, self.w_resize:], dim=-1)  # Lower part

        # Crop the original images
        input_ball_local = torch.zeros_like(resize_batch_input)  # same shape with resize_batch_input
        for i in range(b_size):
            x_min, x_max, y_min, y_max = self.get_crop_params(ball_pos_x[i], ball_pos_y[i], self.w_resize, self.h_resize,
                                                              w_original,
                                                              h_original)
            input_ball_local[i, :, :, :] = original_batch_input[i, :, y_min:y_max, x_min: x_max]

        return input_ball_local

    @staticmethod
    def get_crop_params(x_center, y_center, w_resize, h_resize, w_original, h_original):
        x_min = max(0, x_center - int(w_resize / 2))
        y_min = max(0, y_center - int(h_resize / 2))

        x_max = min(w_original, x_min + w_resize)
        y_max = min(h_original, y_min + h_resize)

        return x_min, x_max, y_min, y_max


if __name__ == '__main__':
    from torchsummary import summary

    ttnet = TTNet(dropout_p=0.5).cuda()
    resize_batch_input = torch.rand((10, 27, 128, 320)).cuda()
    original_batch_input = torch.rand((10, 27, 1080, 1920)).cuda()
    pred_ball_global, pred_ball_local, pred_eventspotting, pred_segmentation = ttnet(original_batch_input,
                                                                                     resize_batch_input)
    print('pred_ball_global: {}, pred_ball_local: {}'.format(pred_ball_global.size(), pred_ball_local.size()))
    print('pred_segmentation: {}, pred_eventspotting: {}'.format(pred_segmentation.size(), pred_eventspotting.size()))
