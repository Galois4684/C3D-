import torch
import torch.nn as nn
import torch.nn.init as init

class C3D(nn.Module):

    def __init__(self, num_classes, weight_path=None):
        super(C3D, self).__init__()

        self.weigh_path = weight_path

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if self.weigh_path:

            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.01)
                init.constant_(m.bias, 0.0)

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weights": "conv1.weights",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weights": "conv2.weights",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weights": "conv3a.weights",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weights": "conv3b.weights",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weights": "conv4a.weights",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weights": "conv4b.weights",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weights": "conv5a.weights",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weights": "conv5b.weights",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weights": "fc6.weights",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weights": "fc7.weights",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(self.weigh_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)