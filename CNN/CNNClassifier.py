import torch
from torch import nn


#implement one of the cnn structure VGG16
class CNNClassifier(nn.Module):
    def __init__(self, ds_method="Maxpooling", dropout=True):
        super(CNNClassifier, self).__init__()

        self.dropout_activate = dropout

        self.ds_method = ds_method
        if self.ds_method == "Stridedconv":
            self.stride = 2
            self.kernel_size = 2
        else:
            self.stride = 1
            self.kernel_size = 3


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel_size, stride=self.stride, padding=1)
        if self.ds_method == "Stridedconv":
            self.features = nn.Sequential(self.conv1, nn.ReLU(inplace=True),
                                          self.conv2, nn.ReLU(inplace=True),
                                          self.conv3, nn.ReLU(inplace=True),
                                          self.conv4, nn.ReLU(inplace=True),
                                          self.conv5, nn.ReLU(inplace=True), )
        else:
            if self.ds_method == "Maxpooling":
                self.ds_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            if self.ds_method == "Averagepooling":
                self.ds_layer = nn.AvgPool2d(kernel_size=2, stride=2)

            self.features = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.ds_layer,
                                          self.conv2, nn.ReLU(inplace=True), self.ds_layer,
                                          self.conv3, nn.ReLU(inplace=True), self.ds_layer,
                                          self.conv4, nn.ReLU(inplace=True),
                                          self.conv5, nn.ReLU(inplace=True), self.ds_layer)



        if self.dropout_activate:
            self.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=128),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(in_features=128, out_features=10),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            )
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=128),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(in_features=128, out_features=10),
                                            nn.ReLU(inplace=True),
                                            )

    def forward(self, x):
        output = self.features(x)
        output = output.view(x.size(0), -1)        # bsz * 2048
        return self.classifier(output)


if __name__ == '__main__':
    test = torch.rand(size=(64, 3, 32, 32))
    #model = CNNClassifier(ds_method="Stridedconv", dropout=True)
    model = CNNClassifier(dropout=True)
    print(model(test))
