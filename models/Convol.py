class Convol(nn.Module):
    def ___init___(self):
        super(Convol, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, 
                                     kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conx2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.linear_1 = nn.Linear(7 * 7 * 64, 128)
        self.linear_2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.linear_2(x)

        return out
