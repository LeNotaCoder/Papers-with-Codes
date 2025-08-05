import torch.nn as nn
import torch.nn.functional as F

class IntermidiateModel(nn.Module):
    def __init__(self):
        super(IntermidiateModel, self).__init__()
        
        self.H, self.W = 360, 240  
        self.proj = nn.Linear(768, 128)
        self.final_conv = nn.Conv2d(3, 3, kernel_size=(3,3), stride=(224//23, 224//120), padding=1)
        self.bn = nn.BatchNorm2d(3)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(5,5), stride=(2,1), padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=(3,3), stride=(2,1), padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=(3,3), stride=(2,1), padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=(3,3), stride=(2,2), padding=1), 
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=(3,3), stride=(1,1), padding=1), 
            nn.ReLU()
        )
        
    def forward(self, x):
        batch = x.size(0)

        x = self.proj(x) 
        x = x.view(batch, self.H, self.W, -1).permute(0,3,1,2)

        x = self.conv_layers(x)
        x = self.final_conv(x)
        x = self.bn(x)
        x = F.relu(x) 

        return x