import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        # Modify the first layer to accept one channel (as input is grayscale)
        self.pretrained_model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
      
    def forward(self, x):
        # Assuming x is a batch of grayscale images with shape (batch_size, 1, height, width)
        #x = torch.squeeze(x, dim=1)  # Remove the channel dimension
        features = self.pretrained_model.features(x)
        #print(features.shape)
        pooled_features = self.pooling_layer(features)
  
        return features,pooled_features

# Here I am Defining BiGRU architecture
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional GRU

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional GRU
        out, hidden = self.bigru(x, h0)
        out = self.fc(out[:, -1, :])
        return out,hidden

class FCNN(nn.Module):
    def __init__(self, input_size, output_size,dropout_prob=0.5):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(input_size//2, input_size//4)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(input_size//4, input_size//8)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc4 = nn.Linear(input_size//8, output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x=self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc4(x)
        
        #x = self.sigmoid(x)
        return x
