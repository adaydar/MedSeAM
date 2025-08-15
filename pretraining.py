import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from utils.config import *
from utils.dataset import *
from utils.MDFCA import *
from utils.cnn_Bigru import *

# Initialize CNN models
cnn_models = CNN().cuda()

# Load saved CNN models
cnn_models.load_state_dict(torch.load(cnn_model_path))
hidden_size=64
# Initialize and load the bigru model
bigru_model = BiGRU(input_size=2304, hidden_size=64, num_layers=2, num_classes=num_cnn).to(device)
bigru_model.load_state_dict(torch.load(bigru_model_path))

cross_attention = MDFCA_CrossAttention().to(device)
cross_attention.load_state_dict(torch.load(CA_path))
input_size=2560
output_size=2
fc_layer = FCNN(input_size, output_size).to(device)

parameters = list(cnn_models.parameters()) + list(bigru_model.parameters())+list(fc_layer.parameters())+list(cross_attention.parameters())
optimizer = optim.Adam(parameters, lr=0.0001)
step_size = 10  # Number of epochs after which to reduce the learning rate
gamma = 0.1  # Factor by which to reduce the learning rate
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) 

def validate_model(val_loader, bigru_model, cnn_models,cross_attention,fc_layer, num_cnn,epoch):
    # Set models to evaluation mode
    bigru_model.eval()
    cnn_models.eval()
    cross_attention.eval()
    fc_layer.eval()

    # Initialize metrics for validation
    total_val_loss = 0
    total_samples = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for i, (sequence_batch,spectral_batch,labels,patient_id,weight) in enumerate(val_loader):
            weight=weight.to(device)  
            cnn_output1,pool1 = cnn_models(sequence_batch[0].float().to(device))
            cnn_output2,pool2=cnn_models(spectral_batch[0].float().to(device))
            CA_output1=cross_attention(cnn_output1,cnn_output2)
            CA_output1=CA_output1.view(17, -1)
            #print('CA_output1',CA_output1.shape)
            features = CA_output1.unsqueeze(0).to(device)
            bigru_output, temporal_features = bigru_model(features)	
            flattened_hidden = temporal_features.permute(1, 0, 2).contiguous().view(-1)
            CA_output1 = CA_output1.view(17,-1).to(device)
            CA_output1 = CA_output1.unsqueeze(0)  # Add a batch dimension
            flattened_cnn_output = nn.AdaptiveAvgPool1d(1)(CA_output1.transpose(2, 1)).squeeze()
            #print('flattened_hidden shape',flattened_hidden.shape)
            #print('flattened_cnn_output shape',flattened_cnn_output.shape)
            concatenated_features = torch.cat((flattened_cnn_output, flattened_hidden), dim=0).to(device)
            # Fully connected layer
            fc_output = fc_layer(concatenated_features)
            fc_output=fc_output.unsqueeze(0)
            probabilities = torch.softmax(fc_output,dim=1).to(device) 
            #probabilities = probabilities.unsqueeze(0)
            #labels = torch.tensor(labels.clone().detach()).cuda()
            #labels = labels.squeeze(0)
            labels=labels.float().to(device)
            #print(labels)
            loss = torch.nn.BCELoss(weight=weight)(probabilities, labels)
            
            #loss=torch.nn.BCEWithLogitsLoss(weight=weight)(fc_output,labels)
            #print('predicted and actual ',fc_output,'    ',labels)
            # Here Iam getting predicted class (index with highest probability)
            #print('predicted ',probabilities,'  actual',labels[0][1])
            predicted_classes = (probabilities[:, 1] > 0.5).float()
            #print('predicted_classes',predicted_classes,'  ','labels[0][1] ',labels[0][1])

            correct_predictions += (predicted_classes == labels[0][1]).item()
            #print('correct_predictions ',correct_predictions)nn.AdaptiveAvgPool1d(1)(cnn_output.transpose(1, 0)).squeeze()
            #print((fc_output[1] == labels[0][1]).sum().item())
            #print('predicted_class ',predicted_class)
            
            total_samples += sequence_batch.size(0)
            #print("seq batch",sequence_batch.size(0))
            total_val_loss += loss.item()
            
            
    average_val_loss = total_val_loss / len(val_loader)
    accuracy = correct_predictions / (len(val_loader))
    #print('type of average_val_loss',type(average_val_loss))
    #print('type of accuracy ',type(accuracy))
    return average_val_loss, accuracy
    
# Training function
def train_model(train_loader, val_loader, bigru_model, cnn_models, cross_attention,fc_layer,optimizer, num_epochs, device,log_file_path):
    lowest_val_loss = float('inf')  # Initialize with a high value
    with open(log_file_path, 'w') as log_file:
        for epoch in range(num_epochs):
            #Here Setting models to training mode
            bigru_model.train()
            cnn_models.train()
            cross_attention.train()
            fc_layer.train()

            total_loss = 0
            for i, (sequence_batch,spectral_batch,labels,patient_id,weight) in enumerate(train_loader):
                #print(labels,'  ',patient_id)
                optimizer.zero_grad()  
                weight=weight.to(device)    
                cnn_output1,pool1 = cnn_models(sequence_batch[0].float().to(device))
                cnn_output2,pool2=cnn_models(spectral_batch[0].float().to(device))
                CA_output1=cross_attention(cnn_output1,cnn_output2)
                CA_output1=CA_output1.view(17, -1)
                #print('CA_output1',CA_output1.shape)
                features = CA_output1.unsqueeze(0).to(device)
                bigru_output, temporal_features = bigru_model(features)	
                flattened_hidden = temporal_features.permute(1, 0, 2).contiguous().view(-1)
                CA_output1 = CA_output1.view(17,-1).to(device)
                CA_output1 = CA_output1.unsqueeze(0)  # Add a batch dimension
                flattened_cnn_output = nn.AdaptiveAvgPool1d(1)(CA_output1.transpose(2, 1)).squeeze()
                #print('flattened_hidden shape',flattened_hidden.shape)
                #print('flattened_cnn_output shape',flattened_cnn_output.shape)
                
                concatenated_features = torch.cat((flattened_cnn_output, flattened_hidden), dim=0).to(device)
                #print('concatenated_features',concatenated_features.shape)
                # Fully connected layer
                fc_output = fc_layer(concatenated_features)
                #print('fc_output  ',fc_output,' labels',labels)
                fc_output=fc_output.unsqueeze(0)
                #probabilities = probabilities.unsqueeze(0)
                #labels = torch.tensor(labels.clone().detach()).cuda()
                #labels = labels.squeeze(0)
                labels=labels.float().to(device)
                probabilities = torch.softmax(fc_output, dim=1) 
            
                #print(labels)
                loss = torch.nn.BCELoss(weight=weight)(probabilities, labels)
                       
                #optimizer.zero_grad()              
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            # Print the average loss for this epoch
            average_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss:.4f}')
            # Validation after each epoch 
            val_loss, val_accuracy = validate_model(val_loader, bigru_model, cnn_models,cross_attention,fc_layer, num_cnn,epoch)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            log_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n')
            # Save the model with the lowest validation loss
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                save_models(bigru_model, cnn_models, fc_layer,cross_attention)


# Training
train_model(train_loader, val_loader, bigru_model, cnn_models,cross_attention, fc_layer, optimizer, num_epochs, device,log_file_path)
