import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils.config import *
from utils.config_downstream import *
from utils.dataset import *
from utils.MDFCA import *
from utils.models import *
from utils.eval_metrics import * 

def test_model(val_loader, bigru_model, cnn_models, cross_attention,fc_layer, num_cnn,epoch=0):
    # Set models to evaluation mode
    bigru_model.eval()
    cnn_models.eval()
    cross_attention.eval()
    fc_layer.eval()

    # Initialize metrics for validation
    total_val_loss = 0
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []
    all_concatenated_features = []  # Collect concatenated features
    all_labels_tsne = [] 
    all_data = {'Patient_ID': [],'Image': [], 'CNN_Output': [], 'BiGRU_Probability': []}
    with torch.no_grad():
        for i, (sequence_batch,spectral_batch,labels,patient_id,weight) in enumerate(val_loader):
            weight=weight.to(device)    
            cnn_output1,pool1 = cnn_models(sequence_batch[0].float().to(device))
            cnn_output2,pool2=cnn_models(spectral_batch[0].float().to(device))
            CA_output1=cross_attention(cnn_output1,cnn_output2)       
            CA_output1 = CA_output1.view(17, -1)
            #GAP_spatial = F.adaptive_avg_pool1d(batches.unsqueeze(0), 1).squeeze(0)
            #print('GAP_spatial.shape ',GAP_spatial.shape)
            features = CA_output1.unsqueeze(0).to(device) # I did this because my bigru input is batch first i.e (batch_size,seq_len, input)            
            bigru_output,temporal_features = bigru_model(features)
            #print(bigru_output.squeeze(0).shape)
            cnn_output_ga = torch.mean(CA_output1, dim=1)
            #print(cnn_output_ga) 
            cnn_output_ga_list = [round(value.item(), 4) for value in cnn_output_ga]
            probs = torch.softmax(bigru_output.squeeze(0), dim=1)
            probabilities_list = [round(value.item(), 4) for value in probs[:, 1]]
            #print(probabilities_list)
            CA_output1 = CA_output1.unsqueeze(0)  # Add a batch dimension
            flattened_cnn_output = nn.AdaptiveAvgPool1d(1)(CA_output1.transpose(2, 1)).squeeze()
            #print('temporal_features.shape ',temporal_features.shape)
            flattened_hidden = temporal_features.permute(1, 0, 2).contiguous().view(-1)
            #print('flattened_hidden shape',flattened_hidden.shape)
            #print('flattened_cnn_output shape',flattened_cnn_output.shape)
            concatenated_features = torch.cat((flattened_cnn_output, flattened_hidden), dim=0).to(device)
            #print('concatenated_features.shape ',concatenated_features.shape)
            fc_output = fc_layer(concatenated_features)
            fc_output=fc_output.unsqueeze(0)
            fc_output=fc_output.float().to(device)
            labels=labels.float().to(device)
            probabilities = torch.softmax(fc_output, dim=1)
            loss=0
            loss = torch.nn.BCELoss(weight=weight)(probabilities, labels)
            #print('predicted and actual ',fc_output,'    ',labels)
            # Here Iam getting predicted class (index with highest probability)
            #print('predicted ',probabilities,'  actual',labels[0][1])
            predicted_classes = (probabilities[:, 1] > 0.5).float()
            #print('predicted_classes',predicted_classes,'  ','labels[0][1] ',labels[0][1])
            correct_predictions += (predicted_classes == labels[0][1]).item()
            total_samples += sequence_batch.size(0)
            #print("seq batch",sequence_batch.size(0))
            total_val_loss += loss.item()
            binary_predictions = (probabilities[:, 1] > 0.5).cpu().numpy()
            all_predictions.extend(binary_predictions)
            all_labels.extend(labels[:, 1].cpu().numpy())
            all_data['Patient_ID'].extend([patient_id.item()] * len(cnn_output_ga_list))
            all_data['Image'].extend(list(range(1, len(cnn_output_ga_list) + 1)))
            all_data['CNN_Output'].extend(cnn_output_ga_list)
            all_data['BiGRU_Probability'].extend(probabilities_list)
            all_concatenated_features.append(concatenated_features.cpu().numpy())
            all_labels_tsne.append(labels[0][0].cpu().numpy())

    # Convert lists to numpy arrays
    all_concatenated_features = np.array(all_concatenated_features)
    all_labels = np.array(all_labels_tsne)

    # Flatten the concatenated features
    flattened_features = all_concatenated_features.reshape(all_concatenated_features.shape[0], -1)

    # Convert tensor labels to numpy arrays
    #all_labels = np.array([label.cpu().numpy() for label in all_labels])

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    tsne_embeddings = tsne.fit_transform(flattened_features)
    # Define labels for the legend
    num_categories = len(np.unique(all_labels))
    legend_labels = ["No Abnormality", "Abnormality"]  # Corresponding to 0 and 1

    # Define colors for each category
    category_colors = ['blue', 'red']  # Blue for "No Abnormality" (0) and Red for "Abnormality" (1)

    for lab in range(num_categories):
        indices = all_labels.squeeze() == lab
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], c=category_colors[lab], label=legend_labels[lab], alpha=0.5)

    # Show legend
    plt.legend(fontsize='large', markerscale=2)
    plt.savefig("../tsne_plot.png", dpi=400)
    plt.show()

         
    average_test_loss = total_val_loss / len(val_loader)
    accuracy = correct_predictions / (len(val_loader))
    #print('type of average_val_loss',type(average_val_loss))
    #print('type of accuracy ',type(accuracy))
    sensitivity, specificity, recall = calculate_metrics(np.array(all_predictions), np.array(all_labels))
    auc_score = calculate_auc(np.array(all_predictions), np.array(all_labels))
    df = pd.DataFrame(all_data)
    df.to_csv('../all_patients_output.csv', index=False)
    return average_test_loss, accuracy, sensitivity, specificity, recall, auc_score

dataset = ImageDataset(test_root_dir1,test_root_dir2, test_csv_path)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)


# Initialize CNN models
cnn_models = CNN().cuda()

# Load saved CNN models
cnn_models.load_state_dict(torch.load(CNN_MODEL_PATH_DOWNSTREAM))
hidden_size=64
# Initialize and load the bigru model
bigru_model = BiGRU(input_size=2304, hidden_size=64, num_layers=2, num_classes=num_cnn).to(device)
bigru_model.load_state_dict(torch.load(BIGRU_MODEL_PATH_DOWNSTREAM))

cross_attention = MDFCA_CrossAttention().to(device)
cross_attention.load_state_dict(torch.load(CROSS_ATTENTION_PATH_DOWNSTREAM))
input_size=2560
output_size=2
fc_layer = FCNN(input_size, output_size).to(device)
fc_layer.load_state_dict(torch.load(FC_lAYER_PATH_DOWNSTREAM))

test_loss, test_accuracy, sensitivity, specificity, recall, auc_score = test_model(test_loader, bigru_model, cnn_models,cross_attention,fc_layer, num_cnn,0)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
print(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Recall: {recall:.4f}, AUC: {auc_score:.4f}')
