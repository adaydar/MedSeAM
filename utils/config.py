import torch

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
TRAIN_DIR1 = "../Knee_png"
TRAIN_DIR2 = "../Knee_png_entropy"
LOG_FILE = "../pretraining.txt"
CNN_MODEL_PATH = "cnn_model.pth"
BIGRU_MODEL_PATH = "bigru_model.pth"
FC_lAYER_PATH = "fc_layer_state_dict.pth"
CROSS_ATTENTION_PATH = 'MDFCA_cross_attention.pth'

# Model hyperparameters
NUM_CLASSES = 2
INPUT_SIZE = 2304
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 1
NUM_EPOCHS = 150
LR = 0.0001
STEP_SIZE = 10
GAMMA = 0.1

def shuffle_data(tensor, labels):
    order = torch.randperm(tensor.size(0))
    return tensor[order], labels[order], order

def save_models(bigru_model, cnn_models):
    torch.save(cnn_models.state_dict(), CNN_MODEL_PATH)
    torch.save(bigru_model.state_dict(), BIGRU_MODEL_PATH)
    torch.save(fc_layer.state_dict(),FC_lAYER_PATH)
    torch.save(cross_attention.state_dict(),CROSS_ATTENTION_PATH)
