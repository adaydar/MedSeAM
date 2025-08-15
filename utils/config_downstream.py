import torch

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir1 = "../Knee_png"
root_dir2 = "../Knee_entropy.png"
test_root_dir1 = "../test_Knee_png"
test_root_dir2 = "../test_Knee_entropy.png"

#Here Iam Initializing CNNs and BiGRU
num_cnn = 2  # Number of CNNs not needed for now
num_epochs=150

CNN_MODEL_PATH = "../cnn_model.pth"
BIGRU_MODEL_PATH = "../bigru_model.pth"
FC_lAYER_PATH = "../fc_layer_state_dict.pth"
CROSS_ATTENTION_PATH = '../MDFCA_cross_attention.pth'

#Downstream_task
LOG_FILE_DOWNSTREAM = "../downstream.txt"
CNN_MODEL_PATH_DOWNSTREAM = "../downstream_cnn_model.pth"
BIGRU_MODEL_PATH_DOWNSTREAM = "../downstream_bigru_model.pth"
FC_lAYER_PATH_DOWNSTREAM = "../downstream_fc_layer_state_dict.pth"
CROSS_ATTENTION_PATH_DOWNSTREAM = '../downstream_MDFCA_cross_attention.pth'
