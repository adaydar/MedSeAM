This repo contains the Official Pytorch implementation of our paper at ICVGIP'24:

[Med-SeAM: Medical Context Aware Self-Supervised Learning Framework for Anomaly Classification in Knee MRI](https://dl.acm.org/doi/full/10.1145/3702250.3702276) by Akshay Daydar, Ajay Kumar Reddy, Sonal Kumar, Arijit Sur, Hanif Laskar

![MedSeAM_architecture](./MedSeAM_final_arch.png) Figure: Overall schematic of the proposed MedSeAM model with Stage I includes the Disease Context Learning pretext task, and Stage II includes the downstream task of abnormality classification in knee MRI scans. * indicates the pre-trained modules that are fine-tuned on downstream tasks.

Requirements

    Linux
    Python3 3.8.10
    Pytorch 1.13.1
    train and test with A100 GPU

Prepare Dataset:

    1. Kindly check "https://github.com/adaydar/MtRA-Unet/tree/main" repository for dataset preparation.
    2. Kindly check preprocessing_entropy_dataset.py in utils folder for entropy map generation.
    2. Then kindly check the "config" file before running the training and testing code.

Training and Testing:

Prepare the dataset and then run the following command for pre-training:

    python3 pretraining.py

For Downstream classification task, run
    
    python3 downstream.py
    
For Testing, run

    python3 test.py

Citation:
 If you find this repo useful for your research, please consider citing our paper:
 
       Akshay Daydar, Ajay Kumar Reddy, Sonal Kumar, Arijit Sur, and Hanif Laskar. 2025. Med-SeAM: Medical Context Aware Self-Supervised Learning Framework for Anomaly Classification in Knee MRI. In Proceedings of the Fifteenth Indian Conference on Computer Vision Graphics and Image Processing (ICVGIP '24). Association for Computing Machinery, New York, NY, USA, Article 26, 1–8. https://doi.org/10.1145/3702250.3702276
