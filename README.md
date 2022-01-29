# MT-UNet

This repo contains python codes used for **"Multi-task UNet: Jointly Boosting Saliency Prediction and Disease Classification on Chest X-ray Images"**, Hongzhi Zhu et al., submitted to Medical Imaging with Deep learning (MIDL) 2022.

- Folder _Data_ contain files used for data pre-processing (before training).
- Folder _Module_ contain files used for network training and evaluation.

The following packages are used with Python 3.7:
- PyTorch 1.8.0
- torchvision 0.9.0
- tensorboard 2.3.0
- pandas 1.1.0
- matplotlib 3.3.1
- numpy 1.19.2
- scipy 1.5.2
- sklearn 0.23.2
- opencv 4.0.1
- pydicom 2.2.0

## _Data_ folder

To run the code, simply execute data.py through “python data.py” in command line or with other Python IDEs. After execution, new folders containing per-processed and split datasets ready for training will be created in the execution directory. The raw dataset is partitioned into training (70%), validation (10%) and testing (20%) subsets. Seeding (value 0) is used for reproducibility.

## _Module_ folder

To run the code, simply execute RUN.py through “python RUN.py” in command line or with other Python IDEs. After execution, a new folder _run_ will be created (if not already exists) in the execution directory. For each execution of RUN.py, a new folder with a random name will be created inside folder _run_ for the temporary storage of network parameters as well as recoding training details and evaluation results. The following lists the file details inside the folder:
- params_MTL_UNet_preset_.txt, records parameters and data related to network training.
- params_NetLearn_.txt, records hyper-parameters for the network.
- QuickHelper_summary.txt, records other data during execution.
- NET_XXXX/classification_report.json, records classification evaluation metrics, and XXXXX are random characters generated during run time.
- NET_XXXX/classification_results.csv, records raw classification output for each test image.
- NET_XXXX/prediction_report.json, records classification and saliency prediction performance matrices.
- NET_XXXX/NET.pt, stores the parameters for the best performing network during or after training.
- NET_XXXX/training_process.png, visualizes the change of learning rate, losses, and validation metrics during the training process.
During execution, RUN.py will also print training details and evaluation results to the console.
