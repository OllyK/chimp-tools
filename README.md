# CHiMP Tools

Code to create and use the CHiMP (Crystal Hits in My Plate) classification and object detection networks

## Installation

Clone the repository and then create a conda environment using the `conda_requirements.yaml`.

```shell
conda env create --name chimp_env --file conda_requirements.yaml
```

Activate the environment and you should then be able ot run the scripts.

## Train a classifier network on images and associated labels

The `train_classifier.py` script is used to train an image classifier. It was used to create the [CHiMP Classifier-v2 model](https://doi.org/10.5281/zenodo.11190974) by fine-tuning a pre-trained ConvNeXt-Tiny CNN first on the [MARCO Dataset](https://marco.ccr.buffalo.edu/download) of 462,804 images and then on the [VMXi Classification Dataset](https://zenodo.org/doi/10.5281/zenodo.11097395) of 13,951 images.

It takes a CSV file as input, which should contain the training data in the form of two columns, the first being the filename and the second being the associated label. The script provides several options to customize the training process.

Here's how to use it from the command line:

```shell
python train_classifier.py [OPTIONS] CSV_FILEPATH
```

Here are the available options:

`--settings_file` SETTINGS_FILE: Path to the settings file. Default is the value of CLF_TRAIN_SETTINGS_FN in the utilities.config module the default is currently `classifier_train_settings.yaml`.

`--prepend_dir` PREPEND_DIR: Prepend this directory if paths in CSV are relative rather than absolute.

`--finetune` FINETUNE: Path to existing model to load in for further training.

`--imbalanced`: Use this option to upsample under-represented classes in imbalanced data.

`--valid_data` VALID_DATA: Path to validation dataset CSV (if data has been split before training).

`--fix_seed`: Use this option to use the random_seed defined in settings to fix the train/valid split.

`--reload_optimizer`: Use this option when adding training to a model to load in previous optimizer weights.

`--distributed`: Use this option to perform training distributed across multiple GPUs.

`--use_accuracy`: Use this option to maximize accuracy metric for saving model and early stopping rather than minimizing validation loss.

Here's another example of how to use the script:

```shell
python train_classifier.py --settings_file settings.json --finetune model.pth --imbalanced --use_accuracy training_data.csv
```

This command will train the classifier using the settings in settings.json, fine-tuning from model.pth, upsampling under-represented classes, and maximizing accuracy instead of minimizing validation loss. The training data is provided in training_data.csv.

## Predict labels for images in a directory using a classifier network

The classify_folder.py script is used to classify images in a directory using a pre-trained model. It takes two required command-line arguments and one optional argument:

`--model_path`: This is the path to the pre-trained model file.

`--image_dir`: This is the path to the directory containing the images to be classified.

`--img_size`: This is the size to which the images will be resized for inference. It defaults to 384 if not provided.

To run the script, you would use a command like the following in your terminal:

```shell
python classify_folder.py --model_path path/to/model --image_dir path/to/images
```

Replace `path/to/model` and `path/to/images` with the paths to your model file and image directory, respectively. If you want to specify a different image size for inference, you can use the `--img_size` option:

```shell
python classify_folder.py --model_path path/to/model --image_dir path/to/images --img_size 512
```

The script will output a CSV file in the current working directory, with the filename in the format `YYYY-MM-DD_predictions.csv`, where YYYY-MM-DD is the current date. This file will contain the predicted classifications for the images in the directory.

This script can be used to predict experimental outcomes from micrographs using the [CHiMP Classifier-v2 model](https://doi.org/10.5281/zenodo.11190974).

## Train a Mask-R-CNN object and instance detection network on images and masks

To train a detector run the script `train_detector.py`. This script is used to train a Mask R-CNN model for object detection and was used to train networks for protein crystal detection at Diamond Light Source, UK on data which can be found [here](https://doi.org/10.5281/zenodo.11110373).

The script takes several command-line arguments:

`--settings_file`: This is the path to the settings YAML file. If not provided, it will use the default settings file defined in cfg.DET_TRAIN_SETTINGS_FN and is currently set to `detector_train_settings.yaml`.

`--finetune`: This is the path to an existing model to load in for further training. If not provided, the script will train a new model from scratch.

`--fix_seed`: This is a flag. If provided, the script will use the random seed defined in the settings to fix the train/valid split.

`data_dir_path`: This is a positional argument that specifies the path to the directory containing the training data.

To run the script, you would use a command like the following in your terminal:
```shell
python train_detector.py --settings_file path/to/settings/file --finetune path/to/model --fix_seed data_directory_path
```

Replace `path/to/settings/file`, `path/to/model`, and `data_directory_path` with the paths to your settings file, pre-trained model, and data directory, respectively. If you want to use the default settings file and train a new model from scratch, you can omit the `--settings_file` and `--finetune options`:

```shell
python train_detector.py --fix_seed data_directory_path
```

The script will output a trained model file in the current working directory, with the filename in the format `YYYY-MM-DD_model_output_fn.pytorch`, where YYYY-MM-DD is the current date and model_output_fn is defined in the settings file. It will also output a figure showing the loss during training.

## Detect crystal positions for images in a directory using a trained Mask-R-CNN network

The script `detect_folder.py` is used to detect positions of crystals in a folder of drop images using a Mask-R-CNN based object detector. Model weights for a network trained to detect crystals can be found [here](https://doi.org/10.5281/zenodo.11164788) and model weights to detect crystals and drops can be found [here](https://doi.org/10.5281/zenodo.11165195).  The script several command-line arguments:

`--MODEL_PATH`: This is the path to the pre-trained model file. This argument is required.

`--IMAGE_PATH`: This is the path to the directory containing the images to be processed. This argument is required.

`--num_classes`: This is the number of classes that the model was trained on. It defaults to 3 if not provided.

`--mode`: This is the mode for point generation. One of "SINGLE", "RANDOM" and "REGULAR. If "SINGLE", one point is generated per crystal, "REGULAR" and "RANDOM" generate multiple points per crystal for larger crystals on a regular grid or on a grid with a randomised offset respectively. Defaults to "SINGLE".

`--preview`: If this flag is provided, the script will output preview images.

`--masks`: If this flag is provided, the script will output masks.

`--echo`: If this flag is provided, the script will output a candidate position for compound dispensing using an echo dispenser.


To run the script, you would use a command like the following in your terminal:

```shell
python detect_folder.py --MODEL_PATH path/to/model --IMAGE_PATH path/to/images
```

The script will output a CSV file with the detected positions of the crystals in the current working directory, with the filename `detector_positions.csv`. Replace `path/to/model` and `path/to/images` with the paths to your model file and image directory, respectively. If the `--preview` flag is provided, the script will also output preview images in a directory named `preview_images`.

## Citation

If you use these tools for your research, please cite:

CHiMP: Deep Learning Tools Trained on Protein Crystallisation Micrographs to Enable Automation of Experiments. Preprint. bioRxiv: https://biorxiv.org/cgi/content/short/2024.05.22.595345v1

```bibtex
@misc{King2024,
    doi = {10.1101/2024.05.22.595345},
    url = {https://doi.org/10.1101/2024.05.22.595345},
    year = {2024},
    publisher = {bioRxiv},
    author = {Oliver N. F. King and Karl E. Levik and James Sandy and Mark Basham},
    title = {CHiMP: Deep Learning Tools Trained on Protein Crystallisation Micrographs to Enable Automation of Experiments} }
```

