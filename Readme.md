# Fine-tuning Instructions

This document outlines the steps to prepare the data, fine-tune the model, and use the custom metric.

## 1. Data Preparation and Image Generation

The fine-tuning process uses specific pre-processed versions of the AirFrance dataset, not the original raw data.

1.  **Download Datasets**: Download the required low-resolution and high-resolution datasets from the following URLs:
    *   **Low-Resolution (`AirfRans_remeshed`)**: [https://zenodo.org/records/12207787](https://zenodo.org/records/12207787)
    *   **High-Resolution (`AirfRans_clipped`)**: [https://zenodo.org/records/12515064](https://zenodo.org/records/12515064)

2.  **Extract Data**: The downloaded archives contain `.cgns` simulation files. Run the `cgnsExtract.ipynb` notebook to extract scalar data from these files. This process will generate a summary file: `mesh_data_summary.csv`.

3.  **Generate Images**: Run the `cgnsImage.ipynb` notebook. This will process the extracted data and generate images for each sample, saving them into the `raw_data_images/` directory.

## 2. Custom LPIPS Model

All files related to the custom LPIPS model are located in the `./LPIPS_custom` directory.

Before proceeding, move the generated `raw_data_images/` folder and the `mesh_data_summary.csv` file into the `./LPIPS_custom` directory.

### Training

Run the `finetune_extracted_images.ipynb` notebook to train the model. Upon completion, the trained model weights will be saved as a `.h5` file.

### Weight Conversion

To use the weights with PyTorch, convert the Keras `.h5` file to a `.pth` file. Run the conversion script, ensuring you provide the correct input filename:

```bash
python convert_weights.py your_model.h5
```

This will create a `your_model.pth` file.

### Usage

The fine-tuned metric can now be used in your code. Instantiate the `CustomFeatureMetric` and pass the path to your trained `.pth` file.

```python
# Import the custom metric
from LPIPS_custom.custom_metric import CustomFeatureMetric

# Initialize the model with the path to the converted weights
cfd_lpips_model = CustomFeatureMetric(custom_vgg_path='path/to/your_model.pth')

# Calculate the distance score between two images (img1, img2)
distance_score = cfd_lpips_model(img1, img2)
```
