# Hugging Face Dataset Visualizer

A Streamlit web application for visualizing Hugging Face datasets. Supports text, images, and videos with an easy-to-use interface.

## Features

- 🔍 Load any dataset from Hugging Face Hub by name
- 📊 View dataset information and available splits
- 📝 Text visualization support
- 🖼️ Image visualization support (both PIL images and numpy arrays)
- 🎥 Video visualization support
- 🎛️ Configurable sample size for visualization
- 🎲 Random sampling option
- ⚠️ Comprehensive error handling

## Installation

1. Create a conda environment:

```
conda create -n dataset_viewer python=3.10
conda activate dataset_viewer
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Run the app:
```
streamlit run dataset_visualizer.py
```

2. Enter a dataset name in the input field
3. Select the split (train/test/validation)
4. Choose number of samples to view (1-10)
5. Use "Get First K Samples" or "Get Random K Samples"

## Example Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| `mnist` | Image | Handwritten digits (28x28 grayscale) |
| `fashion_mnist` | Image | Fashion items (28x28 grayscale) |
| `cifar10` | Image | Natural images (32x32 color) |
| `beans` | Image | Plant disease images |
| `liuhaotian/LLaVA-Instruct-150K` | Image+Text | Visual instruction tuning dataset |

## Project Structure

```
.
├── dataset_visualizer.py  # Main Streamlit application
├── visualizer.py         # Visualization components
├── requirements.txt      # Python dependencies
└── README.md            # Documentation
```

## Features in Detail

### Image Visualization
- Supports multiple image formats:
  - PIL Images
  - NumPy arrays
  - Image bytes
- Automatic size adjustment (400px width)
- Handles both color and grayscale

### Dataset Information
- Shows available splits (train/test/validation)
- Displays feature types and descriptions
- Shows total number of samples in each split

### Sampling Options
- Sequential sampling (first K samples)
- Random sampling (random K samples)
- Configurable sample size (1-10)

### Complex Data Visualization
- Supports conversation format
- Handles JSON strings and nested structures
- Displays image-text pairs
- Shows metadata and additional information

## Requirements

- Python 3.10+
- streamlit>=1.32.0
- datasets (Hugging Face)
- Pillow
- numpy

## Troubleshooting

Common issues and solutions:
1. Dataset loading fails
   - Check if dataset exists on HuggingFace
   - Verify internet connection
   - Try example datasets first

2. Images not displaying
   - Check dataset format
   - Verify image feature name ('image', 'img', or 'pixel_values')

## License

MIT License
```

Key updates:
1. Added detailed features section
2. Created a table for example datasets
3. Added troubleshooting section
4. Improved formatting with tables and code blocks
5. Added more specific usage instructions
6. Included feature details section