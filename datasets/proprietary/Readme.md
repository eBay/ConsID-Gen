# video_eBay

## Reconstructing and Extracting the Dataset

The dataset is split into multiple parts in the `datasets/images/` and `datasets/videos/` directories. To reconstruct and extract the dataset files, follow these steps:

1. **Navigate to the directory containing the split files.**
   For example, to reconstruct the videos dataset:
   ```sh
   cd datasets/videos
   ```
   Or for images:
   ```sh
   cd datasets/images
   ```

2. **Combine the split files into a single zip archive:**
   ```sh
   cat part_* > large.zip
   ```

3. **Unzip the combined archive to extract the dataset files:**
   ```sh
   unzip large.zip
   ```

After extraction, you will find the dataset files in the respective directory.

---

## sample_script.ipynb

This notebook provides utilities for analyzing the metadata files in this project:

- It loads and summarizes metadata from `metadata.json`, `metadata_train.json`, and `metadata_test.json`, printing the count of unique values for each field.
- It prints details for the first 10 items in `metadata_test.json`, including item ID, video path, video prompt, and corresponding image folder.
- Basically the script describes how the metadata can be used to link videos with their prompts and associated images.

### How to run

1. Ensure you have Python and Jupyter installed. If not, install them:
   ```sh
   pip install notebook
   ```
2. Start Jupyter Notebook in the project directory:
   ```sh
   jupyter notebook
   ```
3. Open `sample_script.ipynb` 