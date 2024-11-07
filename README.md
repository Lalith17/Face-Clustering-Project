# Face Clustering Project

This project clusters images based on facial similarities using a deep learning face recognition model and K-Means clustering. It utilizes InsightFace for face detection and feature extraction, allowing images with faces to be organized into folders representing different clusters.

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Explanation of Functions](#explanation-of-functions)
- [Notes](#notes)

## Requirements

To run this project, you'll need the following libraries installed:

- Python 3.6 or later
- numpy
- PIL (Pillow)
- insightface
- scikit-learn (for KMeans clustering)

You can install them using:
```bash
pip install numpy pillow insightface scikit-learn
```

## Project Structure

- `images/`: The folder containing input images for clustering.
- `output/`: The output folder where clustered images are saved.
- `main.py`: The main script for running the face clustering process.

## Usage

1. Place the images you want to cluster in the `images/` folder.
2. Run the script:
   ```bash
   python main.py
   ```

3. The output will be organized in the `output/` folder, with subfolders for each cluster (e.g., `face0`, `face1`, etc.).

## Explanation of Functions

### `load_images_from_folder(folder)`
- Loads images from the specified folder.
- Returns a list of image file paths for further processing.

### `get_face_encodings(images, face_detector, face_recognition_model)`
- Extracts face embeddings from images using the InsightFace detector and recognition model.
- Detects faces in each image, crops and aligns the face region, and generates an encoding.
- Returns a list of face encodings.

### `cluster_faces(encodings)`
- Uses K-Means clustering on face encodings to group similar faces together.
- Returns cluster labels for each image.

### `save_images_to_labels(images, cluster_labels, base_folder)`
- Saves each image to a folder corresponding to its cluster label.
- Creates subfolders within the specified base folder to organize the clustered images.

### `main()`
- Orchestrates the entire clustering process, from loading images to saving clustered images.

## Notes

- **Clustering Configuration**: The `KMeans` clustering is currently set to 4 clusters. Modify `n_clusters` in the `cluster_faces` function to change the number of clusters.
- **InsightFace**: This project uses the `buffalo_l` model from InsightFace for facial recognition.
- **Error Handling**: The script includes error handling for common issues, such as unreadable image files or bounding box errors.

## License

This project is open-source and free to use for non-commercial purposes.
