import os
import numpy as np
from PIL import Image
import insightface
from sklearn.cluster import KMeans
from shutil import copy2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        images.append((filename, img_path))
    return images
def get_face_encodings(images, face_detector, face_recognition_model):
    encodings = []
    for filename, img_path in images:
        img = Image.open(img_path)
        img = np.array(img)
        faces = face_detector.get(img)  # Use `get` method for detection and alignment
        for face in faces:
            bbox = face['bbox']
            if len(bbox) == 4:
                try:
                    bbox = list(map(int, bbox))  # Convert bbox to integers
                    face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
                    face_img = np.transpose(face_img, (0, 3, 1, 2))  # Change format to (N, C, H, W)
                    encoding = face_recognition_model.get_embeddings(face_img)[0]
                    encodings.append((filename, encoding))
                except Exception as e:
                    print(f"Error processing face in file {filename}: {e}")
            else:
                print(f"Invalid bbox format for file {filename}: {bbox}")
    return encodings

def cluster_faces(encodings):
    face_encodings = [encoding for _, encoding in encodings]
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)  # Adjust as necessary
    cluster_labels = kmeans.fit_predict(face_encodings)
    return cluster_labels

def save_images_to_labels(images, cluster_labels, base_folder):
    for (filename, _), label in zip(images, cluster_labels):
        label_folder = os.path.join(base_folder, f'face{label}')
        os.makedirs(label_folder, exist_ok=True)
        try:
            copy2(os.path.join('images', filename), os.path.join(label_folder, filename))
        except Exception as e:
            print(f"Error copying file {filename} to {label_folder}: {e}")

def main():
    folder = 'images'
    base_folder = 'output'

    # Initialize face detector and recognition model
    face_detector = insightface.app.FaceAnalysis()
    face_detector.prepare(ctx_id=-1)  # -1 means CPU, or you can set to 0 for GPU
    face_recognition_model = insightface.model_zoo.get_model('buffalo_l')

    images = load_images_from_folder(folder)
    encodings = get_face_encodings(images, face_detector, face_recognition_model)
    if encodings:
        cluster_labels = cluster_faces(encodings)
        save_images_to_labels(images, cluster_labels, base_folder)
    else:
        print("No face encodings found.")

if __name__ == "__main__":
    main()
