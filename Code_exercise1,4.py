import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def visualize_Images(directory, nr_samples=5):
    classes = {"0": "Without Metastases", "1": "With Metastases"}     #train and validation map exists of a '0' and a '1' map
    samples = {}	
    for Group in classes.keys():
        samples[Group] = get_sample_images(directory, Group, nr_samples)
    print(samples)


    fig, axes = plt.subplots(2, nr_samples, figsize=(15, 6))          #create figure
    fig.suptitle("Comparison of tissue with or without metastases", fontsize=14)


    for i, Group in enumerate(classes.keys()):
        for j, img_name in enumerate(samples[Group]):
            img_path = os.path.join(directory, Group, img_name)
            img = Image.open(img_path)

            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:  # Label only the first image in each row
                axes[i, j].set_title(classes[Group], fontsize=12)

    plt.tight_layout()
    plt.show()


#Function to get the sample images from each class
def get_sample_images(directory, Classnr, num_samples=5):
    Class_path = os.path.join(directory, Classnr)
    image_files = [f for f in os.listdir(Class_path) if f.endswith((".jpg"))]
    return random.sample(image_files, min(num_samples, len(image_files)))               #take random samples from data


# Run for training dataset
visualize_Images("train")


# Run for validation dataset
visualize_Images("valid")
