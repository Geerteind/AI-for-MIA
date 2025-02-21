import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def visualize_Images(directory, nr_samples=5):
    """This function visualizes random image samples grouped by presence of metastasis.
    Args:
        directory (string): The directory of the training and/or validation folder.
        nr_samples (int): The number of samples that are shown on each row of the image. Defaults to 5.
    """
    classes = {"0": "Without Metastasis", "1": "With Metastasis"}     #train and validation folder exists of a '0' and a '1' folder 
    samples = {}	
    for Group in classes.keys():
        samples[Group] = get_sample_images(directory, Group, nr_samples)            #get random samples for each class


    fig, axes = plt.subplots(2, nr_samples, figsize=(15, 6))          #create figure
    fig.suptitle("Comparison of tissue with or without metastasis", fontsize=14)


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
    """This function takes random samples from the class directory.

    Args:
        directory (string): The directory of the training and/or validation folder.
        Classnr (int): number of the class "0" means without and "1" means with metastasis.
        num_samples (int): The number of samples that are shown on each row of the image. Defaults to 5.

    Returns:
        list: A list of randomly selected image filenames from the class directory.
    """
    Class_path = os.path.join(directory, Classnr)
    image_files = [f for f in os.listdir(Class_path)]
    return random.sample(image_files, min(num_samples, len(image_files)))               #take random samples from data


# Run for training dataset, change the directory when necessary.
visualize_Images(r"path")


# Run for validation dataset, if images from validation dataset are wanted.
#visualize_Images(r"path")
