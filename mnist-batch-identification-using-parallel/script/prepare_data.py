import os
import argparse
import gzip
import idx2numpy
import urllib.request
from PIL import Image

def prepare_mnist_data(output_folder):
    """
    Downloads the MNIST test images and saves the first 1000 images as PNG files in the specified output folder.

    Args:
        output_folder (str): The path to the folder where the images will be saved.

    Returns:
        None
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(allow_abbrev=False, description="parse user arguments")
    parser.add_argument("--output_folder", type=str, default=0)
    args, _ = parser.parse_known_args()

    # Create the output folder if it doesn't exist
    data_folder = os.path.join(output_folder, "mnist")
    os.makedirs(data_folder, exist_ok=True)

    # Download the MNIST test images
    urllib.request.urlretrieve(
        "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz",
        filename=os.path.join(os.getcwd(), "test-images.gz"),
    )

    # Read the image data from the gzip file
    file_handler = gzip.open("test-images.gz", "r")
    imagearray = idx2numpy.convert_from_file(file_handler)

    # Choose the first 1000 images and save them as PNG files
    for i in range(1000):
        im = Image.fromarray(imagearray[i])
        im.save(os.path.join(data_folder, f"{i}.png"))

    print("Saved 1000 images to the output folder")
