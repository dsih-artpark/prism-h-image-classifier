import os
import json
import random
import requests
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

class ImageProcessor:
    """
    A class to handle various tasks on a JSON dataset of image URLs:
    1) Counting image extensions (.jpg, .png, etc.)
    2) Analyzing invalid image URLs
    3) Downloading PNG images
    4) Displaying random images from a folder
    """

    def __init__(self, json_files, save_folder, data_key="data"):
        """
        :param json_files: List of paths to JSON files containing image records.
        :param save_folder: Folder where images will be downloaded.
        :param data_key: The key under which the list of records is stored in JSON.
        """
        self.json_files = json_files
        self.save_folder = save_folder
        self.data_key = data_key
        self.data = {}  # Store data from each JSON file
        self.extension_counts = {}

    def load_json(self):
        """Load JSON data from all provided files."""
        for json_file in self.json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                self.data[json_file] = json.load(f).get(self.data_key, [])
            print(f"Loaded {len(self.data[json_file])} records from {json_file}")

    def count_image_extensions(self):
        """Count how many URLs end with .jpg, .png, or other extensions."""
        for json_file, records in self.data.items():
            jpg_count, png_count, other_count = 0, 0, 0

            for entry in tqdm(records, desc=f"Counting extensions in {json_file}"):
                url = entry.get("image_url", "").lower()
                if url.endswith(".jpg"):
                    jpg_count += 1
                elif url.endswith(".png"):
                    png_count += 1
                else:
                    other_count += 1

            self.extension_counts[json_file] = {
                "jpg": jpg_count,
                "png": png_count,
                "other": other_count
            }

    def analyze_invalid_entries(self, sample_size=10):
        """Print details of invalid image URLs that don't end in .jpg or .png."""
        for json_file, records in self.data.items():
            invalid_entries = [entry for entry in records if not entry.get("image_url", "").lower().endswith((".jpg", ".png"))]

            print(f"Total invalid entries in {json_file} = {len(invalid_entries)}")
            for i, entry in enumerate(invalid_entries[:sample_size], 1):
                print(f"\nInvalid Entry #{i} from {json_file}:")
                for key, value in entry.items():
                    print(f"  {key}: {value}")

    def download_image(self, image_url, image_id):
        """Download a single image to the specified folder, using the provided image URL and ID."""
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                ext = image_url.split(".")[-1].lower()  # Get file extension
                file_path = os.path.join(self.save_folder, f"{image_id}.{ext}")
                with open(file_path, 'wb') as out_file:
                    for chunk in response.iter_content(1024):
                        out_file.write(chunk)
                return True
        except requests.RequestException:
            return False
        return False

    def download_images(self):
        """Download only PNG images from 'first100k.json' while skipping the second JSON file."""
        os.makedirs(self.save_folder, exist_ok=True)

        for json_file, records in self.data.items():
            if "first100k.json" in json_file:
                for entry in tqdm(records, desc=f"Downloading PNG images from {json_file}"):
                    image_url = entry.get("image_url", "").lower()
                    image_id = entry.get("Id")
                    if image_url.endswith(".png") and image_id:
                        success = self.download_image(image_url, image_id)
                        if not success:
                            print(f"Failed to download {image_url}")
            else:
                print(f"Skipping downloads from {json_file}")

    def show_random_images(self, num_images=30, rows=3, cols=10):
        """Display a grid of randomly selected images from the saved folder."""
        valid_extensions = ('.jpg', '.jpeg', '.png')
        all_images = [f for f in os.listdir(self.save_folder) if f.lower().endswith(valid_extensions)]

        if len(all_images) < num_images:
            raise ValueError(f"Not enough images in {self.save_folder} to select {num_images}.")

        selected_files = random.sample(all_images, num_images)
        plt.figure(figsize=(cols * 2, rows * 2))

        for idx, filename in enumerate(selected_files, start=1):
            img_path = os.path.join(self.save_folder, filename)
            with Image.open(img_path) as img:
                plt.subplot(rows, cols, idx)
                plt.imshow(img)
                plt.axis('off')

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    # Define JSON files and save location
    json_files = [
        "/Users/kirubeso.r/Documents/ArtPark/all_jsons/first100k.json",
        "/Users/kirubeso.r/Documents/ArtPark/all_jsons/second100k.json"
    ]
    save_folder = "/Users/kirubeso.r/Documents/ArtPark/all_them_images"

    # Create an instance of ImageProcessor
    processor = ImageProcessor(json_files, save_folder)

    # Display 30 random images in a 3x10 grid
    processor.show_random_images(num_images=30, rows=3, cols=10)