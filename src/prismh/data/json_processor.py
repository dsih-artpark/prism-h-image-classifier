import argparse  # Import argparse
import json
import random
from pathlib import Path  # Import Path

import matplotlib.pyplot as plt
import requests
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
        self.json_files = [Path(f) for f in json_files]  # Convert paths to Path objects
        self.save_folder = Path(save_folder)  # Convert save folder to Path object
        self.data_key = data_key
        self.data = {}  # Store data from each JSON file
        self.extension_counts = {}

    def load_json(self):
        """Load JSON data from all provided files."""
        for json_file in self.json_files:
            with json_file.open("r", encoding="utf-8") as f:  # Use Path.open
                self.data[json_file.name] = json.load(f).get(
                    self.data_key, []
                )  # Use file name as key
            print(f"Loaded {len(self.data[json_file.name])} records from {json_file.name}")

    def count_image_extensions(self):
        """Count how many URLs end with .jpg, .png, or other extensions."""
        for json_file_name, records in self.data.items():  # Iterate over names now
            jpg_count, png_count, other_count = 0, 0, 0

            for entry in tqdm(records, desc=f"Counting extensions in {json_file_name}"):
                url = entry.get("image_url", "").lower()
                if url.endswith(".jpg"):
                    jpg_count += 1
                elif url.endswith(".png"):
                    png_count += 1
                else:
                    other_count += 1

            self.extension_counts[json_file_name] = {
                "jpg": jpg_count,
                "png": png_count,
                "other": other_count,
            }

    def analyze_invalid_entries(self, sample_size=10):
        """Print details of invalid image URLs that don't end in .jpg or .png."""
        for json_file_name, records in self.data.items():  # Iterate over names now
            invalid_entries = [
                entry
                for entry in records
                if not entry.get("image_url", "").lower().endswith((".jpg", ".png"))
            ]

            print(f"Total invalid entries in {json_file_name} = {len(invalid_entries)}")
            for i, entry in enumerate(invalid_entries[:sample_size], 1):
                print(f"\nInvalid Entry #{i} from {json_file_name}:")
                for key, value in entry.items():
                    print(f"  {key}: {value}")

    def download_image(self, image_url, image_id):
        """Download a single image to the specified folder, using the provided image URL and ID."""
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                ext = image_url.split(".")[-1].lower()  # Get file extension
                file_path = self.save_folder / f"{image_id}.{ext}"  # Use Path concatenation
                with file_path.open("wb") as out_file:  # Use Path.open
                    for chunk in response.iter_content(1024):
                        out_file.write(chunk)
                return True
        except requests.RequestException:
            return False
        return False

    def download_images(self):
        """Download PNG and JPG images from the specified JSON files."""
        self.save_folder.mkdir(parents=True, exist_ok=True)  # Use Path.mkdir

        for json_file_name, records in self.data.items():  # Iterate over names now
            print(f"Processing {json_file_name} for image downloads...")
            downloaded_count = 0
            failed_count = 0

            for entry in tqdm(records, desc=f"Downloading images from {json_file_name}"):
                image_url = entry.get("image_url", "").lower()
                image_id = entry.get("Id")

                # Only attempt download if URL is valid (.png or .jpg) and ID exists
                if image_id and image_url.endswith((".png", ".jpg")):
                    success = self.download_image(image_url, image_id)
                    if success:
                        downloaded_count += 1
                    else:
                        failed_count += 1

            print(
                f"Finished {json_file_name}: Downloaded: {downloaded_count}, Failed: {failed_count}"
            )

    def show_random_images(self, num_images=30, rows=3, cols=10):
        """Display a grid of randomly selected images from the saved folder."""
        valid_extensions = (".jpg", ".jpeg", ".png")
        # Use Path.iterdir and Path.suffix
        all_images = [f for f in self.save_folder.iterdir() if f.suffix.lower() in valid_extensions]

        if len(all_images) < num_images:
            # Use f-string for Path object
            raise ValueError(f"Not enough images in {self.save_folder} to select {num_images}.")

        selected_files = random.sample(all_images, num_images)
        plt.figure(figsize=(cols * 2, rows * 2))

        for idx, filepath in enumerate(selected_files, start=1):  # filepath is now a Path object
            # img_path = os.path.join(self.save_folder, filename) # No longer needed
            with Image.open(filepath) as img:  # Open directly with Path object
                plt.subplot(rows, cols, idx)
                plt.imshow(img)
                plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Process image URLs from JSON files: count extensions, analyze invalid URLs, download images, and display samples."
    )

    # Option 1: Specify individual JSON files
    parser.add_argument(
        "--json-files",
        nargs="+",  # Accepts one or more file paths
        type=Path,
        help="Paths to one or more JSON files containing image data.",
    )

    # Option 2: Specify a folder containing JSON files (mutually exclusive with --json-files)
    parser.add_argument(
        "--json-folder",
        type=Path,
        help="Path to a folder containing JSON files. All .json files in this folder will be processed.",
    )

    parser.add_argument(
        "--save-folder",
        required=True,
        type=Path,
        help="Path to the folder where images should be downloaded.",
    )

    parser.add_argument(
        "--show-samples",
        action="store_true",  # Flag, doesn't take a value
        help="Display a grid of randomly downloaded images after processing.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of random images to display (default: 30).",
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of rows in the sample image grid (default: 3).",
    )

    parser.add_argument(
        "--sample-cols",
        type=int,
        default=10,
        help="Number of columns in the sample image grid (default: 10).",
    )

    args = parser.parse_args()

    # --- Input Validation and Setup ---
    if args.json_files and args.json_folder:
        parser.error("Please provide either --json-files or --json-folder, not both.")
    elif not args.json_files and not args.json_folder:
        parser.error("Please provide either --json-files or --json-folder.")

    input_json_files = []
    if args.json_files:
        input_json_files = args.json_files
        # Basic check if provided files exist
        for f in input_json_files:
            if not f.is_file():
                print(f"Error: JSON file not found at {f}")
                exit()

    elif args.json_folder:
        if not args.json_folder.is_dir():
            print(f"Error: JSON folder not found at {args.json_folder}")
            exit()
        input_json_files = list(args.json_folder.glob("*.json"))
        if not input_json_files:
            print(f"Error: No JSON files found in {args.json_folder}")
            exit()

    save_location = args.save_folder

    # --- Processing ---
    processor = ImageProcessor(input_json_files, save_location)

    processor.load_json()

    # Download images, skipping existing ones
    processor.download_images()

    # Optional: Display random images based on the flag
    if args.show_samples:
        print("\nAttempting to display random images...")
        try:
            processor.show_random_images(
                num_images=args.num_samples, rows=args.sample_rows, cols=args.sample_cols
            )
        except ValueError as e:
            print(f"Could not display images: {e}")
        except FileNotFoundError:
            print(f"Error: Save folder {save_location} not found or is empty.")
        except Exception as e:  # Catch other potential display errors
            print(f"An unexpected error occurred during image display: {e}")
