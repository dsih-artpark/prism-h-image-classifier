#!/usr/bin/env python3

from pathlib import Path

import fastdup
import numpy as np


# --- Configuration ---
# Use Path objects for paths
embedding_file = Path("simclr_finetuned/all_embeddings.npz")
image_directory = Path("preprocess_results/clean")
fastdup_work_dir = Path("fastdup_clustering_results")
# --- End Configuration ---


def cluster_main():
    # Use Path.mkdir()
    fastdup_work_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings and file paths
    try:
        # np.load works fine with Path objects
        data = np.load(embedding_file, allow_pickle=True)
        embeddings = data["embeddings"]
        # Ensure file_paths are loaded as a list of strings, handling potential object arrays
        file_paths_raw = data["file_paths"]
        if file_paths_raw.dtype == "O":  # Object array, likely strings already
            file_paths = file_paths_raw.tolist()
        else:  # Assume it's a standard numpy array of strings or needs conversion
            file_paths = [str(fp) for fp in file_paths_raw]
        print(f"Loaded {len(embeddings)} embeddings from {embedding_file}")
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {embedding_file}")
        print("Please ensure 'extract_embeddings.py' has run successfully.")
        return
    except KeyError as e:
        print(
            f"Error: Missing key {e} in {embedding_file}. Ensure it contains 'embeddings' and 'file_paths'."
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred loading {embedding_file}: {e}")
        return

    if embeddings.shape[0] != len(file_paths):
        print(
            f"Warning: Embeddings count ({embeddings.shape[0]}) != file paths count ({len(file_paths)}). Check {embedding_file}. Alignment issues may occur."
        )
        # Decide if you want to proceed or exit
        # return

    # Check if the specified image directory exists using Path.is_dir()
    if not image_directory.is_dir():
        print(
            f"Warning: Image directory '{image_directory}' not found. Fastdup galleries might not show images correctly."
        )
        input_dir_for_fastdup = None
    else:
        # fastdup.create input_dir might expect a string, convert Path to string
        input_dir_for_fastdup = str(image_directory)

    # Initialize fastdup
    # work_dir might also expect a string
    try:
        fd = fastdup.create(work_dir=str(fastdup_work_dir), input_dir=input_dir_for_fastdup)
    except Exception as e:
        print(f"Error initializing fastdup in {fastdup_work_dir}: {e}")
        return

    # Run analysis using pre-computed embeddings
    print("Running fastdup analysis on embeddings...")
    try:
        # Pass file_paths via 'filenames' argument (more standard than annotations for linking)
        # Ensure file_paths is a list of strings
        fd.run(embeddings=embeddings, filenames=file_paths)
        print(f"Fastdup analysis complete. Results stored in {fastdup_work_dir}")
    except TypeError as e:
        print(f"Error during fastdup run: {e}")
        print(
            "This might be due to incorrect data types passed to fastdup (e.g., embeddings or filenames)."
        )
        print(f"Embeddings shape: {embeddings.shape}, type: {embeddings.dtype}")
        print(f"Number of filenames: {len(file_paths)}")
        # Check the type of the first few file paths
        if file_paths:
            print(f"Type of first filename: {type(file_paths[0])}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during fastdup run: {e}")
        return

    # Generate visualization galleries
    print("Generating visualization galleries...")
    try:
        # fastdup gallery functions likely expect string paths for output_dir and filename
        # Although we don't use os.path.join here, we pass the directory as a string.
        fd.vis.component_gallery(output_dir=str(fastdup_work_dir), filename="cluster_gallery.html")
        # gallery_path for printing can still be a Path object
        gallery_path = fastdup_work_dir / "cluster_gallery.html"
        print(f"Cluster gallery saved to {gallery_path}")
    except FileNotFoundError:
        print(f"Error: Could not save cluster gallery. Check permissions for {fastdup_work_dir}")
    except AttributeError:
        print("Error: fastdup object might not have been initialized correctly or analysis failed.")
    except Exception as e:
        print(f"Error generating component gallery: {e}")

    try:
        # Use string for output_dir
        fd.vis.outliers_gallery(output_dir=str(fastdup_work_dir), filename="outlier_gallery.html")
        outlier_gallery_path = fastdup_work_dir / "outlier_gallery.html"
        print(f"Outlier gallery saved to {outlier_gallery_path}")
    except FileNotFoundError:
        print(f"Error: Could not save outlier gallery. Check permissions for {fastdup_work_dir}")
    except AttributeError:
        print("Error: fastdup object might not have been initialized correctly or analysis failed.")
    except Exception as e:
        print(f"Error generating outlier gallery: {e}")

    # Add other visualizations if needed (e.g., duplicates)
    try:
        # Use string for output_dir
        fd.vis.duplicates_gallery(
            output_dir=str(fastdup_work_dir), filename="duplicates_gallery.html"
        )
        duplicates_gallery_path = fastdup_work_dir / "duplicates_gallery.html"
        print(f"Duplicates gallery saved to {duplicates_gallery_path}")
    except FileNotFoundError:
        print(f"Error: Could not save duplicates gallery. Check permissions for {fastdup_work_dir}")
    except AttributeError:
        print("Error: fastdup object might not have been initialized correctly or analysis failed.")
    except ValueError as e:
        # Handle cases where duplicates might not be computed or found
        if "No duplicates found" in str(e):
            print("No duplicates found to generate gallery.")
        else:
            print(f"Error generating duplicates gallery: {e}")
    except Exception as e:
        print(f"Error generating duplicates gallery: {e}")


if __name__ == "__main__":
    cluster_main()
