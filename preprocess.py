#!/usr/bin/env python3

import os
import shutil
import fastdup
from pathlib import Path # Import Path
import argparse # Import argparse

class ImagePreprocessor:
    """
    A class that uses the fastdup library to preprocess images by identifying
    invalid files, duplicates, outliers, dark, and blurry images, and segregating
    them into 'clean' and 'problematic' sets.
    """
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 ccthreshold: float = 0.9,
                 outlier_distance: float = 0.68):
        """
        :param input_dir: Path to the folder that contains all your images.
        :param output_dir: Path where the cleaned and problematic folders will be created.
        :param ccthreshold: Threshold for similarity detection in fastdup (default 0.9).
        :param outlier_distance: Distance threshold for outlier detection (default 0.68).
        """
        # Convert to absolute paths using pathlib
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.ccthreshold = ccthreshold
        self.outlier_distance = outlier_distance

        # Folders for final categorized images using pathlib
        self.clean_folder = self.output_dir / "clean"
        self.problematic_folder = self.output_dir / "problematic"
        self.invalid_folder = self.problematic_folder / "invalid"
        self.duplicates_folder = self.problematic_folder / "duplicates"
        self.outliers_folder = self.problematic_folder / "outliers"
        self.dark_folder = self.problematic_folder / "dark"
        self.blurry_folder = self.problematic_folder / "blurry"

        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create the output directory structure using pathlib."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clean_folder.mkdir(parents=True, exist_ok=True)
        self.problematic_folder.mkdir(parents=True, exist_ok=True)
        self.invalid_folder.mkdir(parents=True, exist_ok=True)
        self.duplicates_folder.mkdir(parents=True, exist_ok=True)
        self.outliers_folder.mkdir(parents=True, exist_ok=True)
        self.dark_folder.mkdir(parents=True, exist_ok=True)
        self.blurry_folder.mkdir(parents=True, exist_ok=True)
    
    def _extract_filename(self, path):
        """Extract just the filename from a path (relative or absolute) using pathlib"""
        # Assuming path is a string from fastdup, convert to Path first
        return Path(path).name
    
    def run_preprocessing(self):
        """
        Run the entire fastdup-based preprocessing pipeline:
          1) Detect invalid images
          2) Compute similarity and connected components (duplicates)
          3) Identify outliers, dark images, and blurry images
          4) Copy images to respective categories
        """
        # 1) Create a FastDup object and run the analysis
        fd = fastdup.create(input_dir=self.input_dir)
        fd.run(ccthreshold=self.ccthreshold)

        # 2) Identify invalid images
        broken_images_df = fd.invalid_instances()
        broken_filenames = [self._extract_filename(path) for path in broken_images_df['filename'].tolist()]
        print(f"Found {len(broken_filenames)} invalid images.")

        # 3) Find duplicates via connected components
        connected_components_df, _ = fd.connected_components()
        clusters_df = self._get_clusters(
            connected_components_df, 
            sort_by='count', 
            min_count=2, 
            ascending=False
        )

        keep_filenames = []
        duplicate_filenames = []
        
        for cluster_file_list in clusters_df.filename:
            if not cluster_file_list:  # Skip empty lists
                continue
                
            # We'll keep the first one and mark the rest as duplicates
            keep = self._extract_filename(cluster_file_list[0])
            discard = [self._extract_filename(path) for path in cluster_file_list[1:]]
            
            keep_filenames.append(keep)
            duplicate_filenames.extend(discard)
        
        print(f"Found {len(set(duplicate_filenames))} duplicates.")

        # 4) Find outliers (distance < outlier_distance)
        outlier_df = fd.outliers()
        outlier_filenames = [
            self._extract_filename(path) 
            for path in outlier_df[outlier_df.distance < self.outlier_distance].filename_outlier.tolist()
        ]
        print(f"Found {len(outlier_filenames)} outliers with distance < {self.outlier_distance}.")

        # 5) Dark and blurry images from stats
        stats_df = fd.img_stats()
        
        dark_images = stats_df[stats_df['mean'] < 13]    # threshold for darkness
        dark_filenames = [self._extract_filename(path) for path in dark_images['filename'].tolist()]
        print(f"Found {len(dark_filenames)} dark images (mean < 13).")

        blurry_images = stats_df[stats_df['blur'] < 50]  # threshold for blur
        blurry_filenames = [self._extract_filename(path) for path in blurry_images['filename'].tolist()]
        print(f"Found {len(blurry_filenames)} blurry images (blur < 50).")

        # 6) Collect all problematic filenames
        broken_set = set(broken_filenames)
        duplicates_set = set(duplicate_filenames)
        outlier_set = set(outlier_filenames)
        dark_set = set(dark_filenames)
        blurry_set = set(blurry_filenames)
        keep_set = set(keep_filenames)

        # 7) Build sets for processing
        all_problematic = broken_set.union(duplicates_set, outlier_set, dark_set, blurry_set)
        print(f"Total problematic images: {len(all_problematic)}")
        print(f"Images to keep from clusters: {len(keep_set)}")

        # 8) Process all files in the input directory
        problematic_count = {
            "invalid": 0,
            "duplicates": 0,
            "outliers": 0,
            "dark": 0,
            "blurry": 0
        }
        
        clean_count = 0
        kept_duplicates = 0
        
        # Get a list of all files using pathlib (assuming flat structure or recursion handled by fastdup already, adjust if needed)
        # Using rglob to find all files recursively. Filter for actual files.
        all_paths = [p for p in self.input_dir.rglob('*') if p.is_file()]
        all_files = [(p, p.name) for p in all_paths]
        
        print(f"Found {len(all_files)} total files in input directory: {self.input_dir}")
        
        # Process each file
        for full_path, filename in all_files:
            # Copy to the problematic folders if needed
            if filename in broken_set:
                self._copy_to_folder(full_path, self.invalid_folder)
                problematic_count["invalid"] += 1
            
            if filename in duplicates_set:
                self._copy_to_folder(full_path, self.duplicates_folder)
                problematic_count["duplicates"] += 1
            
            if filename in outlier_set:
                self._copy_to_folder(full_path, self.outliers_folder)
                problematic_count["outliers"] += 1
            
            if filename in dark_set:
                self._copy_to_folder(full_path, self.dark_folder)
                problematic_count["dark"] += 1
            
            if filename in blurry_set:
                self._copy_to_folder(full_path, self.blurry_folder)
                problematic_count["blurry"] += 1
            
            # Copy to clean folder if not problematic or if it's a keeper
            if filename not in all_problematic or filename in keep_set:
                self._copy_to_folder(full_path, self.clean_folder)
                clean_count += 1
                if filename in keep_set:
                    kept_duplicates += 1

        # Print summary
        print("Copying results:")
        print(f"- Invalid: {problematic_count['invalid']}/{len(broken_set)}")
        print(f"- Duplicates: {problematic_count['duplicates']}/{len(duplicates_set)}")
        print(f"- Outliers: {problematic_count['outliers']}/{len(outlier_set)}")
        print(f"- Dark: {problematic_count['dark']}/{len(dark_set)}")
        print(f"- Blurry: {problematic_count['blurry']}/{len(blurry_set)}")
        print(f"- Clean: {clean_count} (including {kept_duplicates} kept duplicates)")

    def _copy_to_folder(self, src_path, dest_folder):
        """Copy a file to the destination folder using pathlib"""
        # Ensure src_path is a Path object if it comes from all_files list
        src_path_obj = Path(src_path) 
        filename = src_path_obj.name 
        # Ensure dest_folder is a Path object
        dest_folder_obj = Path(dest_folder)
        dest_path = os.path.join(dest_folder, filename)
        dest_path_obj = dest_folder_obj / filename # Use pathlib join
        try: 
            shutil.copy2(src_path, dest_path)
            return True
        except Exception as e:
            print(f"Error copying {src_path} to {dest_folder}: {e}")
            return False

    def _get_clusters(self, df, sort_by='count', min_count=2, ascending=False):
        """
        Given a connected_components DataFrame from fastdup, group into clusters
        with the specified sorting options.
        """
        agg_dict = {'filename': list, 'mean_distance': 'max', 'count': 'count'}
        if 'label' in df.columns:
            agg_dict['label'] = list

        # only consider rows where 'count' >= min_count
        df = df[df['count'] >= min_count]

        grouped_df = df.groupby('component_id').agg(agg_dict)
        grouped_df = grouped_df.sort_values(by=[sort_by], ascending=ascending)
        return grouped_df

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Preprocess images using fastdup to find and separate invalid, duplicate, outlier, dark, and blurry images.")
    
    parser.add_argument(
        '--input-dir', 
        required=True, 
        type=Path, 
        help='Path to the folder containing the input images.'
    )
    parser.add_argument(
        '--output-dir', 
        required=True, 
        type=Path, 
        help='Path to the folder where \'clean\' and \'problematic\' subfolders will be created.'
    )
    parser.add_argument(
        '--ccthreshold', 
        type=float, 
        default=0.9, 
        help='Threshold for similarity detection (connected components) in fastdup (default: 0.9).'
    )
    parser.add_argument(
        '--outlier-distance', 
        type=float, 
        default=0.95, 
        help='Distance threshold for outlier detection (default: 0.68).'
    )

    args = parser.parse_args()

    # Basic input validation
    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found or is not a directory: {args.input_dir}")
        exit()

    # --- Processing ---
    preprocessor = ImagePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ccthreshold=args.ccthreshold,
        outlier_distance=args.outlier_distance
    )
    preprocessor.run_preprocessing()
    print("Preprocessing done.")
