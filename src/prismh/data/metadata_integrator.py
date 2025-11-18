#!/usr/bin/env python3

"""
Metadata Integrator

A module for integrating JSON metadata with processed images and providing analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from visualization_utils import VisualizationUtils


class MetadataIntegrator:
    """
    A class for integrating JSON metadata with processed images.
    Provides functions for analysis, statistics, and generating reports.
    """

    def __init__(
        self,
        json_file_path: str,
        processed_dir: str,
        output_dir: str = "metadata_analysis",
        id_field: str = "Id",
        image_url_field: str = "image_url",
        env: str = "dev",
    ):
        """
        Initialize the metadata integrator.

        Args:
            json_file_path (str): Path to the JSON metadata file
            processed_dir (str): Path to the directory with processed images
            output_dir (str): Path to save analysis output
            id_field (str): Field name in JSON that contains the image ID
            image_url_field (str): Field name in JSON that contains the image URL
            env (str): Environment (dev/main)
        """
        self.json_file_path = Path(json_file_path).resolve()
        self.processed_dir = Path(processed_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.id_field = id_field
        self.image_url_field = image_url_field
        self.env = env

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.json_data = self._load_json_data()

        self.image_data = self._load_image_data()

        self.integrated_data = None

        viz_output_dir = self.output_dir / "visualizations"
        self.viz = VisualizationUtils(output_dir=str(viz_output_dir))

    def _load_json_data(self) -> list[dict]:
        """
        Load JSON metadata from file.

        Returns:
            List[Dict]: List of metadata records
        """
        try:
            with self.json_file_path.open("r") as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, dict):
                if "data" in data:
                    return data["data"]
                else:
                    return [data]
            return data
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return []

    def _load_image_data(self) -> dict[str, dict]:
        """
        Load information about processed images in the directories.

        Returns:
            Dict[str, Dict]: Dictionary mapping image IDs to their metadata
        """
        image_data = {}

        # Get all image files from the processed directory
        clean_dir = self.processed_dir / "clean"
        problematic_dir = self.processed_dir / "problematic"

        # Process clean images
        if clean_dir.exists():
            for f_path in clean_dir.iterdir():
                if f_path.is_file() and f_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    image_id = f_path.stem  # Use stem to get filename without extension
                    image_data[image_id] = {
                        "status": "clean",
                        "path": str(f_path),  # Store as string for compatibility if needed
                        "filename": f_path.name,
                    }

        # Process problematic images
        if problematic_dir.exists():
            for category_path in problematic_dir.iterdir():
                if category_path.is_dir():
                    category_name = category_path.name
                    for f_path in category_path.iterdir():
                        if f_path.is_file() and f_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                            image_id = f_path.stem
                            image_data[image_id] = {
                                "status": "problematic",
                                "problem_category": category_name,
                                "path": str(f_path),
                                "filename": f_path.name,
                            }

        return image_data

    def integrate_metadata(self) -> pd.DataFrame:
        """
        Integrate JSON metadata with processed image information.

        Returns:
            pd.DataFrame: DataFrame with integrated metadata
        """
        integrated_records = []

        # Track statistics for reporting
        self.match_stats = {
            "total_metadata_records": len(self.json_data),
            "total_processed_images": len(self.image_data),
            "matched_records": 0,
            "unmatched_metadata": [],
            "unmatched_images": [],
        }

        # Integrate metadata with processed images
        for record in self.json_data:
            # Extract ID
            if self.id_field in record:
                record_id = str(record[self.id_field])

                # Check if we have a processed image for this ID
                if record_id in self.image_data:
                    # Combine metadata with image info
                    integrated_record = {**record, **self.image_data[record_id]}
                    integrated_records.append(integrated_record)
                    self.match_stats["matched_records"] += 1
                else:
                    # Keep track of metadata without matching images
                    self.match_stats["unmatched_metadata"].append(record_id)
            else:
                print(f"Warning: Record missing ID field: {record}")

        # Find processed images without metadata
        metadata_ids = set(
            str(record[self.id_field]) for record in self.json_data if self.id_field in record
        )
        self.match_stats["unmatched_images"] = [
            image_id for image_id in self.image_data.keys() if image_id not in metadata_ids
        ]

        # Convert to DataFrame
        if integrated_records:
            self.integrated_data = pd.DataFrame(integrated_records)
            return self.integrated_data
        else:
            # Return empty DataFrame with expected columns
            self.integrated_data = pd.DataFrame()
            return self.integrated_data

    def analyze_data(self) -> dict[str, Any]:
        """
        Analyze the integrated data and return statistics.

        Returns:
            Dict[str, Any]: Dictionary with analysis results
        """
        # Make sure we have integrated data
        if self.integrated_data is None or self.integrated_data.empty:
            self.integrate_metadata()

        if self.integrated_data.empty:
            return {"error": "No integrated data available for analysis"}

        analysis = {
            "total_records": len(self.integrated_data),
            "match_rate": (
                self.match_stats["matched_records"]
                / self.match_stats["total_metadata_records"]
                * 100
                if self.match_stats["total_metadata_records"] > 0
                else 0
            ),  # Avoid division by zero
            "quality_distribution": {},
            "category_distribution": {},
            "date_distribution": {},
            "location_distribution": {},
        }

        # Analyze image quality distribution
        if "status" in self.integrated_data.columns:
            status_counts = self.integrated_data["status"].value_counts().to_dict()
            analysis["quality_distribution"] = status_counts

            # Get problematic category distribution if available
            if "problem_category" in self.integrated_data.columns:
                problem_categories = (
                    self.integrated_data[self.integrated_data["status"] == "problematic"][
                        "problem_category"
                    ]
                    .value_counts()
                    .to_dict()
                )
                analysis["problem_category_distribution"] = problem_categories

        # Analyze category distribution if available
        category_fields = [
            col
            for col in self.integrated_data.columns
            if col.lower() in ("category", "type", "container type", "breeding spot")
        ]

        if category_fields:
            primary_category_field = category_fields[0]
            category_counts = self.integrated_data[primary_category_field].value_counts().to_dict()
            analysis["category_distribution"] = category_counts

        # Analyze date distribution if available
        date_fields = [
            col
            for col in self.integrated_data.columns
            if any(
                date_term in col.lower() for date_term in ("date", "time", "day", "month", "year")
            )
        ]

        if date_fields:
            # Try to convert to datetime
            try:
                date_field = date_fields[0]
                self.integrated_data[f"{date_field}_dt"] = pd.to_datetime(
                    self.integrated_data[date_field], errors="coerce"
                )

                # Group by month
                date_counts = (
                    self.integrated_data[f"{date_field}_dt"]
                    .dt.to_period("M")
                    .value_counts()
                    .sort_index()
                )
                analysis["date_distribution"] = {
                    str(period): count for period, count in date_counts.items()
                }
            except Exception as e:
                print(f"Error processing dates: {e}")

        # Analyze location distribution if available
        location_fields = [
            col
            for col in self.integrated_data.columns
            if any(
                loc_term in col.lower()
                for loc_term in ("location", "ward", "area", "region", "district")
            )
        ]

        if location_fields:
            location_field = location_fields[0]
            location_counts = self.integrated_data[location_field].value_counts().to_dict()
            # Limit to top 10 for readability
            analysis["location_distribution"] = {
                k: v
                for k, v in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }

        return analysis

    def generate_visualizations(self, analysis: dict[str, Any] | None = None) -> list[str]:
        """
        Generate visualizations from the integrated data.

        Args:
            analysis (Dict[str, Any], optional): Analysis results to visualize

        Returns:
            List[str]: List of paths to generated visualizations
        """
        # Ensure data is analyzed
        if analysis is None:
            analysis = self.analyze_data()
            if "error" in analysis:
                print(f"Cannot generate visualizations: {analysis['error']}")
                return []

        # Use the visualization utility's output dir
        viz_output_dir = Path(self.viz.output_dir)  # Ensure it's a Path object
        viz_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists

        generated_files = []

        # Quality distribution pie chart
        if "quality_distribution" in analysis and analysis["quality_distribution"]:
            quality_data = analysis["quality_distribution"]

            self.viz.plot_pie(
                data=quality_data,
                title="Image Quality Distribution",
                filename=str(viz_output_dir / "quality_distribution_pie.png"),
            )
            generated_files.append(str(viz_output_dir / "quality_distribution_pie.png"))

        # Problematic category distribution bar chart (if available)
        if (
            "problem_category_distribution" in analysis
            and analysis["problem_category_distribution"]
        ):
            self.viz.plot_bar(
                data=analysis["problem_category_distribution"],
                title="Distribution of Problematic Image Categories",
                xlabel="Category",
                ylabel="Count",
                filename=str(viz_output_dir / "problematic_categories_bar.png"),
            )
            generated_files.append(str(viz_output_dir / "problematic_categories_bar.png"))

        # Category distribution bar chart (if available)
        if "category_distribution" in analysis and analysis["category_distribution"]:
            category_data = analysis["category_distribution"]

            self.viz.plot_bar(
                data=category_data,
                title="Data Category Distribution",
                xlabel="Category",
                ylabel="Count",
                filename=str(viz_output_dir / "category_distribution_bar.png"),
            )
            generated_files.append(str(viz_output_dir / "category_distribution_bar.png"))

        # Date distribution line chart (if available)
        if "date_distribution" in analysis and analysis["date_distribution"] and date_fields:
            self.viz.plot_line(
                data=analysis["date_distribution"],
                title="Image Distribution Over Time (Monthly)",
                xlabel="Month",
                ylabel="Count",
                filename=str(viz_output_dir / "date_distribution_line.png"),
            )
            generated_files.append(str(viz_output_dir / "date_distribution_line.png"))

        # Location distribution bar chart (if available)
        if "location_distribution" in analysis and analysis["location_distribution"]:
            self.viz.plot_bar(
                data=analysis["location_distribution"],
                title="Location Distribution",
                xlabel="Location",
                ylabel="Count",
                filename=str(viz_output_dir / "location_distribution_bar.png"),
                top_n=20,  # Show top 20 locations
            )
            generated_files.append(str(viz_output_dir / "location_distribution_bar.png"))

        print(f"Visualizations saved to: {viz_output_dir}")
        return generated_files

    def _find_column(self, possible_names: list[str]) -> str | None:
        """
        Find a column in the data that matches any of the possible names.

        Args:
            possible_names (List[str]): List of possible column name substrings

        Returns:
            Optional[str]: Found column name or None
        """
        if self.integrated_data is None or self.integrated_data.empty:
            return None

        for col in self.integrated_data.columns:
            if any(name.lower() in col.lower() for name in possible_names):
                return col

        return None

    def generate_report(self) -> str:
        """
        Generate a comprehensive report with all visualizations and analysis.

        Returns:
            str: Path to the generated report HTML file
        """
        # Make sure we have integrated data
        if self.integrated_data is None or self.integrated_data.empty:
            self.integrate_metadata()

        if self.integrated_data.empty:
            print("No integrated data available for report generation")
            return ""

        # Run analysis
        analysis = self.analyze_data()

        # Generate visualizations
        visualizations = self.generate_visualizations(analysis)

        # Prepare summary data for the report
        summary_data = {
            "Total Metadata Records": self.match_stats["total_metadata_records"],
            "Total Processed Images": self.match_stats["total_processed_images"],
            "Matched Records": self.match_stats["matched_records"],
            "Match Rate": self.match_stats["matched_records"]
            / self.match_stats["total_metadata_records"]
            * 100,
            "Unmatched Metadata Records": len(self.match_stats["unmatched_metadata"]),
            "Unmatched Images": len(self.match_stats["unmatched_images"]),
        }

        # Add quality stats if available
        if "quality_distribution" in analysis:
            for quality, count in analysis["quality_distribution"].items():
                summary_data[f"{quality} Images"] = count

        # Add problem category stats if available
        if "problem_category_distribution" in analysis:
            for category, count in analysis["problem_category_distribution"].items():
                summary_data[f"{category} Issues"] = count

        # Add category stats if available
        if "category_distribution" in analysis:
            category_col = self._find_column(
                ["category", "type", "container type", "breeding spot"]
            )
            if category_col:
                summary_data[f"{category_col.capitalize()} Distribution"] = ", ".join(
                    f"{k}: {v}"
                    for k, v in sorted(
                        analysis["category_distribution"].items(), key=lambda x: x[1], reverse=True
                    )
                )

        # Check for breeding spot distribution
        breeding_col = self._find_column(["breed", "spot", "larv", "pupa"])
        if breeding_col and breeding_col in self.integrated_data.columns:
            # Calculate breeding spot percentage
            breeding_counts = self.integrated_data[breeding_col].value_counts()
            if "Yes" in breeding_counts:
                yes_count = breeding_counts["Yes"]
                total_count = len(self.integrated_data)
                yes_percent = yes_count / total_count * 100
                summary_data["Breeding Spots Detected"] = yes_count
                summary_data["Breeding Spot Percentage"] = yes_percent

        # Check quality issues
        quality_cols = [
            col
            for col in self.integrated_data.columns
            if col.startswith("is_")
            and col
            in ["is_dark", "is_blurry", "is_bright", "is_duplicate", "is_outlier", "is_redundant"]
        ]
        if quality_cols:
            for col in quality_cols:
                if col in self.integrated_data.columns:
                    issue_count = self.integrated_data[col].sum()
                    issue_pct = issue_count / len(self.integrated_data) * 100
                    issue_name = col.replace("is_", "").capitalize()
                    summary_data[f"{issue_name} Images"] = issue_count
                    summary_data[f"{issue_name} Image Percentage"] = issue_pct

            # Count images with any quality issue
            has_issue = self.integrated_data[quality_cols].any(axis=1)
            issue_count = has_issue.sum()
            issue_pct = issue_count / len(self.integrated_data) * 100
            summary_data["Images with Any Quality Issue"] = issue_count
            summary_data["Quality Issue Percentage"] = issue_pct

        # Add some metadata analysis
        worker_col = self._find_column(["worker", "user", "asha", "uid", "Uid"])
        if worker_col and worker_col in self.integrated_data.columns:
            worker_counts = self.integrated_data[worker_col].value_counts()
            summary_data["Total Workers/Users"] = len(worker_counts)
            summary_data["Most Active Worker"] = (
                f"{worker_counts.index[0]} ({worker_counts.iloc[0]} images)"
            )
            summary_data["Average Images per Worker"] = len(self.integrated_data) / len(
                worker_counts
            )

        ward_col = self._find_column(["ward", "area", "zone", "district"])
        if ward_col and ward_col in self.integrated_data.columns:
            ward_counts = self.integrated_data[ward_col].value_counts()
            summary_data["Total Wards/Areas"] = len(ward_counts)
            summary_data["Most Common Ward"] = (
                f"{ward_counts.index[0]} ({ward_counts.iloc[0]} images)"
            )

        date_col = self._find_column(["date", "time", "timestamp"])
        if date_col and date_col in self.integrated_data.columns:
            # Convert to datetime if not already
            try:
                if self.integrated_data[date_col].dtype != "datetime64[ns]":
                    self.integrated_data[date_col] = pd.to_datetime(
                        self.integrated_data[date_col], errors="coerce"
                    )

                # Filter out invalid dates
                valid_dates = self.integrated_data.dropna(subset=[date_col])

                if not valid_dates.empty:
                    min_date = valid_dates[date_col].min()
                    max_date = valid_dates[date_col].max()
                    date_range = max_date - min_date

                    summary_data["Date Range"] = (
                        f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                    )
                    summary_data["Collection Period"] = f"{date_range.days} days"
                    summary_data["Average Images per Day"] = len(valid_dates) / max(
                        date_range.days, 1
                    )
            except Exception as e:
                print(f"Error analyzing dates: {e}")

        additional_text = self._generate_report_insights()

        # Generate HTML report
        report_file = self.viz.generate_html_report(
            title="Mosquito Breeding Site Data Analysis Report",
            filename="report.html",
            visualizations=visualizations,
            summary_data=summary_data,
            additional_text=additional_text,
        )

        print(f"Report generated at: {report_file}")
        return report_file

    def _generate_report_insights(self) -> str:
        """
        Generate insights text for the report based on the analysis.

        Returns:
            str: HTML-formatted insights text
        """
        insights = []

        # Check match rate
        match_rate = (
            self.match_stats["matched_records"] / self.match_stats["total_metadata_records"] * 100
        )
        if match_rate < 80:
            insights.append(
                f"<p><strong>Low Match Rate:</strong> Only {match_rate:.1f}% of metadata records were matched with processed images. Consider reviewing the data collection process to ensure images are properly linked to metadata.</p>"
            )

        # Check quality issues
        quality_cols = [
            col
            for col in self.integrated_data.columns
            if col.startswith("is_")
            and col
            in ["is_dark", "is_blurry", "is_bright", "is_duplicate", "is_outlier", "is_redundant"]
        ]

        if quality_cols:
            # Check for high quality issue rates
            for col in quality_cols:
                if col in self.integrated_data.columns:
                    issue_pct = self.integrated_data[col].mean() * 100
                    issue_name = col.replace("is_", "").capitalize()

                    if issue_pct > 15:
                        insights.append(
                            f"<p><strong>High {issue_name} Rate:</strong> {issue_pct:.1f}% of images are detected as {issue_name.lower()}. This may indicate issues with camera settings or collection conditions.</p>"
                        )

            # Check overall quality
            has_issue = self.integrated_data[quality_cols].any(axis=1)
            issue_pct = has_issue.sum() / len(self.integrated_data) * 100

            if issue_pct > 40:
                insights.append(
                    f"<p><strong>Poor Overall Quality:</strong> {issue_pct:.1f}% of images have at least one quality issue. Training on proper image capture techniques may be beneficial.</p>"
                )

        # Check breeding spot distribution
        breeding_col = self._find_column(["breed", "spot", "larv", "pupa"])
        if breeding_col and breeding_col in self.integrated_data.columns:
            # Calculate breeding spot percentage
            breeding_counts = self.integrated_data[breeding_col].value_counts()
            if "Yes" in breeding_counts:
                yes_count = breeding_counts["Yes"]
                total_count = len(self.integrated_data)
                yes_percent = yes_count / total_count * 100

                if yes_percent > 80:
                    insights.append(
                        f"<p><strong>Very High Breeding Spot Rate:</strong> {yes_percent:.1f}% of images are labeled as breeding spots. This seems unusually high and may indicate labeling bias or selection bias in collection.</p>"
                    )
                elif yes_percent < 5:
                    insights.append(
                        f"<p><strong>Very Low Breeding Spot Rate:</strong> Only {yes_percent:.1f}% of images are labeled as breeding spots. This may indicate missed detections or focused collection in low-risk areas.</p>"
                    )

        # Check worker/ward distribution
        worker_col = self._find_column(["worker", "user", "asha", "uid", "Uid"])
        if worker_col and worker_col in self.integrated_data.columns:
            worker_counts = self.integrated_data[worker_col].value_counts()

            # Check for worker concentration
            top_worker_pct = worker_counts.iloc[0] / len(self.integrated_data) * 100

            if top_worker_pct > 40 and len(worker_counts) > 3:
                insights.append(
                    f"<p><strong>Worker Concentration:</strong> The most active worker ({worker_counts.index[0]}) contributed {top_worker_pct:.1f}% of all images. Consider ensuring more balanced data collection across workers.</p>"
                )

            # Check for duplicate concentration
            duplicate_col = self._find_column(
                ["duplicate", "redundant", "is_duplicate", "is_redundant"]
            )
            if duplicate_col and duplicate_col in self.integrated_data.columns:
                # Group by worker and calculate duplicate rate
                try:
                    worker_dupes = (
                        self.integrated_data.groupby(worker_col)[duplicate_col].mean() * 100
                    )
                    high_dupe_workers = worker_dupes[worker_dupes > 25]

                    if not high_dupe_workers.empty:
                        top_dupe_worker = high_dupe_workers.index[0]
                        top_dupe_pct = high_dupe_workers.iloc[0]

                        insights.append(
                            f"<p><strong>High Duplicate Rate from Workers:</strong> Worker {top_dupe_worker} has a {top_dupe_pct:.1f}% duplicate rate. Consider additional training on proper image collection protocols.</p>"
                        )
                except Exception as e:
                    print(f"Error analyzing worker duplicates: {e}")

        # Check time patterns
        date_col = self._find_column(["date", "time", "timestamp"])
        if date_col and date_col in self.integrated_data.columns:
            try:
                if self.integrated_data[date_col].dtype != "datetime64[ns]":
                    self.integrated_data[date_col] = pd.to_datetime(
                        self.integrated_data[date_col], errors="coerce"
                    )

                # Get hour of day distribution
                hour_counts = self.integrated_data[date_col].dt.hour.value_counts().sort_index()

                # Check for off-hours collection
                night_hours = (
                    hour_counts.loc[hour_counts.index.isin(range(19, 24))].sum()
                    + hour_counts.loc[hour_counts.index.isin(range(0, 6))].sum()
                )
                night_pct = night_hours / hour_counts.sum() * 100

                if night_pct > 15:
                    insights.append(
                        f"<p><strong>Unusual Collection Hours:</strong> {night_pct:.1f}% of images were collected during night hours (7PM-6AM). This may affect image quality and breeding spot visibility.</p>"
                    )
            except Exception as e:
                print(f"Error analyzing time patterns: {e}")

        # Container type analysis
        container_col = self._find_column(["container", "type"])
        if container_col and container_col in self.integrated_data.columns and breeding_col:
            try:
                # Get container type counts
                container_counts = self.integrated_data[container_col].value_counts()
                top_container = container_counts.index[0]
                top_container_pct = container_counts.iloc[0] / len(self.integrated_data) * 100

                if top_container_pct > 50:
                    insights.append(
                        f"<p><strong>Container Type Focus:</strong> {top_container_pct:.1f}% of images are of '{top_container}' containers. This may indicate over-focus on one container type or common breeding environments in the area.</p>"
                    )

                # Check breeding rate by container
                container_breeding = (
                    self.integrated_data.groupby(container_col)[breeding_col]
                    .apply(lambda x: (x == "Yes").mean() * 100)
                    .sort_values(ascending=False)
                )

                if not container_breeding.empty:
                    highest_breeding = container_breeding.index[0]
                    highest_breeding_pct = container_breeding.iloc[0]

                    if highest_breeding_pct > 70:
                        insights.append(
                            f"<p><strong>High-Risk Container Type:</strong> '{highest_breeding}' containers show a {highest_breeding_pct:.1f}% breeding spot rate, significantly higher than other types. These containers may be priority targets for intervention.</p>"
                        )
            except Exception as e:
                print(f"Error analyzing container types: {e}")

        # If no insights were generated
        if not insights:
            insights.append(
                "<p>No significant insights detected in the current dataset. The data appears to follow expected patterns.</p>"
            )

        # Combine insights
        return "<h3>Key Insights</h3>" + "".join(insights)

    def create_annotated_images(
        self,
        output_dir: str | None = None,
        fields_to_show: list[str] | None = None,
        sample_size: int = 0,
    ) -> list[str]:
        """
        Create annotated versions of images with metadata overlay.

        Args:
            output_dir (str, optional): Directory to save annotated images
            fields_to_show (List[str], optional): Fields to show in annotation
            sample_size (int): If > 0, only process a sample of this size

        Returns:
            List[str]: List of paths to annotated images
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "annotated_images")

        os.makedirs(output_dir, exist_ok=True)

        # Make sure we have integrated data
        if self.integrated_data is None or self.integrated_data.empty:
            self.integrate_metadata()

        if self.integrated_data.empty:
            print("No integrated data available for annotation")
            return []

        # If no fields specified, use common fields
        if fields_to_show is None:
            # Look for common fields in the data
            potential_fields = [
                self.id_field,
                "Ward Name",
                "Ward Number",
                "Date and Time",
                "Breeding spot",
                "Container Type",
                "Latitude and Longitude",
                "Remarks",
                "Asha worker",
            ]
            fields_to_show = [
                field for field in potential_fields if field in self.integrated_data.columns
            ]

        # Get data to process
        if sample_size > 0 and sample_size < len(self.integrated_data):
            data_to_process = self.integrated_data.sample(sample_size)
        else:
            data_to_process = self.integrated_data

        annotated_paths = []

        # Process each image
        for _, row in data_to_process.iterrows():
            if "path" in row and os.path.exists(row["path"]):
                try:
                    # Open image
                    img = Image.open(row["path"])
                    draw = ImageDraw.Draw(img)

                    # Try to get a font
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except:
                        # Fallback to default font
                        font = ImageFont.load_default()

                    annotation = []
                    for field in fields_to_show:
                        if field in row and not pd.isna(row[field]):
                            annotation.append(f"{field}: {row[field]}")

                    # Add status information
                    if "status" in row:
                        status_text = f"Status: {row['status']}"
                        if row["status"] == "problematic" and "problem_category" in row:
                            status_text += f" ({row['problem_category']})"
                        annotation.append(status_text)

                    annotation_text = "\n".join(annotation)

                    text_width, text_height = draw.textsize(annotation_text, font=font)
                    padding = 10

                    # Ensure text bg stays within image bounds
                    bg_width = min(text_width + 2 * padding, img.width)
                    bg_height = min(text_height + 2 * padding, img.height)

                    # Add bg rectangle
                    draw.rectangle(
                        [(0, 0), (bg_width, bg_height)],
                        fill=(0, 0, 0, 128),  # Semi-transparent black
                    )

                    # Add text
                    draw.text((padding, padding), annotation_text, fill=(255, 255, 255), font=font)

                    # Save annotated image
                    output_path = os.path.join(
                        output_dir, f"annotated_{os.path.basename(row['path'])}"
                    )
                    img.save(output_path)
                    annotated_paths.append(output_path)

                except Exception as e:
                    print(f"Error annotating image {row['path']}: {e}")

        print(f"Created {len(annotated_paths)} annotated images in {output_dir}")
        return annotated_paths

    def export_integrated_data(self, output_format: str = "csv") -> str:
        """
        Export the integrated data to a file.

        Args:
            output_format (str): Format to export (csv, excel, json)

        Returns:
            str: Path to the exported file
        """
        # Make sure we have integrated data
        if self.integrated_data is None or self.integrated_data.empty:
            self.integrate_metadata()

        if self.integrated_data.empty:
            print("No integrated data available for export")
            return ""

        # Define output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_metadata_{timestamp}.{output_format}"
        output_path = self.output_dir / filename

        try:
            if output_format == "csv":
                self.integrated_data.to_csv(output_path, index=False)
            elif output_format == "json":
                self.integrated_data.to_json(output_path, orient="records", indent=4)
            elif output_format == "xlsx":
                self.integrated_data.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {output_format}")

            print(f"Integrated data exported successfully to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error exporting data: {e}")
            return ""


def main():
    """
    Main function to run the metadata integrator from command line.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Metadata Integration Tool for Mosquito Breeding Site Analysis"
    )

    # Required arguments
    parser.add_argument("--json", required=True, help="Path to JSON metadata file")
    parser.add_argument("--images", required=True, help="Path to directory with processed images")

    # Optional arguments
    parser.add_argument(
        "--output", default="metadata_analysis", help="Output directory for analysis"
    )
    parser.add_argument(
        "--id-field", default="Id", help="Field name in JSON that contains image ID"
    )
    parser.add_argument(
        "--image-url-field", default="image_url", help="Field name in JSON that contains image URL"
    )
    parser.add_argument(
        "--env", default="dev", choices=["dev", "main"], help="Environment (dev/main)"
    )
    parser.add_argument(
        "--annotate", action="store_true", help="Create annotated images with metadata overlay"
    )
    parser.add_argument(
        "--export",
        choices=["csv", "excel", "json"],
        help="Export integrated data in specified format",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations without full report",
    )

    # Specific visualization options
    parser.add_argument(
        "--worker-viz", action="store_true", help="Generate worker-specific visualizations"
    )
    parser.add_argument(
        "--worker-quality",
        action="store_true",
        help="Generate worker quality analysis visualizations",
    )
    parser.add_argument(
        "--timeline-viz", action="store_true", help="Generate timeline visualizations"
    )
    parser.add_argument(
        "--quality-viz", action="store_true", help="Generate quality distribution visualizations"
    )
    parser.add_argument(
        "--container-viz", action="store_true", help="Generate container type visualizations"
    )
    parser.add_argument(
        "--ward-viz", action="store_true", help="Generate ward-level visualizations"
    )

    args = parser.parse_args()

    print("Mosquito Breeding Site Metadata Integrator")

    # Print configuration
    print("\nConfiguration:")
    print(f"  JSON Metadata: {args.json}")
    print(f"  Processed Images: {args.images}")
    print(f"  Output Directory: {args.output}")
    print(f"  ID Field: {args.id_field}")
    print(f"  Environment: {args.env}")

    if args.annotate:
        print("  Creating annotated images")

    if args.export:
        print(f"  Exporting data as: {args.export}")

    print("")

    try:
        integrator = MetadataIntegrator(
            json_file_path=args.json,
            processed_dir=args.images,
            output_dir=args.output,
            id_field=args.id_field,
            image_url_field=args.image_url_field,
            env=args.env,
        )

        # Integrate metadata with processed images
        print("\nIntegrating metadata with processed images...")
        integrated_data = integrator.integrate_metadata()

        if integrated_data is None or len(integrated_data) == 0:
            print("No data was integrated. Check input files and paths.")
            return 1

        # Get metadata and processed counts from match_stats
        metadata_count = integrator.match_stats["total_metadata_records"]
        processed_count = integrator.match_stats["total_processed_images"]

        # Calculate match rate
        match_rate = (len(integrated_data) / metadata_count) * 100 if metadata_count > 0 else 0

        print("Integration summary:")
        print(f"  Metadata records: {metadata_count}")
        print(f"  Processed images: {processed_count}")
        print(f"  Integrated records: {len(integrated_data)}")
        print(f"  Match rate: {match_rate:.1f}%")

        # Analyze data
        print("\nAnalyzing integrated data...")
        analysis = integrator.analyze_data()

        # Generate specific visualizations if requested
        if any(
            [
                args.worker_viz,
                args.worker_quality,
                args.timeline_viz,
                args.quality_viz,
                args.container_viz,
                args.ward_viz,
            ]
        ):
            print("\nGenerating requested visualizations...")
            viz_list = []

            # Find common columns
            breeding_col = integrator._find_column(["breed", "spot", "larv", "pupa"])
            worker_col = integrator._find_column(["worker", "user", "asha", "uid", "Uid"])
            ward_col = integrator._find_column(["ward", "area", "zone", "district"])
            date_col = integrator._find_column(["date", "time", "timestamp"])
            container_col = integrator._find_column(["container", "type"])
            quality_cols = [
                col
                for col in integrated_data.columns
                if col.startswith("is_")
                and col
                in [
                    "is_dark",
                    "is_blurry",
                    "is_bright",
                    "is_duplicate",
                    "is_outlier",
                    "is_redundant",
                ]
            ]

            # Worker visualizations
            if args.worker_viz and worker_col:
                print("  Generating worker visualizations...")
                if breeding_col:
                    integrator.viz.plot_worker_breeding_spots(
                        data=integrated_data,
                        worker_column=worker_col,
                        spot_column=breeding_col,
                        title=f"Breeding Spot Counts per {worker_col.capitalize()}",
                        filename="worker_breeding_spots.png",
                    )
                    viz_list.append("worker_breeding_spots.png")

                duplicate_col = integrator._find_column(
                    ["duplicate", "redundant", "is_duplicate", "is_redundant"]
                )
                if duplicate_col:
                    integrator.viz.plot_worker_duplicates(
                        data=integrated_data,
                        worker_column=worker_col,
                        duplicate_column=duplicate_col,
                        title=f"Duplicate Submissions per {worker_col.capitalize()}",
                        filename="worker_duplicates.png",
                    )
                    viz_list.append("worker_duplicates.png")

            # Worker quality analysis
            if args.worker_quality and worker_col and quality_cols:
                print("  Generating worker quality analysis...")
                integrator.viz.plot_worker_quality_analysis(
                    data=integrated_data,
                    worker_column=worker_col,
                    quality_columns=quality_cols,
                    title=f"Quality Issues by {worker_col.capitalize()}",
                    filename="worker_quality_analysis.png",
                )
                viz_list.append("worker_quality_analysis.png")

            # Timeline visualizations
            if args.timeline_viz and date_col:
                print("  Generating timeline visualizations...")
                if breeding_col:
                    integrator.viz.plot_timeline_captures(
                        data=integrated_data,
                        date_column=date_col,
                        category_column=breeding_col,
                        category_value="Yes",
                        title="Timeline of Image Captures",
                        filename="image_timeline.png",
                        time_unit="W",
                        show_cumulative=True,
                    )
                else:
                    integrator.viz.plot_timeline_captures(
                        data=integrated_data,
                        date_column=date_col,
                        title="Timeline of Image Captures",
                        filename="image_timeline.png",
                        time_unit="W",
                        show_cumulative=True,
                    )
                viz_list.append("image_timeline.png")

            # Quality visualizations
            if args.quality_viz and quality_cols:
                print("  Generating quality visualizations...")
                integrator.viz.plot_quality_distribution(
                    data=integrated_data,
                    quality_columns=quality_cols,
                    title="Image Quality Issues",
                    filename="quality_issues.png",
                    by_worker=worker_col,
                )
                viz_list.append("quality_issues.png")

            # Container visualizations
            if args.container_viz and container_col and breeding_col:
                print("  Generating container visualizations...")
                integrator.viz.plot_container_breakdown(
                    data=integrated_data,
                    container_column=container_col,
                    breeding_column=breeding_col,
                    title="Container Types vs. Breeding Spots",
                    filename="container_breeding.png",
                )
                viz_list.append("container_breeding.png")

            # Ward-level visualizations
            if args.ward_viz and ward_col:
                print("  Generating ward visualizations...")
                if breeding_col:
                    integrator.viz.plot_worker_breeding_spots(
                        data=integrated_data,
                        worker_column=ward_col,
                        spot_column=breeding_col,
                        title=f"Breeding Spot Counts per {ward_col.capitalize()}",
                        filename="ward_breeding_spots.png",
                    )
                    viz_list.append("ward_breeding_spots.png")

                if quality_cols:
                    integrator.viz.plot_ward_summary(
                        data=integrated_data,
                        ward_column=ward_col,
                        quality_columns=quality_cols,
                        breeding_column=breeding_col,
                        title=f"{ward_col.capitalize()}-Level Quality Summary",
                        filename="ward_summary.png",
                    )
                    viz_list.append("ward_summary.png")

            # If no specific visualization was requested or visualization-only mode,
            # generate all visualizations
            if not viz_list or args.visualize_only:
                print("\nGenerating all visualizations...")
                viz_list = integrator.generate_visualizations(analysis)

                # Additionally generate worker quality analysis if worker column exists
                if worker_col and quality_cols and "worker_quality_analysis.png" not in viz_list:
                    integrator.viz.plot_worker_quality_analysis(
                        data=integrated_data,
                        worker_column=worker_col,
                        quality_columns=quality_cols,
                        title=f"Quality Issues by {worker_col.capitalize()}",
                        filename="worker_quality_analysis.png",
                    )
                    viz_list.append("worker_quality_analysis.png")
        else:
            # Generate all visualizations as part of normal flow
            print("\nGenerating visualizations...")
            viz_list = integrator.generate_visualizations(analysis)

            # Additionally generate worker quality analysis if worker column exists
            worker_col = integrator._find_column(["worker", "user", "asha", "uid", "Uid"])
            quality_cols = [
                col
                for col in integrated_data.columns
                if col.startswith("is_")
                and col
                in [
                    "is_dark",
                    "is_blurry",
                    "is_bright",
                    "is_duplicate",
                    "is_outlier",
                    "is_redundant",
                ]
            ]
            if worker_col and quality_cols:
                integrator.viz.plot_worker_quality_analysis(
                    data=integrated_data,
                    worker_column=worker_col,
                    quality_columns=quality_cols,
                    title=f"Quality Issues by {worker_col.capitalize()}",
                    filename="worker_quality_analysis.png",
                )
                viz_list.append("worker_quality_analysis.png")

        # Create annotated images if requested
        if args.annotate:
            print("\nCreating annotated images...")
            annotated_paths = integrator.create_annotated_images()
            print(f"Created {len(annotated_paths)} annotated images")

        # Export data if requested
        if args.export:
            print(f"\nExporting integrated data as {args.export}...")
            export_path = integrator.export_integrated_data(args.export)
            print(f"Data exported to: {export_path}")

        # Generate report (unless visualize-only mode)
        if not args.visualize_only:
            print("\nGenerating comprehensive report...")
            report_path = integrator.generate_report()

            # Try to open the report
            try:
                import webbrowser

                print(f"Opening report in browser: {report_path}")
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
            except Exception as e:
                print(f"  Note: Could not open report automatically ({e})")
                print(f"  Please open manually: {report_path}")

        print("\n" + "=" * 70)
        print("Processing Complete")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
