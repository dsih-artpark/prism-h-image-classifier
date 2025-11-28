#!/usr/bin/env python3
"""
Generate comprehensive ASHA worker performance and data quality report.
Privacy-first: Uses hashed UIDs in all public outputs.

QUALITY SCORE METHODOLOGY:
--------------------------
Quality Score = clean_rate − severe_issue_rate − 0.5 × duplicate_rate

Where:
    clean_rate        = 100 × (clean_images / total_images)
    severe_issue_rate = 100 × ((blurry + dark + outlier + invalid) / total_images)
    duplicate_rate    = 100 × (duplicate_images / total_images)

Scores are then clipped to stay within [0, 100].

MINIMUM THRESHOLDS:
-------------------
Workers are only ranked if they meet:
- Minimum images: 200
- Minimum active days: 5

IMAGE QUALITY CATEGORIES:
-------------------------
Based on FastDup v2.20 with configuration:
- ccthreshold: 0.9 (similarity threshold)
- outlier_distance: 0.68
"""

import argparse
import hashlib
import json

# Import existing visualization utilities
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


from src.prismh.utils.visualization_utils import VisualizationUtils


# ============================================================================
# CONFIGURATION
# ============================================================================

# Quality score weights (must sum to ~1.0)
QUALITY_WEIGHTS = {
    "duplicate": 0.40,  # Highest penalty - waste of effort
    "blur": 0.30,  # Can't use blurry images
    "dark": 0.20,  # Hard to analyze
    "outlier": 0.10,  # Least critical
}

# Minimum thresholds for fair ranking
MIN_IMAGES_FOR_RANKING = 200
MIN_DAYS_ACTIVE = 5

# Privacy salt for UID hashing (should match download script)
PRIVACY_SALT = "ASHA_PRIVACY_SALT"


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================


class WorkerAnalyzer:
    """Analyzes ASHA worker performance with privacy protections."""

    def __init__(self, metadata_file, preprocess_dir, output_dir, geojson_file=None):
        """
        Initialize worker analyzer.

        Args:
            metadata_file (Path): JSON file with downloaded metadata
            preprocess_dir (Path): Directory with preprocessed images (clean/problematic)
            output_dir (Path): Where to save analysis results
            geojson_file (Path, optional): GeoJSON file for geographic analysis
        """
        self.metadata_file = Path(metadata_file)
        self.preprocess_dir = Path(preprocess_dir)
        self.output_dir = Path(output_dir)
        self.geojson_file = Path(geojson_file) if geojson_file else None

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        self.exports_dir = self.output_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)

        # Initialize visualization utils
        self.viz = VisualizationUtils(output_dir=str(self.viz_dir))

        # Data containers
        self.metadata_df = None
        self.worker_metrics = None
        self.quality_by_worker = None

    def _hash_uid(self, uid):
        """Hash UID for privacy (8-char prefix)."""
        return hashlib.sha256(f"{uid}{PRIVACY_SALT}".encode()).hexdigest()[:8]

    def load_data(self):
        """Load metadata and preprocessing results."""
        print("Loading metadata...")
        with self.metadata_file.open("r") as f:
            data = json.load(f)

        # Handle both list and dict (with 'data' key) formats
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        self.metadata_df = pd.DataFrame(data)

        # Ensure required columns exist
        required_cols = ["Id", "Uid", "Asha worker", "Date and Time"]
        missing_cols = [col for col in required_cols if col not in self.metadata_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in metadata: {missing_cols}")

        print(f"Loaded {len(self.metadata_df)} records")

        # Parse datetime
        self.metadata_df["Date and Time"] = pd.to_datetime(
            self.metadata_df["Date and Time"], errors="coerce"
        )

        # Add quality information from preprocessing
        self._load_quality_info()

    def _load_quality_info(self):
        """Load quality information from preprocessing results."""
        print("Loading quality information from preprocessing...")

        # Map image IDs to quality status
        quality_map = {}

        # Clean images
        clean_dir = self.preprocess_dir / "clean"
        if clean_dir.exists():
            for img_path in clean_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    image_id = str(img_path.stem)
                    quality_map[image_id] = {
                        "status": "clean",
                        "problem_category": None,
                    }

        # Problematic images
        problematic_dir = self.preprocess_dir / "problematic"
        if problematic_dir.exists():
            for category_dir in problematic_dir.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    for img_path in category_dir.glob("*"):
                        if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                            image_id = str(img_path.stem)
                            quality_map[image_id] = {
                                "status": "problematic",
                                "problem_category": category,
                            }

        # Merge quality info with metadata
        self.metadata_df["quality_status"] = (
            self.metadata_df["Id"]
            .astype(str)
            .map(lambda x: quality_map.get(x, {}).get("status", "unknown"))
        )
        self.metadata_df["problem_category"] = (
            self.metadata_df["Id"]
            .astype(str)
            .map(lambda x: quality_map.get(x, {}).get("problem_category"))
        )

        print(f"Quality mapping complete: {len(quality_map)} images categorized")

        # Validate dataset alignment
        n_metadata_ids = self.metadata_df["Id"].nunique()
        n_quality_ids = len(quality_map)
        metadata_id_set = set(self.metadata_df["Id"].astype(str))
        quality_id_set = set(quality_map.keys())
        n_overlap = len(metadata_id_set & quality_id_set)

        print("\n📊 Dataset Alignment Check:")
        print(f"  Metadata IDs: {n_metadata_ids:,}")
        print(f"  FastDup processed: {n_quality_ids:,}")
        print(f"  Matched: {n_overlap:,} ({n_overlap/n_metadata_ids*100:.1f}%)")

        if n_overlap / n_metadata_ids < 0.80:
            raise ValueError(
                f"\n❌ Dataset mismatch detected!\n"
                f"Only {n_overlap:,} of {n_metadata_ids:,} metadata IDs were found in FastDup results.\n"
                f"This suggests --metadata and --preprocess-results are from different datasets.\n"
                f"Please verify you're using matching metadata and preprocessing outputs."
            )

    def _calculate_duplicate_images(self, group):
        """
        Count duplicate/redundant images for this worker based on FastDup.

        IMPORTANT: This metric comes directly from FastDup preprocessing results.
        Images in preprocess_dir/problematic/duplicates/ were identified by FastDup
        as visually near-identical (ccthreshold=0.85).

        We do NOT recompute similarity here - we only count how many of this worker's
        images FastDup flagged as duplicates during preprocessing.

        This represents redundant submissions from this worker, regardless of whether
        the duplicates overlap with other workers' submissions.
        """
        worker_duplicates = group[group["problem_category"] == "duplicates"]
        return len(worker_duplicates)

    def calculate_worker_metrics(self):
        """Calculate comprehensive metrics for each worker."""
        print("\nCalculating worker metrics...")

        # Verify grouping field
        print(f"Total records: {len(self.metadata_df)}")
        print(f"Unique workers: {self.metadata_df['Asha worker'].nunique()}")

        # Group by actual worker name (NOT Uid which is per-image!)
        grouped = self.metadata_df.groupby("Asha worker")

        metrics_list = []

        for worker_name, group in tqdm(grouped, desc="Processing workers"):
            # Basic stats
            total_images = len(group)

            # Get first Uid for this worker (for hashing/tracking)
            first_uid = group["Uid"].iloc[0] if "Uid" in group.columns else worker_name

            # Quality metrics
            clean_count = (group["quality_status"] == "clean").sum()
            problematic_count = (group["quality_status"] == "problematic").sum()
            unknown_count = total_images - (clean_count + problematic_count)  # NEW

            # Problem breakdown
            duplicate_images = self._calculate_duplicate_images(group)
            blurry = (group["problem_category"] == "blurry").sum()
            dark = (group["problem_category"] == "dark").sum()
            outliers = (group["problem_category"] == "outliers").sum()
            invalid = (group["problem_category"] == "invalid").sum()

            # Severe issues = images that are fundamentally unusable
            severe_issues = blurry + dark + outliers + invalid
            severe_issue_rate = (severe_issues / total_images * 100) if total_images > 0 else 0

            # Rates (as percentages 0-100)
            clean_rate = (clean_count / total_images * 100) if total_images > 0 else 0
            unknown_rate = (unknown_count / total_images * 100) if total_images > 0 else 0
            duplicate_rate = (duplicate_images / total_images * 100) if total_images > 0 else 0
            blur_rate = blurry / total_images if total_images > 0 else 0
            dark_rate = dark / total_images if total_images > 0 else 0
            outlier_rate = outliers / total_images if total_images > 0 else 0

            # Quality score (0-100, higher is better)
            #   Each 1% of severe issues lowers the score by 1 point,
            #   and each 2% of duplicates lowers it by 1 point.
            #   Quality Score = clean_rate − severe_issue_rate − 0.5 × duplicate_rate
            quality_score = clean_rate - severe_issue_rate - 0.5 * duplicate_rate
            quality_score = max(0, min(100, quality_score))  # Clamp to [0, 100]

            # Temporal metrics
            dates = group["Date and Time"].dropna()
            if len(dates) > 0:
                first_submission = dates.min()
                last_submission = dates.max()
                days_active = (last_submission - first_submission).days + 1
                images_per_day = total_images / days_active if days_active > 0 else 0

                # USEFUL images per day (clean only)
                useful_images_per_day = clean_count / days_active if days_active > 0 else 0
            else:
                first_submission = None
                last_submission = None
                days_active = 0
                images_per_day = 0
                useful_images_per_day = 0

            # Geographic coverage
            unique_wards = 0
            if "Ward Name" in group.columns:
                unique_wards = group["Ward Name"].nunique()

            # Determine if worker meets minimum thresholds for ranking
            meets_threshold = (
                total_images >= MIN_IMAGES_FOR_RANKING and days_active >= MIN_DAYS_ACTIVE
            )

            metrics_list.append(
                {
                    "worker_name": worker_name,  # Real name (internal only)
                    "hashed_uid": self._hash_uid(worker_name),  # Hash the worker name for privacy
                    "uid_sample": first_uid,  # Keep one UID for reference
                    "total_images": total_images,
                    "clean_images": clean_count,
                    "problematic_images": problematic_count,
                    "unknown_images": unknown_count,
                    "duplicate_images": duplicate_images,
                    "blurry_images": blurry,
                    "dark_images": dark,
                    "outlier_images": outliers,
                    "invalid_images": invalid,
                    "severe_issues": severe_issues,
                    "severe_issue_rate": severe_issue_rate,
                    "clean_rate": clean_rate,
                    "unknown_rate": unknown_rate,
                    "duplicate_rate": duplicate_rate,
                    "blur_rate": blur_rate * 100,
                    "dark_rate": dark_rate * 100,
                    "outlier_rate": outlier_rate * 100,
                    "quality_score": quality_score,
                    "first_submission": first_submission,
                    "last_submission": last_submission,
                    "days_active": days_active,
                    "images_per_day": images_per_day,
                    "useful_images_per_day": useful_images_per_day,
                    "unique_wards_covered": unique_wards,
                    "meets_ranking_threshold": meets_threshold,
                }
            )

        self.worker_metrics = pd.DataFrame(metrics_list)

        # Calculate DATA-DRIVEN thresholds (not hard-coded!)
        img_q50 = self.worker_metrics["total_images"].quantile(0.5)
        days_q50 = self.worker_metrics["days_active"].quantile(0.5)

        # Use quantile-based OR minimum sensible values
        self.min_images_threshold = max(20, img_q50)  # At least 20 images
        self.min_days_threshold = max(2, days_q50)  # At least 2 days

        # Recalculate eligibility with data-driven thresholds
        self.worker_metrics["meets_ranking_threshold"] = (
            self.worker_metrics["total_images"] >= self.min_images_threshold
        ) & (self.worker_metrics["days_active"] >= self.min_days_threshold)

        # Sort by useful_images_per_day for primary ranking
        self.worker_metrics = self.worker_metrics.sort_values(
            "useful_images_per_day", ascending=False
        )

        print(f"\nProcessed {len(self.worker_metrics)} workers")
        eligible_count = self.worker_metrics["meets_ranking_threshold"].sum()
        print(
            f"Data-driven thresholds: ≥{self.min_images_threshold:.0f} images, ≥{self.min_days_threshold:.0f} days"
        )
        print(f"Workers meeting ranking threshold: {eligible_count}")

    def calculate_temporal_patterns(self):
        """Analyze temporal submission patterns."""
        print("\nAnalyzing temporal patterns...")

        # Parse dates
        self.metadata_df["parsed_date"] = pd.to_datetime(
            self.metadata_df["Date and Time"], errors="coerce"
        )

        # Daily submission counts
        daily_submissions = (
            self.metadata_df.groupby(self.metadata_df["parsed_date"].dt.date)
            .size()
            .reset_index(name="count")
        )
        daily_submissions.columns = ["date", "submissions"]

        # Hour of day analysis
        self.metadata_df["hour"] = self.metadata_df["parsed_date"].dt.hour
        self.metadata_df["day_of_week"] = self.metadata_df["parsed_date"].dt.day_name()

        hourly_dist = self.metadata_df.groupby("hour").size()
        dow_dist = self.metadata_df.groupby("day_of_week").size()

        # Hour × Day of week heatmap data
        heatmap_data = (
            self.metadata_df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
        )

        self.temporal_data = {
            "daily": daily_submissions,
            "hourly": hourly_dist,
            "day_of_week": dow_dist,
            "heatmap": heatmap_data,
        }

        print("✅ Temporal patterns calculated")

    def calculate_geographic_coverage(self):
        """Analyze geographic distribution and coverage."""
        print("\nAnalyzing geographic coverage...")

        # Ward-level analysis
        ward_stats = (
            self.metadata_df.groupby("Ward Name")
            .agg({"Id": "count", "Asha worker": "nunique"})
            .reset_index()
        )
        ward_stats.columns = ["ward", "total_images", "unique_workers"]

        # Calculate clean rate per ward
        ward_quality = (
            self.metadata_df.groupby("Ward Name")
            .apply(
                lambda x: (x["quality_status"] == "clean").sum() / len(x) * 100 if len(x) > 0 else 0
            )
            .reset_index(name="clean_rate")
        )

        ward_stats = ward_stats.merge(
            ward_quality, left_on="ward", right_on="Ward Name", how="left"
        )
        ward_stats = ward_stats.drop("Ward Name", axis=1)

        # Create coverage bins for better map visualization
        import pandas as pd

        ward_stats["coverage_bin"] = pd.qcut(
            ward_stats["total_images"],
            q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
            duplicates="drop",
        )

        self.geographic_data = {"ward_stats": ward_stats}

        print(f"✅ Geographic coverage calculated for {len(ward_stats)} wards")

    def generate_ward_maps(self):
        """Generate interactive choropleth maps for ward-level analysis."""
        if not self.geojson_file or not self.geojson_file.exists():
            print("⚠️  No GeoJSON file provided, skipping map generation")
            return

        if not hasattr(self, "geographic_data") or "ward_stats" not in self.geographic_data:
            print("⚠️  Geographic data not calculated, skipping map generation")
            return

        print("\nGenerating ward-level maps...")

        try:
            import geopandas as gpd
            import plotly.express as px

            # Load BBMP ward boundaries
            wards_gdf = gpd.read_file(self.geojson_file)

            # Merge with our ward stats
            ward_stats = self.geographic_data["ward_stats"]

            # ---- FIX: Extract ward names from brackets ----
            # GeoJSON: "WARD NO.50 (BENNIGANAHALLI)" → "BENNIGANAHALLI"
            wards_gdf["ward_clean"] = (
                wards_gdf["name"]
                .str.extract(r"\((.*)\)", expand=False)  # text inside brackets
                .fillna(wards_gdf["name"])  # fallback to full name
                .str.replace(r"^WARD\s*NO\.?\s*\d+\s*-?\s*", "", regex=True)
                .str.strip()
                .str.upper()
            )

            # From stats: "BENNIGANAHALLI" → "BENNIGANAHALLI"
            ward_stats["ward_clean"] = ward_stats["ward"].astype(str).str.strip().str.upper()

            merged = wards_gdf.merge(ward_stats, on="ward_clean", how="left")

            # Fill NaN values for wards with no data
            merged["total_images"] = merged["total_images"].fillna(0)
            merged["clean_rate"] = merged["clean_rate"].fillna(0)
            merged["unique_workers"] = merged["unique_workers"].fillna(0)

            # ---- Minimalist grayscale map ----

            fig = px.choropleth_mapbox(
                merged,
                geojson=merged.geometry.__geo_interface__,
                locations=merged.index,
                color="total_images",
                color_continuous_scale="Greys",  # grayscale for government report
                range_color=[0, merged["total_images"].max()],
                mapbox_style="carto-positron",  # clean light basemap
                zoom=10.5,
                center={"lat": 12.9716, "lon": 77.5946},
                opacity=0.8,
                hover_name="name",
                hover_data={"total_images": ":,", "unique_workers": True, "clean_rate": ":.1f"},
                labels={
                    "total_images": "Total Images",
                    "unique_workers": "Active Workers",
                    "clean_rate": "Clean Rate (%)",
                },
            )

            fig.update_layout(
                height=600,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                coloraxis_colorbar={"title": "Images", "thickness": 15, "len": 0.7},
            )

            # Save map with proper config for interactivity
            map_path = self.viz_dir / "ward_coverage_map.html"
            fig.write_html(
                str(map_path),
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )

            print(f"✅ Ward map saved: {map_path}")
            self.ward_map_generated = True

        except ImportError as e:
            print(f"⚠️  Map generation skipped: missing dependencies ({e})")
            self.ward_map_generated = False
        except Exception as e:
            print(f"⚠️  Map generation failed: {e}")
            import traceback

            traceback.print_exc()
            self.ward_map_generated = False

    def generate_visualizations(self):
        """Generate all visualization charts."""
        print("\nGenerating visualizations...")

        # Filter to workers meeting threshold for fair comparison
        ranked_workers = self.worker_metrics[self.worker_metrics["meets_ranking_threshold"]]

        if len(ranked_workers) == 0:
            print("Warning: No workers meet minimum threshold for ranking")
            ranked_workers = self.worker_metrics.head(10)  # Fall back to top 10 by volume

        try:
            # 1. Top performers bar chart
            top_20 = ranked_workers.head(20)
            # Convert to dictionary format expected by plot_bar_chart
            bar_data = {
                f"Worker {i+1}": row["useful_images_per_day"]
                for i, (_, row) in enumerate(top_20.iterrows())
            }

            self.viz.plot_bar_chart(
                data=bar_data,
                title="Top 20 Workers by Useful Images/Day",
                xlabel="Worker",
                ylabel="Useful Images per Day",
                filename="top_performers.png",
            )

            # 2. Quality distribution pie chart
            clean_total = self.worker_metrics["clean_images"].sum()
            prob_total = self.worker_metrics["problematic_images"].sum()
            unknown_total = self.worker_metrics["unknown_images"].sum()

            self.viz.plot_pie_chart(
                data={
                    "Clean": int(clean_total),
                    "Problematic": int(prob_total),
                    "Unknown": int(unknown_total),
                },
                title="Overall Image Quality Distribution",
                filename="quality_distribution.png",
            )

            print(f"✅ Generated 2 visualizations in {self.viz_dir}")
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
            import traceback

            traceback.print_exc()
            print("    Continuing without charts...")

    def export_results(self):
        """Export analysis results to CSV files."""
        print("\nExporting results...")

        # Main summary (PUBLIC - uses hashed UIDs)
        public_summary = self.worker_metrics[
            [
                "hashed_uid",
                "total_images",
                "clean_images",
                "problematic_images",
                "unknown_images",
                "clean_rate",
                "unknown_rate",
                "severe_issue_rate",
                "duplicate_rate",
                "blur_rate",
                "dark_rate",
                "outlier_rate",
                "quality_score",
                "useful_images_per_day",
                "days_active",
                "images_per_day",
                "unique_wards_covered",
                "meets_ranking_threshold",
            ]
        ].copy()

        public_summary_path = self.exports_dir / "worker_performance_summary.csv"
        public_summary.to_csv(public_summary_path, index=False)
        print(f"✅ Public summary: {public_summary_path}")

        # Detailed quality breakdown (PUBLIC)
        quality_detail = self.worker_metrics[
            [
                "hashed_uid",
                "duplicate_images",
                "blurry_images",
                "dark_images",
                "outlier_images",
                "invalid_images",
                "severe_issues",
                "severe_issue_rate",
                "quality_score",
            ]
        ].copy()

        quality_detail_path = self.exports_dir / "worker_quality_details.csv"
        quality_detail.to_csv(quality_detail_path, index=False)
        print(f"✅ Quality details: {quality_detail_path}")

        # Training recommendations (PUBLIC - for workers below threshold)
        low_quality = self.worker_metrics[
            (self.worker_metrics["meets_ranking_threshold"])
            & (self.worker_metrics["quality_score"] < 70)
        ].copy()

        training_rec = low_quality[
            [
                "hashed_uid",
                "quality_score",
                "duplicate_rate",
                "blur_rate",
                "dark_rate",
                "total_images",
            ]
        ].copy()

        training_rec_path = self.exports_dir / "training_recommendations.csv"
        training_rec.to_csv(training_rec_path, index=False)
        print(f"✅ Training recommendations: {training_rec_path}")

        # Internal report with real names (PRIVATE - do not share)
        internal_report = self.worker_metrics.copy()
        internal_path = self.exports_dir / "worker_report_INTERNAL.csv"
        internal_report.to_csv(internal_path, index=False)
        print(f"🔒 Internal report (PRIVATE): {internal_path}")

        # Geographic coverage (ward-level stats)
        if hasattr(self, "geographic_data") and "ward_stats" in self.geographic_data:
            ward_path = self.exports_dir / "ward_coverage.csv"
            self.geographic_data["ward_stats"].to_csv(ward_path, index=False)
            print(f"✅ Ward coverage: {ward_path}")

        # Temporal patterns (daily submissions)
        if hasattr(self, "temporal_data") and "daily" in self.temporal_data:
            temporal_path = self.exports_dir / "temporal_daily.csv"
            self.temporal_data["daily"].to_csv(temporal_path, index=False)
            print(f"✅ Temporal patterns: {temporal_path}")

    def generate_html_report(self):
        """Generate interactive HTML dashboard."""
        print("\nGenerating HTML report...")

        # Get summary stats
        total_workers = len(self.worker_metrics)
        total_images = self.worker_metrics["total_images"].sum()
        overall_clean_rate = (
            (self.worker_metrics["clean_images"].sum() / total_images * 100)
            if total_images > 0
            else 0
        )
        overall_usable_rate = (
            (self.worker_metrics["clean_images"].sum() / total_images * 100)
            if total_images > 0
            else 0
        )  # Fixed: only clean images are usable

        # Aggregate issue rates
        overall_duplicate_rate = (
            (self.worker_metrics["duplicate_images"].sum() / total_images * 100)
            if total_images > 0
            else 0
        )
        overall_blur_rate = (
            (self.worker_metrics["blurry_images"].sum() / total_images * 100)
            if total_images > 0
            else 0
        )
        overall_dark_rate = (
            (self.worker_metrics["dark_images"].sum() / total_images * 100)
            if total_images > 0
            else 0
        )
        overall_outlier_rate = (
            (self.worker_metrics["outlier_images"].sum() / total_images * 100)
            if total_images > 0
            else 0
        )
        median_quality_score = self.worker_metrics["quality_score"].median()

        ranked_workers_count = self.worker_metrics["meets_ranking_threshold"].sum()

        # Analysis period (from earliest to latest submission date)
        analysis_period_str = "Not available"
        try:
            if hasattr(self, "metadata_df") and "parsed_date" in self.metadata_df.columns:
                valid_dates = self.metadata_df["parsed_date"].dropna()
                if not valid_dates.empty:
                    start_date = valid_dates.min().date()
                    end_date = valid_dates.max().date()
                    # Example: "03 Jan 2025 – 28 Feb 2025"
                    analysis_period_str = (
                        f"{start_date.strftime('%d %b %Y')} – " f"{end_date.strftime('%d %b %Y')}"
                    )
        except Exception:
            # If anything goes wrong here, we just skip showing the period
            pass

        # Top and bottom performers (meeting threshold only)
        ranked = self.worker_metrics[self.worker_metrics["meets_ranking_threshold"]]

        # Prepare data for HTML tables with short UIDs and rounded rates
        def prepare_for_display(df):
            display_df = df.copy()
            # Short UID (8 chars)
            display_df["Worker ID"] = display_df["hashed_uid"].str.slice(0, 8)
            # Round all rates to 1 decimal
            rate_cols = [
                "clean_rate",
                "severe_issue_rate",
                "duplicate_rate",
                "blur_rate",
                "dark_rate",
                "outlier_rate",
                "quality_score",
            ]
            for col in rate_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(1)
            # Round useful images/day to 2 decimals
            if "useful_images_per_day" in display_df.columns:
                display_df["useful_images_per_day"] = display_df["useful_images_per_day"].round(2)
            return display_df

        # Lens A: high-throughput workers (sorted by useful images per day)
        top_5 = prepare_for_display(ranked.head(5))

        # Lens B: high-quality workers – effective clean images per day
        ranked["effective_clean_per_day"] = ranked["useful_images_per_day"] * (
            ranked["quality_score"] / 100.0
        )
        quality_ranked = ranked.sort_values(
            ["quality_score", "effective_clean_per_day"], ascending=[False, False]
        )
        top_quality = prepare_for_display(quality_ranked.head(5))

        # Lens C: workers with highest issue rates (severe issues + duplicates)
        issue_ranked = ranked.sort_values(
            ["severe_issue_rate", "duplicate_rate"], ascending=[False, False]
        )
        bottom_5 = prepare_for_display(issue_ranked.head(5))

        # Prepare renamed columns for display
        top_5_display = top_5[
            [
                "Worker ID",
                "useful_images_per_day",
                "quality_score",
                "total_images",
                "clean_rate",
                "days_active",
            ]
        ].copy()
        top_5_display.columns = [
            "Worker ID",
            "Useful Images/Day",
            "Quality Score",
            "Total Images",
            "Clean Rate (%)",
            "Active Days",
        ]

        high_quality_display = top_quality[
            [
                "Worker ID",
                "effective_clean_per_day",
                "quality_score",
                "total_images",
                "clean_rate",
                "severe_issue_rate",
            ]
        ].copy()
        high_quality_display.columns = [
            "Worker ID",
            "Effective Clean/Day",
            "Quality Score",
            "Total Images",
            "Clean Rate (%)",
            "Severe Issue Rate (%)",
        ]

        bottom_5_display = bottom_5[
            [
                "Worker ID",
                "quality_score",
                "severe_issue_rate",
                "duplicate_rate",
                "blur_rate",
                "dark_rate",
                "clean_rate",
                "total_images",
            ]
        ].copy()
        bottom_5_display.columns = [
            "Worker ID",
            "Quality Score",
            "Severe Issue Rate (%)",
            "Duplicate Rate (%)",
            "Blur Rate (%)",
            "Dark Rate (%)",
            "Clean Rate (%)",
            "Total Images",
        ]

        # Ward stats for tables
        ward_stats = None
        top_10_wards = None
        bottom_10_wards = None
        if hasattr(self, "geographic_data") and "ward_stats" in self.geographic_data:
            ward_stats = self.geographic_data["ward_stats"].sort_values(
                "total_images", ascending=False
            )

            # Prepare ward tables with proper column names and rounding
            def prepare_ward_table(df):
                display_df = df.copy()
                display_df["Ward"] = display_df["ward"]
                display_df["Total Images"] = display_df["total_images"]
                display_df["Active Workers"] = display_df["unique_workers"]
                display_df["Clean Rate (%)"] = display_df["clean_rate"].round(1)
                return display_df[["Ward", "Total Images", "Active Workers", "Clean Rate (%)"]]

            top_10_wards = prepare_ward_table(ward_stats.head(10))
            bottom_10_wards = prepare_ward_table(
                ward_stats[ward_stats["total_images"] > 0].tail(10)
            )

        # Get actual thresholds used
        min_imgs = getattr(self, "min_images_threshold", MIN_IMAGES_FOR_RANKING)
        min_days = getattr(self, "min_days_threshold", MIN_DAYS_ACTIVE)

        # Temporal summary
        date_range_text = ""
        if hasattr(self, "temporal_data") and "daily" in self.temporal_data:
            daily_df = self.temporal_data["daily"]
            if len(daily_df) > 0:
                first_date = daily_df["date"].min()
                last_date = daily_df["date"].max()
                peak_day = daily_df.loc[daily_df["submissions"].idxmax()]
                date_range_text = f"Data collection spans from {first_date} to {last_date}. Peak activity: {peak_day['submissions']} images on {peak_day['date']}."

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ASHA Worker Performance Analysis | BBMP</title>

    <!-- Bootstrap -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />


<style>
    :root {{
        --bg-body: #ffffff;
        --bg-page: #ffffff;
        --surface-soft: #f7f7f7;
        --surface-subtle: #fafafa;
        --text-main: #111111;
        --text-muted: #555555;
        --text-soft: #888888;
    }}

    html, body {{
        height: 100%;
    }}

    body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
        background-color: var(--bg-body);
        color: var(--text-main);
        margin: 0;
    }}

    .page-shell {{
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        background: var(--bg-page);
    }}

    .page-header {{
        position: sticky;
        top: 0;
        z-index: 10;
        background: #ffffff;
        box-shadow: 0 1px 0 rgba(0, 0, 0, 0.06);
    }}

    .page-header-inner {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }}

    .header-title-block h1 {{
        font-size: 0.9rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-soft);
        margin: 0 0 4px 0;
    }}

    .header-title-block p {{
        margin: 0;
        font-size: 0.8rem;
        color: var(--text-muted);
    }}

    .header-pills {{
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }}

    .pill,
    .pill-pill {{
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.72rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--surface-subtle);
        color: var(--text-soft);
    }}

    .pill-dot {{
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: #111111;
    }}

    main.dashboard-main {{
        flex: 1;
        max-width: 1100px;
        margin: 24px auto 40px auto;
        padding: 0 16px 32px 16px;
    }}

    .section-label {{
        font-size: 0.75rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-soft);
        margin-bottom: 6px;
    }}

    .section-title-row {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 16px;
    }}

    .section-title-row h2 {{
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-main);
        margin: 0;
    }}

    .section-kicker {{
        font-size: 0.8rem;
        color: var(--text-soft);
    }}

    .card-shell {{
        background: #ffffff;
        border-radius: 6px;
        margin-bottom: 16px;
    }}

    .card-shell-soft {{
        background: var(--surface-soft);
        border-radius: 6px;
        margin-bottom: 16px;
    }}

    .card-inner {{
        padding: 16px 16px 14px 16px;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }}

    @media (max-width: 992px) {{
        .metric-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}

    @media (max-width: 576px) {{
        .metric-grid {{
            grid-template-columns: 1fr 1fr;
        }}
    }}

    .metric-card {{
        background: var(--surface-subtle);
        border-radius: 4px;
        padding: 10px 11px;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }}

    .metric-label {{
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-soft);
    }}

    .metric-value {{
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-main);
    }}

    .metric-caption {{
        font-size: 0.7rem;
        color: var(--text-soft);
    }}

    .pill-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 999px;
        background: var(--surface-subtle);
        color: var(--text-muted);
        font-size: 0.72rem;
    }}

    .pill-badge-dot {{
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: #111111;
    }}

    .light-list {{
        list-style: none;
        padding-left: 0;
        margin-bottom: 0;
        font-size: 0.8rem;
        color: var(--text-muted);
    }}

    .light-list li {{
        padding-left: 10px;
        position: relative;
        margin-bottom: 3px;
    }}

    .light-list li::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 0.55rem;
        width: 3px;
        height: 3px;
        border-radius: 999px;
        background: #bbbbbb;
    }}

    .pill-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.7rem;
        color: var(--text-soft);
        background: var(--surface-subtle);
    }}

    .pill-chip strong {{
        color: var(--text-main);
    }}

    .table-shell {{
        border-radius: 4px;
        background: #ffffff;
        margin-top: 8px;
    }}

    table {{
        width: 100%;
        table-layout: auto;
        font-size: 0.8rem;
        color: var(--text-main);
        border-collapse: collapse;
    }}

    thead {{
        background: #f3f3f3;
    }}

    thead th {{
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        color: var(--text-soft) !important;
        padding: 8px 10px;
    }}

    tbody tr:nth-child(odd) {{
        background: #ffffff;
    }}

    tbody tr:nth-child(even) {{
        background: #fafafa;
    }}

    tbody td {{
        padding: 7px 10px;
    }}

    table thead th:first-child,
    table tbody td:first-child {{
        text-align: left !important;
    }}

    table thead th:not(:first-child),
    table tbody td:not(:first-child) {{
        text-align: right !important;
    }}

    .map-card {{
        margin-top: 10px;
    }}

    .map-frame {{
        width: 100%;
        height: 520px;
        border-radius: 6px;
        background: #f5f5f5;
    }}

    .map-meta {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin-top: 8px;
        font-size: 0.76rem;
        color: var(--text-soft);
        flex-wrap: wrap;
    }}

    .map-legend-chip {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 999px;
        background: var(--surface-subtle);
        font-size: 0.72rem;
        color: var(--text-muted);
    }}

    .legend-swatch {{
        width: 10px;
        height: 10px;
        border-radius: 2px;
        background: #111111;
    }}

    .btn-outline-light-soft {{
        border-radius: 999px;
        font-size: 0.72rem;
        padding: 4px 10px;
        border: 1px solid rgba(0, 0, 0, 0.14);
        color: var(--text-main);
        background: #ffffff;
    }}

    .btn-outline-light-soft:hover {{
        border-color: #111111;
        background: #f7f7f7;
    }}

    .footer {{
        max-width: 1100px;
        margin: 0 auto 20px auto;
        padding: 10px 16px 0 16px;
        font-size: 0.7rem;
        color: var(--text-soft);
        display: flex;
        justify-content: space-between;
        gap: 8px;
        flex-wrap: wrap;
    }}

    .footer span {{
        opacity: 0.9;
    }}
</style>
</head>
<body>
<div class="page-shell">

    <!-- Sticky header -->
    <header class="page-header">
        <div class="page-header-inner">
            <div class="header-title-block">
                <h1>ASHA WORKER PERFORMANCE</h1>
                <p>Bruhat Bengaluru Mahanagara Palike (BBMP)</p>
                <p style="font-size: 0.8rem; color: var(--text-muted);">
                    Analysis period: <strong>{analysis_period_str}</strong>
                </p>
        </div>
            <div class="header-pills">
                <div class="pill">
                    <span class="pill-dot"></span>
                    <span>Live field data ingested</span>
                </div>
                <div class="pill-pill">
                    <span>View type: <strong>Ward &amp; Worker Overview</strong></span>
                </div>
            </div>
        </div>
    </header>

    <main class="dashboard-main">

        <!-- EXEC SUMMARY + KEY NOTES -->
        <section class="mb-4">
            <p class="section-label">Overview</p>
            <div class="section-title-row">
        <h2>1. Executive Summary</h2>
                <p class="section-kicker">
                    High-level snapshot of worker performance &amp; image quality across BBMP.
                </p>
            </div>

            <div class="row g-3">
                <div class="col-lg-8">
                    <div class="card-shell">
                        <div class="card-inner">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span class="pill-badge">
                                    <span class="pill-badge-dot"></span>
                                    Core coverage metrics
                                </span>
                                <span class="text-muted" style="font-size: 0.7rem;">
                                    All worker IDs are SHA256-hashed; mapping is stored separately.
                                </span>
                            </div>

                            <!-- Metric grid: update values server-side as you already do -->
                            <div class="metric-grid mt-2">
                <div class="metric-card">
                    <div class="metric-label">Total Workers</div>
                    <div class="metric-value">{total_workers}</div>
                                    <div class="metric-caption">All ASHA workers with at least one image</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Images</div>
                    <div class="metric-value">{total_images:,}</div>
                                    <div class="metric-caption">Images processed in this report window</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Usable Rate</div>
                    <div class="metric-value">{overall_usable_rate:.1f}%</div>
                                    <div class="metric-caption">Clean, non-corrupted images</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Workers Ranked</div>
                    <div class="metric-value">{ranked_workers_count}</div>
                                    <div class="metric-caption">≥{min_imgs:.0f} images, ≥{min_days:.0f} active days</div>
            </div>
        </div>

                            <div class="mt-3">
                                <ul class="light-list">
                                    <li>Clean images: {overall_clean_rate:.1f}% · Duplicates: {overall_duplicate_rate:.1f}% · Blur: {overall_blur_rate:.1f}% · Dark: {overall_dark_rate:.1f}%</li>
                                    <li>Median worker quality score: {median_quality_score:.1f} / 100</li>
                                    <li>
                                        <strong>Ranking filter:</strong> workers with ≥{min_imgs:.0f} images and ≥{min_days:.0f} active days for stable
                                        comparisons.
                                    </li>
            </ul>
                            </div>
                        </div>
                    </div>
        </div>

                <!-- Key takeaways / methodology quick view -->
                <div class="col-lg-4">
                    <div class="card-shell-soft h-100">
                        <div class="card-inner">
                            <div class="d-flex justify-content-between align-items-center mb-1">
                                <span class="section-label mb-0">Quick read</span>
                                <span class="pill-chip">
                                    <strong>Methodology</strong>
                                    <span>v1.0</span>
                                </span>
                            </div>

                            <p class="mb-2" style="font-size: 0.8rem; color: var(--text-muted);">
                                Quality is computed per worker using a weighted issue rate. Scores are on a
                                0-100 scale where higher is better.
                            </p>

                            <pre class="code-block">
Quality Score = clean_rate − severe_issue_rate − 0.5 × duplicate_rate

where:
  clean_rate        = 100 × (clean_images / total_images)
  severe_issue_rate = 100 × ((blurry + dark + outlier + invalid) / total_images)
  duplicate_rate    = 100 × (duplicate_images / total_images)
</pre>

                            <ul class="light-list mt-2">
                                <li>
                                    Image quality categories derived using FastDup (duplicates, blur, dark,
                                    outliers).
                                </li>
                                <li>
                                    Designed so field teams can act directly:
                                    <strong>who to reward, who to retrain, where to redeploy.</strong>
                                </li>
            </ul>
        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- WORKER RANKINGS -->
        <section class="mt-4">
            <p class="section-label">Worker level</p>
            <div class="section-title-row">
        <h2>2. Worker Performance Rankings</h2>
                <p class="section-kicker">
                    Balance of volume, quality, and stability across the reporting period.
                </p>
            </div>

            <div class="row g-3">
                <div class="col-lg-6">
                    <div class="card-shell-soft h-100">
                        <div class="card-inner">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h3 class="mb-0" style="font-size: 0.9rem;">Top Performing Workers</h3>
                                <span class="pill-chip">
                                    <span>Sorted by</span> <strong>Useful images / day</strong>
                                </span>
                            </div>
                            <p class="mb-2" style="font-size: 0.8rem; color: var(--text-muted);">
                                High-throughput workers sorted by useful images per day. Quality Score is shown for context;
                                some may still need quality coaching despite high volume.
                            </p>

                            <div class="table-shell mt-2">
                                {top_5_display.to_html(
                                    classes='table table-sm table-striped align-middle mb-0',
            index=False,
                                    border=0,
            float_format='%.2f',
            escape=False
        )}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="card-shell-soft h-100">
                        <div class="card-inner">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h3 class="mb-0" style="font-size: 0.9rem;">Workers with Highest Issue Rates</h3>
                                <span class="pill-chip">
                                    <strong>Action list</strong>
                                    <span>for supervision &amp; retraining</span>
                                </span>
                            </div>
                            <p class="mb-2" style="font-size: 0.8rem; color: var(--text-muted);">
                                These workers pass the minimum activity filters but show high rates of severe issues
                                (blur/dark/outlier/invalid) and/or duplicates. Useful to prioritise training or monitoring.
                            </p>

                            <div class="table-shell mt-2">
                                {bottom_5_display.to_html(
                                    classes='table table-sm table-striped align-middle mb-0',
            index=False,
                                    border=0,
                                    float_format='%.1f',
            escape=False
        )}
                            </div>
                        </div>
                    </div>
                </div>

            <!-- HIGH-QUALITY WORKERS -->
            <div class="row g-3 mt-3">
                <div class="col-12">
                    <div class="card-shell-soft h-100">
                        <div class="card-inner">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h3 class="mb-0" style="font-size: 0.9rem;">High-Quality Workers</h3>
                                <span class="pill-chip">
                                    <span>Sorted by</span> <strong>Quality Score &amp; effective clean/day</strong>
                                </span>
                            </div>
                            <p class="mb-2" style="font-size: 0.8rem; color: var(--text-muted);">
                                Workers who combine strong image quality with meaningful throughput. Effective Clean/Day
                                approximates how many high-quality images they contribute per active day.
                            </p>

                            <div class="table-shell mt-2">
                                {high_quality_display.to_html(
                                    classes='table table-sm table-striped align-middle mb-0',
            index=False,
                                    border=0,
                                    float_format='%.2f',
            escape=False
        )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            </div>
        </section>

        <!-- WARD COVERAGE + MAP -->
        <section class="mt-5">
            <p class="section-label">Geography</p>
            <div class="section-title-row">
                <h2>3. Ward-Level Coverage</h2>
                <p class="section-kicker">
                    Which parts of the city are well-covered vs under-served by ASHA submissions.
                </p>
            </div>

            <div class="card-shell map-card">
                <div class="card-inner">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <div>
                            <div class="pill-badge mb-1">
                                <span class="pill-badge-dot"></span>
                                Interactive ward-level map
        </div>
                            <p class="mb-0" style="font-size: 0.8rem; color: var(--text-muted);">
                                Darker wards represent higher submission volume. Hover to inspect exact counts and
                                quality rates.
                            </p>
                        </div>
                        <div class="d-flex flex-column align-items-end gap-1">
                            <button
                                type="button"
                                class="btn btn-outline-light-soft btn-sm"
                                onclick="openFullMap()"
                            >
                                Open full-screen map
                            </button>
                            <span style="font-size: 0.7rem; color: var(--text-soft);">
                                Source: ward_coverage.csv
                            </span>
                        </div>
                    </div>

                    <iframe
                        src="visualizations/ward_coverage_map.html"
                        class="map-frame"
                        loading="lazy"
                    ></iframe>

                    <div class="map-meta">
                        <div class="map-legend-chip">
                            <span class="legend-swatch"></span>
                            <span>Scale: light = low volume, dark = high volume</span>
                        </div>
                        <div class="d-flex flex-wrap gap-2">
                            <span class="pill-chip">
                                <strong>Top 10 wards</strong> = high volume hubs
                            </span>
                            <span class="pill-chip">
                                <strong>Bottom 10 wards</strong> = under-coverage candidates
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Ward coverage tables -->
            <div class="row g-3 mt-3">
                <div class="col-lg-6">
                    <div class="card-shell-soft h-100">
                        <div class="card-inner">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h3 class="mb-0" style="font-size: 0.9rem;">Top 10 Wards by Image Count</h3>
                                <span class="pill-chip"><strong>Deployment hubs</strong></span>
                            </div>
                            <div class="table-shell mt-2">
                                {top_10_wards.to_html(
                                    classes='table table-sm table-striped align-middle mb-0',
            index=False,
                                    border=0,
            float_format='%.1f',
            escape=False
                                ) if top_10_wards is not None else '<p class="text-muted">No ward data available</p>'}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="card-shell-soft h-100">
                        <div class="card-inner">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h3 class="mb-0" style="font-size: 0.9rem;">Under-Covered Wards (Bottom 10)</h3>
                                <span class="pill-chip"><strong>Follow-up list</strong></span>
                            </div>
                            <p class="mb-2" style="font-size: 0.8rem; color: var(--text-muted);">
                                Wards with the lowest submission volume that may need extra deployment,
                                nudging, or troubleshooting.
                            </p>
                            <div class="table-shell mt-2">
                                {bottom_10_wards.to_html(
                                    classes='table table-sm table-striped align-middle mb-0',
            index=False,
                                    border=0,
            float_format='%.1f',
            escape=False
                                ) if bottom_10_wards is not None else '<p class="text-muted">No ward data available</p>'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <footer class="footer">
        <span><strong>BBMP · Vector-borne disease surveillance</strong> — ASHA image quality &amp; coverage report</span>
        <span>All IDs anonymised using SHA256; mapping stored separately by the core project team.</span>
    </footer>

        </div>

<script>
    function openFullMap() {{
        window.open("visualizations/ward_coverage_map.html", "_blank");
    }}
</script>

</body>
</html>
"""

        report_path = self.output_dir / "index.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML report: {report_path}")

    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("ASHA WORKER PERFORMANCE ANALYSIS")
        print("=" * 80)

        self.load_data()
        self.calculate_worker_metrics()
        self.calculate_temporal_patterns()
        self.calculate_geographic_coverage()
        self.generate_ward_maps()  # NEW - Generate ward-level maps
        self.generate_visualizations()
        self.export_results()
        self.generate_html_report()

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nView report: {self.output_dir / 'index.html'}")
        print(f"CSV exports: {self.exports_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ASHA worker performance and data quality report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Example usage:
  python generate_worker_report.py \\
    --metadata report_images/downloaded_metadata.json \\
    --preprocess-results report_preprocess_results \\
    --output worker_analysis_report

Configuration:
  - Quality weights: See QUALITY_WEIGHTS in script
  - Minimum thresholds: {MIN_IMAGES_FOR_RANKING} images, {MIN_DAYS_ACTIVE} days
  - Privacy: UIDs hashed with SHA256
        """,
    )

    parser.add_argument(
        "--metadata", required=True, type=Path, help="Path to downloaded_metadata.json"
    )
    parser.add_argument(
        "--preprocess-results",
        required=True,
        type=Path,
        help="Path to preprocessing output directory (contains clean/ and problematic/)",
    )
    parser.add_argument(
        "--output",
        default="worker_analysis_report",
        type=Path,
        help="Output directory for report and exports",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        help="Optional: BBMP ward boundaries GeoJSON for geographic analysis",
    )

    args = parser.parse_args()

    analyzer = WorkerAnalyzer(
        metadata_file=args.metadata,
        preprocess_dir=args.preprocess_results,
        output_dir=args.output,
        geojson_file=args.geojson,
    )

    analyzer.run_analysis()


if __name__ == "__main__":
    main()
