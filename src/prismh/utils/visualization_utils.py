#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


class VisualizationUtils:
    """
    A class that provides reusable visualization functions for creating
    high-quality, insightful visualizations for various types of data.
    """

    def __init__(
        self,
        output_dir: str = "visualizations",
        style: str = "seaborn-v0_8-whitegrid",
        color_palette: str = "viridis",
        dpi: int = 300,
        fig_width: int = 12,
        fig_height: int = 8,
    ):
        """
        Initialize the visualization utilities with specified settings.

        Args:
            output_dir (str): Directory to save visualizations
            style (str): Matplotlib style to use
            color_palette (str): Seaborn color palette to use
            dpi (int): DPI for output images
            fig_width (int): Default figure width
            fig_height (int): Default figure height
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.fig_width = fig_width
        self.fig_height = fig_height

        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use(style)
        sns.set_palette(color_palette)

        # Custom color scheme
        self.colors = {
            "dark": "#2C3E50",  # Dark blue/slate
            "blurry": "#E74C3C",  # Red
            "bright": "#F39C12",  # Orange
            "duplicate": "#3498DB",  # Blue
            "outlier": "#9B59B6",  # Purple
            "invalid": "#1ABC9C",  # Turquoise
            "good": "#2ECC71",  # Green
            "main": "#2980B9",  # Blue
            "accent": "#F1C40F",  # Yellow
            "neutral": "#95A5A6",  # Gray
        }

        self.bar_colors = [
            self.colors["dark"],
            self.colors["blurry"],
            self.colors["bright"],
            self.colors["duplicate"],
            self.colors["outlier"],
            self.colors["invalid"],
        ]

    def plot_bar_chart(
        self,
        data: dict[str, int],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        total: int | None = None,
        sort: bool = True,
        colors: list | None = None,
        horizontal: bool = False,
        add_percentage: bool = True,
    ) -> plt.Figure:
        """
        Create a bar chart visualization.

        Args:
            data (Dict[str, int]): Dictionary of category names and their values
            title (str): Chart title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            filename (str): Output filename
            total (int, optional): Total for percentage calculation
            sort (bool): Whether to sort data by values (default True)
            colors (List, optional): List of colors for bars
            horizontal (bool): Whether to create a horizontal bar chart
            add_percentage (bool): Whether to add percentage labels

        Returns:
            plt.Figure: The generated figure
        """
        # Create sorted items if requested
        if sort:
            items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            labels, values = zip(*items, strict=False)
        else:
            labels = list(data.keys())
            values = list(data.values())

        if colors is None:
            colors = self.bar_colors[: len(data)]

        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        if horizontal:
            bars = ax.barh(
                labels, values, color=colors, edgecolor="white", linewidth=1.5, alpha=0.85
            )

            # Add count labels
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}",
                    va="center",
                    ha="left",
                    fontweight="bold",
                    fontsize=11,
                )
        else:
            bars = ax.bar(
                labels, values, color=colors, edgecolor="white", linewidth=1.5, alpha=0.85
            )

            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                )

        # Add percentage labels if requested
        if add_percentage and total is not None:
            for i, value in enumerate(values):
                percentage = value / total * 100
                if percentage > 3:  # Only show if enough space
                    if horizontal:
                        ax.text(
                            value / 2,
                            i,
                            f"{percentage:.1f}%",
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                            fontsize=10,
                        )
                    else:
                        ax.text(
                            i,
                            value / 2,
                            f"{percentage:.1f}%",
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                            fontsize=10,
                        )

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=11)

        # Add info about total if provided
        if total is not None:
            plt.figtext(0.5, 0.01, f"Total: {total}", ha="center", fontsize=12, fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_pie_chart(
        self,
        data: dict[str, int],
        title: str,
        filename: str,
        colors: list | None = None,
        explode: list[float] | None = None,
        add_legend: bool = True,
        add_count_annotation: bool = True,
    ) -> plt.Figure:
        """
        Create a pie chart visualization.

        Args:
            data (Dict[str, int]): Dictionary of category names and their values
            title (str): Chart title
            filename (str): Output filename
            colors (List, optional): List of colors for segments
            explode (List[float], optional): List of explosion values for segments
            add_legend (bool): Whether to add a legend
            add_count_annotation (bool): Whether to add count annotation

        Returns:
            plt.Figure: The generated figure
        """
        labels = list(data.keys())
        sizes = list(data.values())

        if colors is None:
            colors = self.bar_colors[: len(data)]

        if explode is None:
            explode = [0.1 if i == 0 else 0 for i in range(len(data))]

        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=None if add_legend else labels,  # No labels if legend is used
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )

        # Styling
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(12)

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Add legend if requested
        if add_legend:
            ax.legend(
                wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
            )

        # Add count annotation if requested
        if add_count_annotation:
            total = sum(sizes)
            annotation_text = f"Total: {total}"

            # Add individual counts
            for label, size in zip(labels, sizes, strict=False):
                annotation_text += f"\n{label}: {size}"

            plt.annotate(
                annotation_text,
                xy=(0.5, 0.02),
                xycoords="figure fraction",
                ha="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
            )

        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_histogram(
        self,
        data: list[float],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        bins: int = 20,
        color: str = None,
        add_vertical_line: float | None = None,
        line_label: str | None = None,
    ) -> plt.Figure:
        """
        Create a histogram visualization.

        Args:
            data (List[float]): Data to plot in histogram
            title (str): Chart title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            filename (str): Output filename
            bins (int): Number of bins
            color (str, optional): Color for histogram bars
            add_vertical_line (float, optional): Value to add vertical line at
            line_label (str, optional): Label for vertical line

        Returns:
            plt.Figure: The generated figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        if color is None:
            color = self.colors["main"]

        # Create histogram
        n, bins, patches = ax.hist(
            data, bins=bins, color=color, edgecolor="white", linewidth=1, alpha=0.8
        )

        # Color bars by value (gradient)
        if isinstance(color, str):
            # Use a colormap if single color provided
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            cmap = plt.cm.get_cmap("viridis")
            norm = plt.Normalize(min(data), max(data))

            for c, p in zip(bin_centers, patches, strict=False):
                plt.setp(p, "facecolor", cmap(norm(c)))

        # Add vertical line if requested
        if add_vertical_line is not None:
            ax.axvline(x=add_vertical_line, color="red", linestyle="--", linewidth=2.5, alpha=0.8)

            # Add line label if provided
            if line_label is not None:
                ax.text(
                    add_vertical_line + 0.02,
                    ax.get_ylim()[1] * 0.9,
                    line_label,
                    color="red",
                    fontweight="bold",
                    fontsize=12,
                    bbox=dict(
                        facecolor="white", alpha=0.7, edgecolor="red", boxstyle="round,pad=0.5"
                    ),
                )

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=11)

        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_time_series(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_columns: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        colors: list[str] | None = None,
        add_markers: bool = True,
        secondary_y_column: str | None = None,
        secondary_ylabel: str | None = None,
    ) -> plt.Figure:
        """
        Create a time series visualization.

        Args:
            data (pd.DataFrame): DataFrame containing the data
            date_column (str): Column name for dates
            value_columns (List[str]): Column names for values to plot
            title (str): Chart title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            filename (str): Output filename
            colors (List[str], optional): Colors for lines
            add_markers (bool): Whether to add markers
            secondary_y_column (str, optional): Column name for secondary y-axis
            secondary_ylabel (str, optional): Label for secondary y-axis

        Returns:
            plt.Figure: The generated figure
        """
        # Create figure
        fig, ax1 = plt.subplots(figsize=(self.fig_width, self.fig_height))

        if colors is None:
            colors = [
                self.colors["main"] if i == 0 else self.bar_colors[i % len(self.bar_colors)]
                for i in range(len(value_columns))
            ]

        if secondary_y_column is not None and secondary_y_column in value_columns:
            ax2 = ax1.twinx()
            primary_columns = [col for col in value_columns if col != secondary_y_column]
            secondary_columns = [secondary_y_column]
        else:
            primary_columns = value_columns
            secondary_columns = []

        # Plot primary columns
        for i, col in enumerate(primary_columns):
            marker = "o" if add_markers else None
            ax1.plot(
                data[date_column],
                data[col],
                marker=marker,
                linewidth=2.5,
                label=col,
                color=colors[i],
                markersize=6 if add_markers else 0,
                alpha=0.85,
            )

        # Plot secondary column if requested
        if secondary_columns:
            marker = "o" if add_markers else None
            ax2.plot(
                data[date_column],
                data[secondary_columns[0]],
                marker=marker,
                linewidth=2.5,
                label=secondary_columns[0],
                color=colors[len(primary_columns)],
                markersize=6 if add_markers else 0,
                linestyle="--",
                alpha=0.7,
            )

            if secondary_ylabel is not None:
                ax2.set_ylabel(secondary_ylabel, fontsize=13, fontweight="bold")

        # Styling
        ax1.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax1.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax1.set_ylabel(ylabel, fontsize=13, fontweight="bold")

        # Enhance grid
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Improve legend
        if secondary_columns:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper right",
                fontsize=11,
                framealpha=0.9,
                facecolor="white",
                edgecolor="gray",
            )
        else:
            ax1.legend(
                loc="upper right", fontsize=11, framealpha=0.9, facecolor="white", edgecolor="gray"
            )

        # Format date axis
        fig.autofmt_xdate()

        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_image_grid(
        self,
        image_paths: list[str | Path],
        title: str,
        filename: str,
        cols: int = 4,
        figsize: tuple[int, int] = None,
        max_images: int = 16,
    ) -> plt.Figure:
        """
        Create a grid of images.

        Args:
            image_paths (List[Union[str, Path]]): List of paths to images
            title (str): Grid title
            filename (str): Output filename
            cols (int): Number of columns in grid
            figsize (Tuple[int, int], optional): Figure size
            max_images (int): Maximum number of images to display

        Returns:
            plt.Figure: The generated figure
        """
        # Limit number of images
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]

        # Calculate rows needed
        n_images = len(image_paths)
        rows = (n_images + cols - 1) // cols

        if figsize is None:
            figsize = (self.fig_width, rows * (self.fig_height / 4))

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Make axes a 2D array even if it's 1D
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Fill grid with images
        for i, ax in enumerate(axes.flat):
            if i < n_images:
                try:
                    img_path = Path(image_paths[i])  # Ensure Path object
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(img_path.name, fontsize=10)  # Use Path.name
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error: {str(e)}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="red",
                    )

            # Hide axes ticks
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Give space for the title

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_heatmap(
        self,
        data: pd.DataFrame | np.ndarray,
        title: str,
        filename: str,
        xlabel: str | None = None,
        ylabel: str | None = None,
        cmap: str = "viridis",
        annot: bool = True,
        fmt: str = ".2f",
        linewidths: float = 0.5,
    ) -> plt.Figure:
        """
        Create a heatmap visualization.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Data for heatmap
            title (str): Heatmap title
            filename (str): Output filename
            xlabel (str, optional): X-axis label
            ylabel (str, optional): Y-axis label
            cmap (str): Colormap to use
            annot (bool): Whether to annotate cells
            fmt (str): Format for annotations
            linewidths (float): Width of lines between cells

        Returns:
            plt.Figure: The generated figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        sns.heatmap(data, annot=annot, fmt=fmt, linewidths=linewidths, cmap=cmap, ax=ax)

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")

        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")

        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def generate_html_report(
        self,
        title: str,
        filename: str = "report.html",
        visualizations: list[str] = None,
        summary_data: dict[str, Any] = None,
        additional_text: str = None,
    ) -> str:
        """
        Generate an HTML report with visualizations and summary data.

        Args:
            title (str): Report title
            filename (str): Output filename
            visualizations (List[str], optional): List of visualization filenames to include
            summary_data (Dict[str, Any], optional): Summary data to include in the report
            additional_text (str, optional): Additional text to include in the report

        Returns:
            str: Path to the generated HTML file
        """
        # Find all visualization files if not provided
        if visualizations is None:
            visualizations = []
            for entry in self.output_dir.iterdir():  # Use iterdir
                if (
                    entry.is_file()
                    and entry.suffix.lower() in [".png", ".jpg", ".jpeg"]
                    and "sample" not in entry.name
                ):
                    visualizations.append(entry.name)  # Use entry.name

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                h1, h2, h3 {{
                    color: #2C3E50;
                }}
                h1 {{
                    border-bottom: 2px solid #3498DB;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }}
                .dashboard-header {{
                    background-color: #2C3E50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .summary-box {{
                    background-color: white;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .viz-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin-bottom: 30px;
                }}
                .viz-item {{
                    flex: 0 0 calc(50% - 20px);
                    background-color: white;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .viz-wide {{
                    flex: 0 0 calc(100% - 20px);
                }}
                .viz-item img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                .viz-title {{
                    text-align: center;
                    font-weight: bold;
                    margin: 15px 0;
                    color: #2C3E50;
                    font-size: 1.1em;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
                .good {{ color: #2ECC71; }}
                .problem {{ color: #E74C3C; }}
                .medium {{ color: #F39C12; }}
                @media (max-width: 768px) {{
                    .viz-item {{
                        flex: 0 0 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{title}</h1>
                <p>Analysis and visualization of image data</p>
            </div>
        """

        # Add summary data if provided
        if summary_data is not None and len(summary_data) > 0:
            html += """
            <div class="summary-box">
                <h2>Analysis Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """

            for key, value in summary_data.items():
                # Format the key for display
                display_key = key.replace("_", " ").title()

                # Determine styling based on key
                css_class = ""
                if "clean" in key.lower() or "good" in key.lower():
                    css_class = "good"
                elif any(
                    problem in key.lower()
                    for problem in ["problem", "dark", "blurry", "invalid", "duplicate", "outlier"]
                ):
                    css_class = "problem"

                # Format the value (add percentage if it's a tuple)
                if isinstance(value, tuple) and len(value) == 2:
                    html += f"""
                    <tr>
                        <td>{display_key}</td>
                        <td><span class="{css_class}">{value[0]} ({value[1]:.1f}%)</span></td>
                    </tr>
                    """
                else:
                    html += f"""
                    <tr>
                        <td>{display_key}</td>
                        <td><span class="{css_class}">{value}</span></td>
                    </tr>
                    """

            html += """
                </table>
            </div>
            """

        # Add additional text if provided
        if additional_text is not None:
            html += f"""
            <div class="summary-box">
                <h2>Additional Information</h2>
                <p>{additional_text}</p>
            </div>
            """

        # Add visualizations if provided
        if visualizations is not None and len(visualizations) > 0:
            html += """
            <h2>Visualizations</h2>
            <div class="viz-container">
            """

            for viz_file in visualizations:
                # Extract title from filename (Path object not needed here)
                viz_title = Path(viz_file).stem.replace("_", " ").title()

                html += f"""
                <div class="viz-item">
                    <div class="viz-title">{viz_title}</div>
                    <img src="{viz_file}" alt="{viz_title}">
                </div>
                """

            html += """
            </div>
            """

        # Add footer
        html += f"""
            <div class="footer">
                <p>Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</p>
                <p>Output directory: {self.output_dir.resolve()}</p> # Use resolve for absolute path
            </div>
        </body>
        </html>
        """

        # Save HTML file
        output_path = self.output_dir / filename
        output_path.write_text(html)  # Use write_text

        return str(output_path)  # Return path as string

    def plot_worker_breeding_spots(
        self,
        data: pd.DataFrame,
        worker_column: str,
        spot_column: str,
        title: str = "Breeding Spot Counts per Worker",
        filename: str = "worker_breeding_spots.png",
        horizontal: bool = True,
        max_workers: int = 15,
        spot_yes_value: str = "Yes",
    ) -> plt.Figure:
        """
        Create a horizontal bar chart showing breeding spot counts by worker or ward.

        Args:
            data (pd.DataFrame): DataFrame with worker and breeding spot data
            worker_column (str): Column name for worker or ward identifier
            spot_column (str): Column name for breeding spot indicator
            title (str): Chart title
            filename (str): Output filename
            horizontal (bool): Whether to create a horizontal bar chart
            max_workers (int): Maximum number of workers to show
            spot_yes_value (str): Value that indicates a positive breeding spot

        Returns:
            plt.Figure: The generated figure
        """
        if data.empty or worker_column not in data.columns or spot_column not in data.columns:
            print("Error: Missing required columns for worker breeding spots visualization")
            return None

        # Aggregate data by worker and spot status
        worker_spots = data.groupby([worker_column, spot_column]).size().unstack(fill_value=0)

        # Ensure we have Yes and No columns
        if spot_yes_value not in worker_spots.columns:
            worker_spots[spot_yes_value] = 0
        if "No" not in worker_spots.columns:
            worker_spots["No"] = 0

        # Calculate total spots per worker and sort
        worker_spots["Total"] = worker_spots.sum(axis=1)
        worker_spots = worker_spots.sort_values("Total", ascending=False)

        # Limit to top N workers
        if len(worker_spots) > max_workers:
            worker_spots = worker_spots.head(max_workers)

        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Prepare data for stacked bar chart
        workers = worker_spots.index
        yes_counts = worker_spots[spot_yes_value]
        no_counts = worker_spots["No"]

        bar_width = 0.75

        if horizontal:
            ax.barh(
                workers,
                yes_counts,
                bar_width,
                label=f"Breeding Spot ({spot_yes_value})",
                color=self.colors["accent"],
            )
            ax.barh(
                workers,
                no_counts,
                bar_width,
                left=yes_counts,
                label="No Breeding Spot",
                color=self.colors["neutral"],
            )

            # Add count labels
            for i, worker in enumerate(workers):
                # Yes count label
                if yes_counts[worker] > 0:
                    ax.text(
                        yes_counts[worker] / 2,
                        i,
                        f"{int(yes_counts[worker])}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                        fontsize=10,
                    )

                # No count label
                if no_counts[worker] > 0:
                    ax.text(
                        yes_counts[worker] + no_counts[worker] / 2,
                        i,
                        f"{int(no_counts[worker])}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                        fontsize=10,
                    )

                # Total count label
                ax.text(
                    yes_counts[worker] + no_counts[worker] + 0.5,
                    i,
                    f"Total: {int(yes_counts[worker] + no_counts[worker])}",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
        else:
            ax.bar(
                workers,
                yes_counts,
                bar_width,
                label=f"Breeding Spot ({spot_yes_value})",
                color=self.colors["accent"],
            )
            ax.bar(
                workers,
                no_counts,
                bar_width,
                bottom=yes_counts,
                label="No Breeding Spot",
                color=self.colors["neutral"],
            )

            # Add count labels
            for i, worker in enumerate(workers):
                # Yes count label
                if yes_counts[worker] > 0:
                    ax.text(
                        i,
                        yes_counts[worker] / 2,
                        f"{int(yes_counts[worker])}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                        fontsize=10,
                    )

                # No count label
                if no_counts[worker] > 0:
                    ax.text(
                        i,
                        yes_counts[worker] + no_counts[worker] / 2,
                        f"{int(no_counts[worker])}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                        fontsize=10,
                    )

                # Total count label
                ax.text(
                    i,
                    yes_counts[worker] + no_counts[worker] + 0.5,
                    f"Total: {int(yes_counts[worker] + no_counts[worker])}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=90,
                )

        # Calculate percentage of breeding spots for each worker
        pct_yes = yes_counts / (yes_counts + no_counts) * 100

        # Add percentage annotation to the right side (horizontal) or top (vertical)
        for i, worker in enumerate(workers):
            if horizontal:
                ax.text(
                    ax.get_xlim()[1] * 0.95,
                    i,
                    f"{pct_yes[worker]:.1f}%",
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=self.colors["main"],
                )
            else:
                ax.text(
                    i,
                    ax.get_ylim()[1] * 0.95,
                    f"{pct_yes[worker]:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    color=self.colors["main"],
                    rotation=90,
                )

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        if horizontal:
            ax.set_xlabel("Number of Images", fontsize=13, fontweight="bold")
            ax.set_ylabel(worker_column, fontsize=13, fontweight="bold")
        else:
            ax.set_ylabel("Number of Images", fontsize=13, fontweight="bold")
            ax.set_xlabel(worker_column, fontsize=13, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=11)

        # Add legend
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fontsize=11
        )

        # Add explanatory text
        plt.figtext(
            0.01,
            0.01,
            "Percentage values show proportion of images labeled as breeding spots.",
            ha="left",
            fontsize=10,
            fontstyle="italic",
        )

        plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_timeline_captures(
        self,
        data: pd.DataFrame,
        date_column: str,
        category_column: str | None = None,
        category_value: str | None = None,
        title: str = "Timeline of Image Captures",
        filename: str = "timeline_captures.png",
        time_unit: str = "D",
        show_cumulative: bool = False,
    ) -> plt.Figure:
        """
        Create a timeline visualization of image captures over time.

        Args:
            data (pd.DataFrame): DataFrame with date information
            date_column (str): Column name containing date/time data
            category_column (str, optional): Column to use for categorization (e.g., "Breeding spot")
            category_value (str, optional): Value in category column that indicates positive cases
            title (str): Chart title
            filename (str): Output filename
            time_unit (str): Time unit for grouping ('D'=daily, 'W'=weekly, 'M'=monthly)
            show_cumulative (bool): Whether to include cumulative line

        Returns:
            plt.Figure: The generated figure
        """
        if data.empty or date_column not in data.columns:
            print("Error: Missing required date column for timeline visualization")
            return None

        # Ensure date column is datetime
        df = data.copy()
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            # Remove rows with invalid dates
            df = df.dropna(subset=[date_column])
        except Exception as e:
            print(f"Error converting {date_column} to datetime: {e}")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Group by time unit
        if time_unit == "D":
            df["time_period"] = df[date_column].dt.date
            time_format = "%Y-%m-%d"
            period_label = "Day"
        elif time_unit == "W":
            df["time_period"] = df[date_column].dt.to_period("W").dt.start_time.dt.date
            time_format = "%Y-%m-%d"
            period_label = "Week"
        elif time_unit == "M":
            df["time_period"] = df[date_column].dt.to_period("M").dt.start_time.dt.date
            time_format = "%Y-%m"
            period_label = "Month"
        else:
            print(f"Invalid time unit: {time_unit}")
            time_unit = "D"
            df["time_period"] = df[date_column].dt.date
            time_format = "%Y-%m-%d"
            period_label = "Day"

        # Count submissions by time period
        if category_column and category_column in df.columns:
            # For breeding spots, count Yes vs No
            if category_value:
                positive_counts = (
                    df[df[category_column] == category_value].groupby("time_period").size()
                )
                negative_counts = (
                    df[df[category_column] != category_value].groupby("time_period").size()
                )

                # Get complete date range
                all_dates = sorted(list(set(positive_counts.index) | set(negative_counts.index)))

                # Create bar chart with two categories
                width = 0.8
                timeline_df = pd.DataFrame(index=all_dates)
                timeline_df["positive"] = pd.Series(positive_counts)
                timeline_df["negative"] = pd.Series(negative_counts)
                timeline_df = timeline_df.fillna(0)

                # Sort by date
                timeline_df = timeline_df.sort_index()

                # Plot stacked bars
                ax.bar(
                    range(len(timeline_df)),
                    timeline_df["positive"],
                    width,
                    label=f"{category_column}={category_value}",
                    color=self.colors["accent"],
                )
                ax.bar(
                    range(len(timeline_df)),
                    timeline_df["negative"],
                    width,
                    bottom=timeline_df["positive"],
                    label=f"{category_column} {category_value}",
                    color=self.colors["neutral"],
                )

                # Set custom x-ticks
                date_labels = [d.strftime(time_format) for d in timeline_df.index]
                if len(date_labels) > 12:
                    # Show fewer tick labels to avoid crowding
                    tick_step = max(1, len(date_labels) // 12)
                    ax.set_xticks(range(0, len(timeline_df), tick_step))
                    ax.set_xticklabels(
                        [date_labels[i] for i in range(0, len(date_labels), tick_step)], rotation=45
                    )
                else:
                    ax.set_xticks(range(len(timeline_df)))
                    ax.set_xticklabels(date_labels, rotation=45)

                # Add cumulative line if requested
                if show_cumulative:
                    timeline_df["cumulative"] = timeline_df["positive"].cumsum()

                    # Create secondary y-axis for cumulative
                    ax2 = ax.twinx()
                    ax2.plot(
                        range(len(timeline_df)),
                        timeline_df["cumulative"],
                        "o-",
                        color=self.colors["main"],
                        linewidth=2.5,
                        markersize=6,
                        label="Cumulative Breeding Spots",
                    )

                    ax2.set_ylabel(
                        "Cumulative Count",
                        fontsize=13,
                        fontweight="bold",
                        color=self.colors["main"],
                    )
                    ax2.tick_params(axis="y", labelcolor=self.colors["main"])

                    # Combine legends
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)
                else:
                    ax.legend(fontsize=11)
            else:
                # Group by category
                category_counts = (
                    df.groupby(["time_period", category_column]).size().unstack(fill_value=0)
                )

                # Sort by date
                category_counts = category_counts.sort_index()

                # Plot grouped bars for each category
                category_counts.plot(
                    kind="bar",
                    ax=ax,
                    width=0.8,
                    color=self.bar_colors[: len(category_counts.columns)],
                )

                # Format x-axis
                date_labels = [d.strftime(time_format) for d in category_counts.index]
                ax.set_xticklabels(date_labels, rotation=45)
        else:
            # Simple count by time period
            time_counts = df.groupby("time_period").size()

            # Sort by date
            time_counts = time_counts.sort_index()

            # Plot bars
            bars = ax.bar(
                range(len(time_counts)), time_counts.values, width=0.8, color=self.colors["main"]
            )

            # Set custom x-ticks
            date_labels = [d.strftime(time_format) for d in time_counts.index]
            if len(date_labels) > 12:
                # Show fewer tick labels to avoid crowding
                tick_step = max(1, len(date_labels) // 12)
                ax.set_xticks(range(0, len(time_counts), tick_step))
                ax.set_xticklabels(
                    [date_labels[i] for i in range(0, len(date_labels), tick_step)], rotation=45
                )
            else:
                ax.set_xticks(range(len(time_counts)))
                ax.set_xticklabels(date_labels, rotation=45)

            # Add count labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.5,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=9,
                    )

            # Add cumulative line if requested
            if show_cumulative:
                cumulative = time_counts.cumsum()

                # Create secondary y-axis for cumulative
                ax2 = ax.twinx()
                ax2.plot(
                    range(len(time_counts)),
                    cumulative.values,
                    "o-",
                    color=self.colors["accent"],
                    linewidth=2.5,
                    markersize=6,
                    label="Cumulative Count",
                )

                ax2.set_ylabel(
                    "Cumulative Count", fontsize=13, fontweight="bold", color=self.colors["accent"]
                )
                ax2.tick_params(axis="y", labelcolor=self.colors["accent"])

                # Add legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

        # Add annotations for data patterns
        if len(df) > 0:
            avg_per_period = len(df) / max(1, len(pd.unique(df["time_period"])))
            plt.figtext(
                0.01,
                0.01,
                f"Average: {avg_per_period:.1f} images per {period_label.lower()}",
                ha="left",
                fontsize=10,
                fontstyle="italic",
            )

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel(f"Time ({period_label})", fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Images", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False) if not show_cumulative else None
        ax.tick_params(axis="both", which="major", labelsize=11)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_worker_duplicates(
        self,
        data: pd.DataFrame,
        worker_column: str,
        duplicate_column: str,
        title: str = "Duplicates per Worker",
        filename: str = "worker_duplicates.png",
        max_workers: int = 15,
        sort_by: str = "percentage",
    ) -> plt.Figure:
        """
        Create a visualization of duplicates/redundant submissions per worker.

        Args:
            data (pd.DataFrame): DataFrame with worker and duplicate data
            worker_column (str): Column name for worker identifier
            duplicate_column (str): Column that indicates duplicate/redundant status
            title (str): Chart title
            filename (str): Output filename
            max_workers (int): Maximum number of workers to show
            sort_by (str): Sort criteria ('total', 'duplicates', or 'percentage')

        Returns:
            plt.Figure: The generated figure
        """
        if data.empty or worker_column not in data.columns or duplicate_column not in data.columns:
            print("Error: Missing required columns for worker duplicates visualization")
            return None

        # Make sure duplicate column is boolean
        if data[duplicate_column].dtype != bool:
            # Check if it's 'Yes'/'No' or 1/0
            if data[duplicate_column].dtype == object:
                data[duplicate_column] = (
                    data[duplicate_column].map({"Yes": True, "No": False}).fillna(False)
                )
            else:
                data[duplicate_column] = data[duplicate_column].astype(bool)

        # Count total submissions and duplicates by worker
        worker_stats = data.groupby(worker_column).agg({duplicate_column: ["sum", "count"]})

        # Flatten column names
        worker_stats.columns = ["duplicates", "total"]

        # Calculate percentage
        worker_stats["percentage"] = (
            worker_stats["duplicates"] / worker_stats["total"] * 100
        ).round(1)

        # Sort based on criteria
        if sort_by == "percentage":
            worker_stats = worker_stats.sort_values("percentage", ascending=False)
        elif sort_by == "duplicates":
            worker_stats = worker_stats.sort_values("duplicates", ascending=False)
        else:  # sort by total
            worker_stats = worker_stats.sort_values("total", ascending=False)

        # Limit to top N workers
        if len(worker_stats) > max_workers:
            worker_stats = worker_stats.head(max_workers)

        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Set up data for horizontal bars
        workers = worker_stats.index
        duplicates = worker_stats["duplicates"]
        originals = worker_stats["total"] - worker_stats["duplicates"]
        percentages = worker_stats["percentage"]

        # Create horizontal stacked bars
        ax.barh(workers, originals, color=self.colors["good"], label="Original Images")
        ax.barh(
            workers,
            duplicates,
            left=originals,
            color=self.colors["duplicate"],
            label="Duplicate Images",
        )

        # Add percentage labels
        for i, worker in enumerate(workers):
            # Only add percentage label if there are duplicates
            if duplicates[worker] > 0:
                ax.text(
                    originals[worker] + duplicates[worker] + 0.5,
                    i,
                    f"{percentages[worker]:.1f}%",
                    va="center",
                    ha="left",
                    fontweight="bold",
                    fontsize=10,
                )

            # Add count labels inside bars
            # Original count
            if originals[worker] > 3:
                ax.text(
                    originals[worker] / 2,
                    i,
                    f"{int(originals[worker])}",
                    va="center",
                    ha="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10,
                )

            # Duplicate count
            if duplicates[worker] > 3:
                ax.text(
                    originals[worker] + duplicates[worker] / 2,
                    i,
                    f"{int(duplicates[worker])}",
                    va="center",
                    ha="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10,
                )

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Number of Images", fontsize=13, fontweight="bold")
        ax.set_ylabel(worker_column, fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=11)

        # Add legend
        ax.legend(loc="upper right", bbox_to_anchor=(1, -0.12), ncol=2, frameon=True, fontsize=11)

        # Add explanatory text
        plt.figtext(
            0.01,
            0.01,
            "Percentage values show proportion of duplicate images per worker.",
            ha="left",
            fontsize=10,
            fontstyle="italic",
        )

        plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_container_breakdown(
        self,
        data: pd.DataFrame,
        container_column: str,
        breeding_column: str,
        title: str = "Container Type vs. Breeding Spots",
        filename: str = "container_breakdown.png",
        breeding_yes_value: str = "Yes",
        min_count: int = 5,
    ) -> plt.Figure:
        """
        Create a breakdown of container types versus detected breeding spots.

        Args:
            data (pd.DataFrame): DataFrame with container and breeding spot data
            container_column (str): Column name for container type
            breeding_column (str): Column name for breeding spot indicator
            title (str): Chart title
            filename (str): Output filename
            breeding_yes_value (str): Value that indicates a positive breeding spot
            min_count (int): Minimum count to include a container type

        Returns:
            plt.Figure: The generated figure
        """
        if (
            data.empty
            or container_column not in data.columns
            or breeding_column not in data.columns
        ):
            print("Error: Missing required columns for container breakdown visualization")
            return None

        # Remove rows with missing container values
        df = data.dropna(subset=[container_column])

        # Get container type counts
        container_counts = df[container_column].value_counts()

        # Filter to containers with at least min_count
        common_containers = container_counts[container_counts >= min_count].index.tolist()

        # Add an "Other" category for less common containers
        df[f"{container_column}_grouped"] = df[container_column].apply(
            lambda x: x if x in common_containers else "Other"
        )

        # Group by container type and breeding status
        container_breeding = (
            df.groupby([f"{container_column}_grouped", breeding_column])
            .size()
            .unstack(fill_value=0)
        )

        # Ensure we have Yes and No columns
        if breeding_yes_value not in container_breeding.columns:
            container_breeding[breeding_yes_value] = 0
        if "No" not in container_breeding.columns:
            container_breeding["No"] = 0

        # Add a total column and sort
        container_breeding["Total"] = container_breeding.sum(axis=1)
        container_breeding = container_breeding.sort_values("Total", ascending=False)

        # Calculate percentage of breeding spots
        container_breeding["Yes_Pct"] = (
            container_breeding[breeding_yes_value] / container_breeding["Total"] * 100
        ).round(1)

        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Prepare data for stacked bar chart
        containers = container_breeding.index
        yes_counts = container_breeding[breeding_yes_value]
        no_counts = container_breeding["No"]
        yes_pct = container_breeding["Yes_Pct"]

        # Create horizontal stacked bars
        ax.barh(
            containers,
            yes_counts,
            color=self.colors["accent"],
            label=f"Breeding Spot ({breeding_yes_value})",
        )
        ax.barh(
            containers,
            no_counts,
            left=yes_counts,
            color=self.colors["neutral"],
            label="No Breeding Spot",
        )

        # Add count and percentage labels
        for i, container in enumerate(containers):
            # Yes count label
            if yes_counts[container] > 0:
                ax.text(
                    yes_counts[container] / 2,
                    i,
                    f"{int(yes_counts[container])}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=10,
                )

            # No count label
            if no_counts[container] > 0:
                ax.text(
                    yes_counts[container] + no_counts[container] / 2,
                    i,
                    f"{int(no_counts[container])}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=10,
                )

            # Percentage label
            ax.text(
                ax.get_xlim()[1] * 0.95,
                i,
                f"{yes_pct[container]:.1f}%",
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=self.colors["main"],
            )

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Number of Images", fontsize=13, fontweight="bold")
        ax.set_ylabel("Container Type", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=11)

        # Add legend
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fontsize=11
        )

        # Add explanatory text
        plt.figtext(
            0.01,
            0.01,
            "Percentage values show proportion of containers identified as breeding spots.",
            ha="left",
            fontsize=10,
            fontstyle="italic",
        )

        plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_ward_summary(
        self,
        data: pd.DataFrame,
        ward_column: str,
        quality_columns: list[str],
        breeding_column: str | None = None,
        breeding_yes_value: str | None = "Yes",
        title: str = "Ward-Level Quality Summary",
        filename: str = "ward_summary.png",
        max_wards: int = 15,
    ) -> plt.Figure:
        """
        Create a ward-level summary visualization showing image quality and breeding spots.

        Args:
            data (pd.DataFrame): DataFrame with ward and quality data
            ward_column (str): Column name for ward identifier
            quality_columns (List[str]): Columns indicating quality issues (e.g., ['is_dark', 'is_blurry'])
            breeding_column (str, optional): Column name for breeding spot indicator
            breeding_yes_value (str, optional): Value that indicates a positive breeding spot
            title (str): Chart title
            filename (str): Output filename
            max_wards (int): Maximum number of wards to show

        Returns:
            plt.Figure: The generated figure
        """
        if data.empty or ward_column not in data.columns:
            print("Error: Missing required ward column for ward summary visualization")
            return None

        # Make sure quality columns exist
        existing_columns = [col for col in quality_columns if col in data.columns]
        if not existing_columns:
            print("Error: No quality columns found in data")
            return None

        # Calculate ward stats
        ward_stats = data.groupby(ward_column).agg(
            {**{col: "sum" for col in existing_columns}, ward_column: "count"}
        )

        # Rename count column to 'total'
        ward_stats = ward_stats.rename(columns={ward_column: "total"})

        # Add breeding spot percentage if column exists
        if breeding_column and breeding_column in data.columns:
            # Count breeding spots by ward
            breeding_counts = (
                data[data[breeding_column] == breeding_yes_value].groupby(ward_column).size()
            )
            ward_stats["breeding_spots"] = breeding_counts
            ward_stats["breeding_spots"] = ward_stats["breeding_spots"].fillna(0).astype(int)
            ward_stats["breeding_pct"] = (
                ward_stats["breeding_spots"] / ward_stats["total"] * 100
            ).round(1)

        # Calculate quality issue percentages
        for col in existing_columns:
            ward_stats[f"{col}_pct"] = (ward_stats[col] / ward_stats["total"] * 100).round(1)

        # Calculate combined quality issues (images with any issue)
        if len(existing_columns) > 1:
            temp_df = data[data[ward_column].notna()].copy()

            # Add a column indicating if the image has any quality issue
            temp_df["has_issue"] = temp_df[existing_columns].any(axis=1)

            # Count images with issues by ward
            issue_counts = temp_df[temp_df["has_issue"]].groupby(ward_column).size()
            ward_stats["any_issue"] = issue_counts
            ward_stats["any_issue"] = ward_stats["any_issue"].fillna(0).astype(int)
            ward_stats["any_issue_pct"] = (
                ward_stats["any_issue"] / ward_stats["total"] * 100
            ).round(1)

        # Sort by total images and limit to max_wards
        ward_stats = ward_stats.sort_values("total", ascending=False)
        if len(ward_stats) > max_wards:
            ward_stats = ward_stats.head(max_wards)

        # Create figure - use a wider figure for more wards
        fig_width = min(self.fig_width + len(ward_stats) // 5, 20)
        fig, ax = plt.subplots(figsize=(fig_width, self.fig_height))

        # Prepare data for plotting
        wards = ward_stats.index
        totals = ward_stats["total"]

        # Set up x locations
        x = np.arange(len(wards))

        # Plot total images as bars
        bars = ax.bar(
            x, totals, width=0.7, color=self.colors["main"], alpha=0.7, label="Total Images"
        )

        # Add count labels to bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Create a secondary y-axis for percentages
        ax2 = ax.twinx()

        # Plot breeding percentage if available
        if "breeding_pct" in ward_stats.columns:
            ax2.plot(
                x,
                ward_stats["breeding_pct"],
                "o-",
                color=self.colors["accent"],
                markersize=8,
                linewidth=2.5,
                label="Breeding Spot %",
            )

        # Plot overall quality issue percentage if available
        if "any_issue_pct" in ward_stats.columns:
            ax2.plot(
                x,
                ward_stats["any_issue_pct"],
                "s-",
                color=self.colors["blurry"],
                markersize=8,
                linewidth=2.5,
                label="Quality Issues %",
            )

        # Plot individual quality issue percentages if requested
        # This could make the chart too busy, so commented out by default
        """
        markers = ['v', '^', '<', '>', 'p', '*']
        for i, col in enumerate(existing_columns):
            marker = markers[i % len(markers)]
            ax2.plot(
                x,
                ward_stats[f'{col}_pct'],
                marker=marker,
                linestyle='--',
                linewidth=1.5,
                markersize=6,
                alpha=0.7,
                label=f'{col.replace("is_", "").capitalize()} %'
            )
        """

        # Set labels and styling
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel(ward_column, fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Images", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Percentage (%)", fontsize=13, fontweight="bold")

        # Set y-axis limits for percentage
        ax2.set_ylim(0, min(100, max(ward_stats.filter(like="_pct").max().max() * 1.2, 10)))

        ax.set_xticks(x)
        ax.set_xticklabels(wards, rotation=45, ha="right")

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(lines1 + lines2)),
            frameon=True,
            fontsize=11,
        )

        # Add explanatory text
        plt.figtext(
            0.01,
            0.01,
            "Chart shows total images per ward and key quality/content metrics.",
            ha="left",
            fontsize=10,
            fontstyle="italic",
        )

        plt.tight_layout(rect=[0, 0.08, 0.98, 0.98])

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_quality_distribution(
        self,
        data: pd.DataFrame,
        quality_columns: list[str],
        title: str = "Image Quality Distribution",
        filename: str = "quality_distribution.png",
        by_worker: str | None = None,
        top_n_workers: int = 5,
    ) -> plt.Figure:
        """
        Plot distribution of quality issues in the dataset, optionally broken down by worker.

        Args:
            data: DataFrame containing the data
            quality_columns: List of column names representing different quality issues (e.g., is_dark, is_blurry)
            title: Title for the figure
            filename: Output filename
            by_worker: Column name for worker/user identification (if None, no breakdown by worker)
            top_n_workers: Number of top workers to include in breakdown

        Returns:
            matplotlib Figure object
        """
        if data.empty:
            print("Error: Empty data for quality distribution visualization")
            return None

        # Make sure quality columns exist
        existing_columns = [col for col in quality_columns if col in data.columns]
        if not existing_columns:
            print("Error: No quality columns found in data")
            return None

        # Clean up column names for display
        display_names = {col: col.replace("is_", "").capitalize() for col in existing_columns}

        # If grouping by worker
        if by_worker and by_worker in data.columns:
            # Count total images per worker
            worker_counts = data[by_worker].value_counts()

            # Get top N workers by image count
            top_workers = worker_counts.head(top_n_workers).index.tolist()

            # Calculate quality issues per worker (for top workers only)
            worker_quality = {}
            for worker in top_workers:
                worker_data = data[data[by_worker] == worker]
                total = len(worker_data)
                issues = {
                    display_names[col]: int(worker_data[col].sum()) for col in existing_columns
                }
                pct = {
                    display_names[col]: worker_data[col].mean() * 100 for col in existing_columns
                }
                worker_quality[worker] = {"total": total, "issues": issues, "pct": pct}

            # Create figure
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(self.fig_width * 1.5, self.fig_height),
                gridspec_kw={"width_ratios": [2, 3]},
            )

            # Plot 1: Pie chart of overall quality distribution
            quality_counts = {display_names[col]: int(data[col].sum()) for col in existing_columns}

            # Sort by value
            quality_counts = {
                k: v
                for k, v in sorted(quality_counts.items(), key=lambda item: item[1], reverse=True)
            }

            # Calculate total
            total_issues = sum(quality_counts.values())

            if total_issues > 0:
                # Create pie chart with percentage labels
                wedges, texts, autotexts = axes[0].pie(
                    quality_counts.values(),
                    labels=quality_counts.keys(),
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={"linewidth": 1, "edgecolor": "white"},
                    textprops={"fontsize": 10},
                    colors=self.bar_colors[: len(quality_counts)],
                )

                # Style the percentage text
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_fontweight("bold")

                # Add count annotation
                axes[0].annotate(
                    f"Total Issues: {total_issues}",
                    xy=(0, -0.1),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )

                axes[0].set_title("Overall Quality Issues", fontsize=14, fontweight="bold", pad=20)
            else:
                # No quality issues
                axes[0].text(
                    0.5,
                    0.5,
                    "No quality issues found",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=axes[0].transAxes,
                )
                axes[0].set_title("Overall Quality Issues", fontsize=14, fontweight="bold", pad=20)
                axes[0].axis("off")

            # Plot 2: Horizontal stacked bar chart by worker
            if worker_quality:
                # Prepare data
                workers = list(worker_quality.keys())

                issue_types = list(display_names.values())
                issue_data = np.zeros((len(workers), len(issue_types)))

                for i, worker in enumerate(workers):
                    for j, issue in enumerate(issue_types):
                        issue_data[i, j] = worker_quality[worker]["issues"].get(issue, 0)

                # Create stacked horizontal bars
                left = np.zeros(len(workers))
                for j, issue in enumerate(issue_types):
                    axes[1].barh(
                        workers,
                        issue_data[:, j],
                        left=left,
                        color=self.bar_colors[j % len(self.bar_colors)],
                        alpha=0.8,
                        label=issue,
                    )

                    # Update left for next stack
                    left += issue_data[:, j]

                # Add percentage annotations
                for i, worker in enumerate(workers):
                    total = worker_quality[worker]["total"]

                    # Calculate total percentage of images with any issue
                    all_issues = sum(worker_quality[worker]["issues"].values())
                    issue_pct = min(100, all_issues / total * 100) if total > 0 else 0

                    axes[1].text(
                        axes[1].get_xlim()[1] * 0.98,
                        i,
                        f"{issue_pct:.1f}%",
                        ha="right",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color=self.colors["main"],
                    )

                # Styling
                axes[1].set_title(
                    f"Quality Issues by {by_worker}", fontsize=14, fontweight="bold", pad=20
                )
                axes[1].set_xlabel("Number of Images with Issues", fontsize=12, fontweight="bold")
                axes[1].set_ylabel(by_worker, fontsize=12, fontweight="bold")
                axes[1].spines["top"].set_visible(False)
                axes[1].spines["right"].set_visible(False)

                # Add legend
                axes[1].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    ncol=min(3, len(issue_types)),
                    frameon=True,
                    fontsize=10,
                )
            else:
                # No worker data
                axes[1].text(
                    0.5,
                    0.5,
                    f"No data for {by_worker}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=axes[1].transAxes,
                )
                axes[1].axis("off")
        else:
            # Just create a single enhanced plot of quality distribution
            fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

            # Get counts for each quality issue
            quality_counts = {display_names[col]: int(data[col].sum()) for col in existing_columns}

            # Sort by value
            quality_counts = {
                k: v
                for k, v in sorted(quality_counts.items(), key=lambda item: item[1], reverse=True)
            }

            # Calculate percentages
            total_images = len(data)
            quality_pct = {k: v / total_images * 100 for k, v in quality_counts.items()}

            # Create bar chart
            bars = ax.bar(
                quality_counts.keys(),
                quality_counts.values(),
                color=self.bar_colors[: len(quality_counts)],
                edgecolor="white",
                linewidth=1.5,
                alpha=0.85,
            )

            # Add count and percentage labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    # Count label
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.5,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=11,
                    )

                    # Percentage label (inside bar if tall enough)
                    issue_name = bar.get_x() + bar.get_width() / 2.0
                    issue_idx = list(quality_counts.keys()).index(
                        ax.get_xticklabels()[int(issue_name)].get_text()
                    )
                    issue_key = list(quality_counts.keys())[issue_idx]

                    pct = quality_pct[issue_key]

                    if height > total_images * 0.05:  # Only show if bar is tall enough
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height / 2,
                            f"{pct:.1f}%",
                            ha="center",
                            va="center",
                            color="white",
                            fontweight="bold",
                            fontsize=10,
                        )

            # Add "Good Images" bar
            # Calculate number of images with no quality issues
            if len(existing_columns) > 1:
                no_issues_mask = ~data[existing_columns].any(axis=1)
                good_images = no_issues_mask.sum()

                # Add as the last bar
                good_bar = ax.bar(
                    ["Good (No Issues)"],
                    [good_images],
                    color=self.colors["good"],
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.85,
                )

                # Add labels
                for bar in good_bar:
                    height = bar.get_height()
                    if height > 0:
                        # Count label
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.5,
                            f"{int(height)}",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                            fontsize=11,
                        )

                        # Percentage label
                        if height > total_images * 0.05:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height / 2,
                                f"{height / total_images * 100:.1f}%",
                                ha="center",
                                va="center",
                                color="white",
                                fontweight="bold",
                                fontsize=10,
                            )

            # Styling
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
            ax.set_ylabel("Number of Images", fontsize=13, fontweight="bold")
            ax.set_xlabel("Quality Issue Type", fontsize=13, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=11)

            # Add info about total images
            plt.figtext(
                0.5,
                0.01,
                f"Total Images Analyzed: {total_images}",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )

        plt.tight_layout(rect=[0, 0.05, 0.98, 0.98])

        # Save figure
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_time_vs_distance(
        self,
        data: pd.DataFrame,
        time_column: str,
        lat_column: str,
        lon_column: str,
        worker_column: str | None = None,
        title: str = "Time vs Distance Analysis",
        filename: str = "time_vs_distance.png",
    ) -> plt.Figure:
        """
        Analyze and visualize the time spent vs distance traveled between submissions.

        Args:
            data: DataFrame containing the data
            time_column: Column name for timestamp
            lat_column: Column name for latitude
            lon_column: Column name for longitude
            worker_column: Column name for worker identification (for coloring)
            title: Title for the figure
            filename: Output filename

        Returns:
            matplotlib Figure object
        """
        # Ensure we have necessary data
        if data.empty or not all(
            col in data.columns for col in [time_column, lat_column, lon_column]
        ):
            print("Missing required columns for time vs distance analysis.")
            return None

        # Make a copy to avoid warnings
        df = data.copy()

        # Convert time column to datetime if needed
        if df[time_column].dtype != "datetime64[ns]":
            try:
                df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            except Exception as e:
                print(f"Error converting time column to datetime: {e}")
                return None

        # Drop rows with NaN in required columns
        df = df.dropna(subset=[time_column, lat_column, lon_column])

        if len(df) < 2:
            print("Not enough valid data points for time vs distance analysis.")
            return None

        # Sort data by timestamp
        df = df.sort_values(by=time_column)

        # Add worker colors if worker column is provided
        worker_colors = None
        if worker_column and worker_column in df.columns:
            workers = df[worker_column].unique()
            colormap = plt.cm.get_cmap("tab10", len(workers))
            worker_colors = {worker: colormap(i) for i, worker in enumerate(workers)}

        # Calculate time differences and distances between consecutive submissions
        time_diffs = []
        distances = []
        workers = []

        # Group by worker if worker column provided, otherwise process entire dataset
        if worker_column and worker_column in df.columns:
            for worker, group in df.groupby(worker_column):
                if len(group) < 2:
                    continue

                group = group.sort_values(by=time_column)

                for i in range(len(group) - 1):
                    # Calculate time difference in hours
                    time_diff = (
                        group.iloc[i + 1][time_column] - group.iloc[i][time_column]
                    ).total_seconds() / 3600

                    # Skip if time difference is negative (data error) or too large (e.g., different days)
                    if time_diff < 0 or time_diff > 24:
                        continue

                    # Calculate distance in kilometers using Haversine formula
                    lat1, lon1 = group.iloc[i][lat_column], group.iloc[i][lon_column]
                    lat2, lon2 = group.iloc[i + 1][lat_column], group.iloc[i + 1][lon_column]

                    # Validate coordinates
                    try:
                        # Convert to float first
                        lat1, lon1 = float(lat1), float(lon1)
                        lat2, lon2 = float(lat2), float(lon2)

                        if (
                            not (-90 <= lat1 <= 90)
                            or not (-180 <= lon1 <= 180)
                            or not (-90 <= lat2 <= 90)
                            or not (-180 <= lon2 <= 180)
                        ):
                            continue

                        distance = self._haversine_distance(lat1, lon1, lat2, lon2)

                        # Add point if distance is reasonable (less than 50km)
                        if distance < 50:
                            time_diffs.append(time_diff)
                            distances.append(distance)
                            workers.append(worker)
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        continue
        else:
            # Process entire dataset without worker grouping
            for i in range(len(df) - 1):
                # Calculate time difference in hours
                time_diff = (
                    df.iloc[i + 1][time_column] - df.iloc[i][time_column]
                ).total_seconds() / 3600

                # Skip if time difference is negative or too large
                if time_diff < 0 or time_diff > 24:
                    continue

                # Calculate distance in kilometers using Haversine formula
                lat1, lon1 = df.iloc[i][lat_column], df.iloc[i][lon_column]
                lat2, lon2 = df.iloc[i + 1][lat_column], df.iloc[i + 1][lon_column]

                # Validate coordinates
                try:
                    # Convert to float first
                    lat1, lon1 = float(lat1), float(lon1)
                    lat2, lon2 = float(lat2), float(lon2)

                    if (
                        not (-90 <= lat1 <= 90)
                        or not (-180 <= lon1 <= 180)
                        or not (-90 <= lat2 <= 90)
                        or not (-180 <= lon2 <= 180)
                    ):
                        continue

                    distance = self._haversine_distance(lat1, lon1, lat2, lon2)

                    # Add point if distance is reasonable (less than 50km)
                    if distance < 50:
                        time_diffs.append(time_diff)
                        distances.append(distance)
                        workers.append("All Workers")
                except (ValueError, TypeError):
                    # Skip if conversion fails
                    continue

        # Create the plot if we have data
        if not time_diffs:
            print("No valid time-distance pairs found for analysis.")
            return None

        plt.figure(figsize=(self.fig_width, self.fig_height))

        # Create scatter plot
        if worker_column and worker_column in df.columns and worker_colors:
            # Scatter plot with worker colors
            unique_workers = set(workers)
            for worker in unique_workers:
                worker_indices = [i for i, w in enumerate(workers) if w == worker]
                plt.scatter(
                    [time_diffs[i] for i in worker_indices],
                    [distances[i] for i in worker_indices],
                    color=worker_colors.get(worker),
                    label=worker,
                    alpha=0.7,
                    edgecolors="k",
                    linewidths=0.5,
                )
            plt.legend(title=worker_column.capitalize(), loc="upper left", bbox_to_anchor=(1, 1))
        else:
            # Simple scatter plot
            plt.scatter(time_diffs, distances, alpha=0.7, edgecolors="k", linewidths=0.5)

        # Add trend line
        if len(time_diffs) > 1:
            try:
                z = np.polyfit(time_diffs, distances, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(time_diffs), max(time_diffs), 100)
                plt.plot(
                    x_trend,
                    p(x_trend),
                    "r--",
                    linewidth=1,
                    label=f"Trend: y = {z[0]:.2f}x + {z[1]:.2f}",
                )
                plt.legend(loc="upper left")
            except Exception as e:
                print(f"Could not compute trend line: {e}")

        # Add labels and title
        plt.xlabel("Time Between Submissions (hours)")
        plt.ylabel("Distance Between Submissions (km)")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add statistics as text
        avg_time = np.mean(time_diffs)
        avg_dist = np.mean(distances)
        speed = avg_dist / max(avg_time, 0.01)  # km/h, avoid division by zero

        stats_text = (
            f"Avg Time: {avg_time:.2f} hours\n"
            f"Avg Distance: {avg_dist:.2f} km\n"
            f"Avg Speed: {speed:.2f} km/h\n"
            f"Data Points: {len(time_diffs)}"
        )

        plt.annotate(
            stats_text,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
        )

        # Adjust layout and save
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

        print(f"Time vs distance analysis saved to {output_path}")
        # Return the path, not the figure object which is already closed
        return output_path

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    def plot_worker_quality_analysis(
        self,
        data: pd.DataFrame,
        worker_column: str,
        quality_columns: list[str],
        title: str = "Quality Issues by Worker",
        filename: str = "worker_quality_analysis.png",
        max_workers: int = 15,
        sort_by: str = "total_issues",
    ):
        """
        Analyze and visualize quality issues by worker to identify which workers
        take problematic images.

        Args:
            data: DataFrame with worker and quality issue data
            worker_column: Column with worker identifiers
            quality_columns: List of columns indicating quality issues (is_dark, is_blurry, etc.)
            title: Chart title
            filename: Output filename
            max_workers: Maximum number of workers to display
            sort_by: How to sort workers - 'total_issues', 'issue_rate', or a specific issue column

        Returns:
            Path to the saved visualization
        """
        print("Starting worker quality analysis...")

        # Input validation with detailed error messages
        if worker_column not in data.columns:
            error_msg = f"Worker column '{worker_column}' not found in data. Available columns: {data.columns.tolist()}"
            print(error_msg)
            raise ValueError(error_msg)

        # Ensure worker column doesn't have all null values
        if data[worker_column].isna().all():
            error_msg = f"Worker column '{worker_column}' contains only null values"
            print(error_msg)
            raise ValueError(error_msg)

        # Check if quality columns exist
        missing_cols = [col for col in quality_columns if col not in data.columns]
        if missing_cols:
            error_msg = f"Quality columns not found in data: {missing_cols}. Available columns: {data.columns.tolist()}"
            print(error_msg)
            raise ValueError(error_msg)

        print(f"Processing data with {len(data)} records, {data[worker_column].nunique()} workers")

        # Calculate total submissions per worker
        total_by_worker = data.groupby(worker_column).size()
        print(f"Found submissions for {len(total_by_worker)} workers")

        # Calculate quality issues per worker
        issue_data = {}

        for worker, count in total_by_worker.items():
            issue_data[worker] = {"total_submissions": count}
            # Initialize issue counts to 0
            for col in quality_columns:
                issue_data[worker][col.replace("is_", "")] = 0

        # Count each type of issue per worker
        for col in quality_columns:
            try:
                # Get subset of data where this quality issue is True
                issue_subset = data[data[col] == True]
                print(f"Found {len(issue_subset)} records with {col} = True")

                # Count occurrences by worker
                if not issue_subset.empty:
                    issue_counts = issue_subset.groupby(worker_column).size()
                    # Update issue data
                    for worker, count in issue_counts.items():
                        if worker in issue_data:
                            issue_data[worker][col.replace("is_", "")] = count
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                # Continue with next column
                continue

        # Calculate total issues and rates
        for worker in issue_data:
            # Sum all issue counts (excluding total_submissions)
            issue_values = [v for k, v in issue_data[worker].items() if k != "total_submissions"]
            issue_data[worker]["total_issues"] = sum(issue_values)
            # Calculate issue rate
            total_subs = issue_data[worker]["total_submissions"]
            issue_data[worker]["issue_rate"] = (
                issue_data[worker]["total_issues"] / total_subs * 100 if total_subs > 0 else 0
            )

        print(f"Calculated issues for {len(issue_data)} workers")

        # Convert to DataFrame for easier manipulation
        issue_df = pd.DataFrame.from_dict(issue_data, orient="index")

        # Sort the data
        if sort_by == "total_issues":
            issue_df = issue_df.sort_values("total_issues", ascending=False)
        elif sort_by == "issue_rate":
            issue_df = issue_df.sort_values("issue_rate", ascending=False)
        elif sort_by in issue_df.columns:
            issue_df = issue_df.sort_values(sort_by, ascending=False)
        else:
            # Default to sorting by total issues
            issue_df = issue_df.sort_values("total_issues", ascending=False)

        # Limit to max workers
        if max_workers and len(issue_df) > max_workers:
            issue_df = issue_df.head(max_workers)

        # Get issue columns (excluding metadata columns)
        issue_cols = [
            col
            for col in issue_df.columns
            if col not in ["total_submissions", "total_issues", "issue_rate"]
        ]

        if not issue_cols:
            print("WARNING: No quality issue columns available for visualization")
            issue_df["unknown"] = 0
            issue_cols = ["unknown"]

        print(
            f"Creating visualization with {len(issue_df)} workers and {len(issue_cols)} issue types"
        )

        fig = plt.figure(figsize=(self.fig_width * 1.5, self.fig_height * 1.8))

        gs = plt.GridSpec(2, 1, height_ratios=[2, 1.5], hspace=0.3)

        try:
            # Plot 1: Issue counts by worker
            ax1 = plt.subplot(gs[0])

            # Prepare data for stacked bar chart
            indices = np.arange(len(issue_df))
            bar_width = 0.8
            bottom = np.zeros(len(issue_df))

            issue_colors = {}
            for i, col in enumerate(issue_cols):
                # Use custom colors if available
                if col.lower() in self.colors:
                    issue_colors[col] = self.colors[col.lower()]
                else:
                    # Use a color from our palette
                    color_idx = i % len(self.colors)
                    issue_colors[col] = list(self.colors.values())[color_idx]

            # Plot stacked bars for each issue type
            for col in issue_cols:
                ax1.bar(
                    indices,
                    issue_df[col],
                    bottom=bottom,
                    width=bar_width,
                    label=col.capitalize(),
                    color=issue_colors.get(col, "gray"),
                )
                bottom += issue_df[col].values

            # Add total numbers at the top of each bar
            for i, total in enumerate(issue_df["total_issues"]):
                if total > 0:
                    ax1.text(
                        i, total + 0.5, f"{int(total)}", ha="center", va="bottom", fontweight="bold"
                    )

            # Add issue rate as text for each bar
            for i, rate in enumerate(issue_df["issue_rate"]):
                if rate > 0:
                    ax1.text(
                        i,
                        issue_df["total_issues"].iloc[i] / 2,
                        f"{rate:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=10,
                    )

            # Set x-axis labels (worker names)
            ax1.set_xticks(indices)
            ax1.set_xticklabels(issue_df.index, rotation=45, ha="right")

            # Apply styling
            self._apply_chart_style(
                ax1,
                title="Quality Issue Counts by Worker",
                xlabel="Worker",
                ylabel="Number of Quality Issues",
            )

            # Add legend
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Plot 2: Issue percentage by worker
            ax2 = plt.subplot(gs[1])

            # Calculate percentage of each issue type for each worker
            pct_data = issue_df.copy()
            for col in issue_cols:
                pct_data[f"{col}_pct"] = pct_data[col] / pct_data["total_submissions"] * 100

            # Keep only percentage columns for plotting
            pct_cols = [f"{col}_pct" for col in issue_cols]

            # Prepare data for grouped bar chart
            x = np.arange(len(pct_data))
            width = 0.8 / len(pct_cols) if pct_cols else 0.4
            offsets = (
                np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(pct_cols)) if pct_cols else [0]
            )

            # Plot bars for each issue type percentage
            for i, col in enumerate(pct_cols):
                issue_name = col.replace("_pct", "")
                color = issue_colors.get(issue_name, "gray")
                bars = ax2.bar(
                    x + offsets[i],
                    pct_data[col],
                    width,
                    label=issue_name.capitalize(),
                    color=color,
                    alpha=0.8,
                )

                # Add percentage labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 5:  # Only show labels for significant percentages
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.5,
                            f"{height:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=0,
                        )

            # Set x-axis labels (worker names)
            ax2.set_xticks(x)
            ax2.set_xticklabels(pct_data.index, rotation=45, ha="right")

            # Apply styling
            self._apply_chart_style(
                ax2,
                title="Quality Issue Percentages by Worker",
                xlabel="Worker",
                ylabel="Percentage of Submissions with Issues",
            )

            # Add overall title
            plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

            # Save figure
            filepath = self._save_figure(fig, filename)
            print(f"Worker quality analysis visualization saved to: {filepath}")

            return filepath

        except Exception as e:
            import traceback

            print(f"Error creating worker quality visualization: {e}")
            traceback.print_exc()
            # Try to save what we have
            try:
                plt.tight_layout()
                filepath = self._save_figure(fig, filename)
                print(f"Attempted to save partial visualization: {filepath}")
                return filepath
            except:
                # Return a default path even though visualization failed
                return self.output_dir / filename


# Usage example if run as a script
if __name__ == "__main__":
    example_output_dir = Path("example_visualizations")
    viz = VisualizationUtils(output_dir=example_output_dir)

    # Example data
    categories = {
        "Clean": 120,
        "Dark": 45,
        "Blurry": 30,
        "Duplicate": 25,
        "Outlier": 15,
        "Invalid": 10,
    }

    # Example bar chart
    viz.plot_bar_chart(
        data=categories,
        title="Image Quality Distribution",
        xlabel="Category",
        ylabel="Number of Images",
        filename="quality_distribution.png",
        total=sum(categories.values()),
    )

    # Example pie chart
    viz.plot_pie_chart(
        data=categories, title="Image Quality Distribution", filename="quality_pie.png"
    )

    print(f"Example visualizations created in {viz.output_dir}")
