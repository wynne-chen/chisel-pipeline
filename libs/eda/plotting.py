import math
from collections import Counter

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from shapely.geometry import box


class EDAPlotter:
    """Figure-generation helpers extracted from DatasetEDA."""

    def __init__(self, config=None, top_k: int = 10, set_name: str = "", logger=None):
        self.config = config or {}
        self.top_k = top_k
        self.set_name = set_name
        self.logger = logger

    def _warn(self, msg: str):
        if self.logger is not None:
            self.logger.warning(msg)

    def plot_image_size_distribution(self, image_shapes_counts, mask_shapes_counts):
        if image_shapes_counts is None or mask_shapes_counts is None:
            self._warn("No image or mask size data to plot")
            return None
        image_hw = [(s[0], s[1]) for s in image_shapes_counts.keys()]
        mask_hw = [(s[0], s[1]) for s in mask_shapes_counts.keys()]
        image_heights, image_widths = zip(*image_hw)
        mask_heights, mask_widths = zip(*mask_hw)
        image_counts = list(image_shapes_counts.values())
        mask_counts = list(mask_shapes_counts.values())
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].hist(image_heights, bins=20, weights=image_counts, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Image Height Distribution")
        axes[0, 0].set_xlabel("Height (pixels)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 1].hist(image_widths, bins=20, weights=image_counts, alpha=0.7, edgecolor="black")
        axes[0, 1].set_title("Image Width Distribution")
        axes[0, 1].set_xlabel("Width (pixels)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 2].scatter(image_widths, image_heights, s=[c * 10 for c in image_counts], alpha=0.6)
        axes[0, 2].set_title("Image Width vs Height")
        axes[0, 2].set_xlabel("Width (pixels)")
        axes[0, 2].set_ylabel("Height (pixels)")
        axes[1, 0].hist(mask_heights, bins=20, weights=mask_counts, alpha=0.7, edgecolor="black")
        axes[1, 0].set_title("Mask Height Distribution")
        axes[1, 0].set_xlabel("Height (pixels)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 1].hist(mask_widths, bins=20, weights=mask_counts, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Mask Width Distribution")
        axes[1, 1].set_xlabel("Width (pixels)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 2].scatter(mask_widths, mask_heights, s=[c * 10 for c in mask_counts], alpha=0.6)
        axes[1, 2].set_title("Mask Width vs Height")
        axes[1, 2].set_xlabel("Width (pixels)")
        axes[1, 2].set_ylabel("Height (pixels)")
        fig.tight_layout()
        return fig

    
    # --- Spatial Metadata Plots --- #
    def get_layer_source(self, spatial_metadata, spatial_mode):
        if not spatial_metadata:
            self._warn("No spatial metadata to plot")
            return None
        gdfs = []
        if spatial_mode == "dataset_level":
            sm = spatial_metadata[0]
            if sm is None:
                self._warn("Spatial metadata is None. Skipping spatial metadata plot.")
                return None
            if sm.bounds is None:
                self._warn("No bounds found for the spatial metadata. Skipping spatial metadata plot.")
                return None
            source_crs = sm.epsg if sm.epsg is not None else getattr(sm, "crs_wkt", None)
            if not source_crs:
                self._warn("No source CRS found for the spatial metadata. Skipping spatial metadata plot.")
                return None
            try:
                temp_gdf = gpd.GeoDataFrame(geometry=[box(*sm.bounds)], crs=source_crs)
                temp_gdf = temp_gdf.to_crs(epsg=4326)
                gdfs.append(temp_gdf)
            except Exception:
                if self.logger is not None:
                    self.logger.exception("Failed to build/reproject dataset level bbox for %s", getattr(sm, "source_path", "unknown"))
        else:
            for sm in spatial_metadata:
                if sm.bounds is None:
                    continue
                source_crs = f"EPSG:{sm.epsg}" if sm.epsg is not None else getattr(sm, "crs_wkt", None)
                if not source_crs:
                    continue
                try:
                    temp_gdf = gpd.GeoDataFrame(geometry=[box(*sm.bounds)], crs=source_crs)
                    temp_gdf = temp_gdf.to_crs(epsg=4326)
                    gdfs.append(temp_gdf)
                except Exception:
                    if self.logger is not None:
                        self.logger.exception("Failed to build/reproject image level bbox for %s", getattr(sm, "source_path", "unknown"))
        if not gdfs:
            self._warn("No valid spatial layers were created. All records were missing/invalid CRS (EPSG and WKT). Skipping spatial metadata plot.")
            return None
        final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
        union_geom = final_gdf.geometry.union_all()
        centroid = union_geom.centroid
        center_lat = centroid.y
        center_lon = centroid.x
        minx, miny, maxx, maxy = union_geom.bounds
        return final_gdf, center_lat, center_lon, minx, miny, maxx, maxy

    def calc_zoom_for_map(self, minx, miny, maxx, maxy):
        width_y = max(maxy - miny, 1e-9)
        width_x = max(maxx - minx, 1e-9)
        mid_lat = (miny + maxy) / 2.0
        lon_diff_adj = width_x * math.cos(math.radians(mid_lat))
        max_diff = max(width_y, lon_diff_adj)
        zoom = math.log2(360.0 / max_diff)
        return max(0.0, min(zoom, 22))

    def get_bounds_area(self, minx, miny, maxx, maxy):
        earth_radius_km = 6371
        min_lat, max_lat = math.radians(miny), math.radians(maxy)
        min_lon, max_lon = math.radians(minx), math.radians(maxx)
        return earth_radius_km**2 * abs(math.sin(max_lat) - math.sin(min_lat)) * abs(max_lon - min_lon)

    def pad_bounds_to_aspect(self, minx, miny, maxx, maxy, target_aspect):
        width = max(maxx - minx, 1e-12)
        height = max(maxy - miny, 1e-12)
        current = width / height
        if current < target_aspect:
            pad = ((height * target_aspect) - width) / 2
            minx -= pad
            maxx += pad
        elif current > target_aspect:
            pad = ((width / target_aspect) - height) / 2
            miny -= pad
            maxy += pad
        return minx, miny, maxx, maxy

    def plot_spatial_metadata(self, final_gdf, center_lat, center_lon, zoom_level, bounds_area, bounds=None):
        if bounds_area is None:
            self.logger.warning("Bounding boxes have no area; skipping map plot.")
            return None
        if bounds_area > self.config.get("max_bounds_area", 500_000):
            self.logger.warning("Bounds are too large to plot")
            return None
        fig = go.Figure(go.Scattermap())
        fig.update_layout(
            map={
                "style": "carto-darkmatter",
                "center": {"lon": center_lon, "lat": center_lat},
                "zoom": zoom_level,
                "layers": [{
                    "source": final_gdf.__geo_interface__,
                    "type": "fill",
                    "below": "traces",
                    "color": "red",
                    "opacity": 0.5,
                }]
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0}
        )
        if bounds is not None:
            fig.update_layout(map={"bounds": {"west": bounds[0], "east": bounds[2], "south": bounds[1], "north": bounds[3]}})
        return fig

    def plot_country_distribution(self, spatial_metadata):
        if not spatial_metadata:
            self._warn("No spatial metadata to plot")
            return None
        country_counts = Counter([sm.country for sm in spatial_metadata if sm.country is not None])
        top_countries = country_counts.most_common(self.top_k)
        country_names = [country for country, _ in top_countries]
        counts = [count for _, count in top_countries]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(country_names, counts)
        ax.set_yticks(list(range(len(country_names))))
        ax.set_yticklabels(country_names)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Country")
        ax.set_title(f"Top {self.top_k} Countries by Frequency")
        fig.tight_layout()
        return fig

    
    # --- Image Analysis Plots --- #
    def plot_colour_distribution_analysis(self, colour_data):
        if not colour_data or "unique_colours" not in colour_data or not colour_data["unique_colours"]:
            self._warn("No colour distribution data to plot")
            return []
        top_colours = colour_data["unique_colours"][: self.top_k]
        n = len(top_colours)
        colours = [c for c, _ in top_colours]
        counts = [count for _, count in top_colours]
        percentages = [colour_data["colour_distribution_percentages"].get(c, 0.0) for c in colours]
        rgb_colours = [(c[2] / 255, c[1] / 255, c[0] / 255) for c in colours]
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        bars = ax_bar.bar(range(n), counts, color=rgb_colours, edgecolor="black")
        ax_bar.set_xticks(range(n))
        ax_bar.set_xticklabels([f"BGR({int(c[0])}, {int(c[1])}, {int(c[2])})" for c in colours], rotation=90)
        ax_bar.set_xlabel("Colour (BGR)")
        ax_bar.set_ylabel("Pixel Count")
        for idx, bar in enumerate(bars):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{percentages[idx]:.2f}%", ha="center", va="bottom", fontsize=8)
        fig_bar.suptitle(f"Top {n} Most Frequent Colours", fontsize=16)
        fig_bar.tight_layout()
        cols = min(5, n)
        rows = int(np.ceil(n / cols))
        fig_swatches, ax_swatches = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        ax_swatches = np.atleast_1d(ax_swatches).ravel()
        for i in range(rows * cols):
            ax_i = ax_swatches[i]
            if i < n:
                rgb = rgb_colours[i]
                ax_i.imshow(np.ones((20, 20, 3), dtype=np.float32) * rgb)
                c = colours[i]
                ax_i.set_title(f"BGR({int(c[0])}, {int(c[1])}, {int(c[2])})", fontsize=10)
            ax_i.axis("off")
        fig_swatches.suptitle("Colour Swatches", fontsize=16)
        fig_swatches.tight_layout()
        return [fig_bar, fig_swatches]

    def plot_channels_histogram(self, hist_data):
        if not hist_data:
            self._warn("No histogram data to plot")
            return None
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Comprehensive Intensity Distribution Analysis", fontsize=16)
        if "50_bins_intensity" in hist_data:
            counts, bins = hist_data["50_bins_intensity"]
            axes[0, 0].hist(bins[:-1], bins, weights=counts, alpha=0.7, color="gray", edgecolor="black")
            axes[0, 0].set_title("Overall Intensity Distribution (50 bins)")
            axes[0, 0].set_xlabel("Pixel Intensity")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].grid(True, alpha=0.3)
            if "mean_intensity" in hist_data:
                axes[0, 0].axvline(
                    hist_data["mean_intensity"],
                    color="red",
                    linestyle="--",
                    label=f"Mean: {hist_data['mean_intensity']:.2f}",
                )
                axes[0, 0].legend()
        colours = ["blue", "green", "red"]
        colour_names = ["B", "G", "R"]
        for i, (colour, colour_name) in enumerate(zip(colours, colour_names)):
            key = f"{colour_name}_50_bins_intensity"
            if key in hist_data:
                counts, bins = hist_data[key]
                row, col = (0, i + 1) if i == 0 else (1, i - 1)
                axes[row, col].hist(bins[:-1], bins, weights=counts, alpha=0.7, color=colour, edgecolor="black")
                axes[row, col].set_title(f"{colour.capitalize()} Channel Distribution")
                axes[row, col].set_xlabel("Pixel Intensity")
                axes[row, col].set_ylabel("Frequency")
                axes[row, col].grid(True, alpha=0.3)
                mean_key = f"{colour_name}_mean_intensity"
                if mean_key in hist_data:
                    mean_val = hist_data[mean_key]
                    axes[row, col].axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
                    axes[row, col].legend()
        ax_stats = axes[0, 2]
        ax_stats.axis("off")
        stats_text = "Statistical Summary:\n\n"
        stats_text += "Overall Statistics:\n"
        stats_text += f"Mean: {hist_data.get('mean_intensity', np.nan):.2f}\n"
        stats_text += f"Std: {hist_data.get('std_intensity', np.nan):.2f}\n"
        stats_text += f"Range: [{hist_data.get('min_intensity', 'N/A')}, {hist_data.get('max_intensity', 'N/A')}]\n\n"
        for colour_name in ["B", "G", "R"]:
            if f"{colour_name}_mean_intensity" in hist_data:
                stats_text += f"\n{colour_name} Channel:\nMean: {hist_data[f'{colour_name}_mean_intensity']:.2f}\nStd: {hist_data[f'{colour_name}_std_intensity']:.2f}\n"
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace")
        axes[1, 2].axis("off")
        plt.tight_layout()
        return plt.gcf()

    def plot_vegetation_water_indices(self, vegetation_water_indices):
        '''
        Plots a grid of vegetation and water indices, followed by summary histogram plots.

        Args:
            vegetation_water_indices: 
            dict: {
                "histograms": dict[str, tuple[np.ndarray, np.ndarray]],
                    # index_name -> (counts, bin_edges)
                    # counts shape: (bins,), dtype int64
                    # bin_edges shape: (bins + 1,), dtype float32

                "summary": dict[str, dict[str, float | int | np.ndarray]],
                    # index_name -> {
                    #   "mean": float,
                    #   "std": float,
                    #   "min": float,
                    #   "max": float,
                    #   "count": int,
                    #   "percentiles": np.ndarray  # from _hist_percentiles(...), shape (5,)
                    # }

                "sample_maps": dict[str, list[np.ndarray]],
                    # index_name -> up to n_display 2D index arrays (one per selected image)
            }

        '''

        if not vegetation_water_indices:
            self.logger.warning("No vegetation or water indices to plot.")
            return []
        
        histograms = vegetation_water_indices["histograms"]
        summary = vegetation_water_indices["summary"]
        sample_maps = vegetation_water_indices["sample_maps"]

        if not histograms and not sample_maps:
            self.logger.warning("No histograms or sample maps to plot.")
            return []
        
        index_names = [name for name in histograms.keys()]

        figs = []
        for name in index_names:
            
            maps = sample_maps.get(name, [])
            hist_tuple = histograms.get(name, None)
            stats = summary.get(name, {})

            n_maps = len(maps)
            if n_maps == 0:
                n_maps = 1  

            fig = plt.figure(figsize=(20, 9))
            outer = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.15)

            top = outer[0].subgridspec(1, n_maps, wspace=0.08)

            cmap = "RdYlGn" if name not in {"bri", "bgi"} else "Blues"
            top_axes = []
            mappable = None

            if len(maps) == 0:
                ax = fig.add_subplot(top[0, 0])
                ax.text(0.5, 0.5, f"No sample maps available for {name.upper()}",
                        ha="center", va="center", fontsize=13, fontweight="bold")
                ax.axis("off")
            else:
                if name == "exg":
                    finite_all = np.concatenate([m[np.isfinite(m)] for m in maps if np.isfinite(m).any()])
                    if finite_all.size:
                        vmin, vmax = np.percentile(finite_all, [1, 99])
                        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                            vmin, vmax = float(finite_all.min()), float(finite_all.max())
                    else:
                        vmin, vmax = -1.0, 1.0
                else:
                    vmin, vmax = -1.0, 1.0

                for j, arr in enumerate(maps):
                    ax = fig.add_subplot(top[0, j])
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
                        mappable = im
                    else:
                        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", fontsize=10)
                    ax.set_title(f"Sample {j+1}", fontsize=11)
                    ax.axis("off")
                    top_axes.append(ax)

                if mappable is not None and top_axes:
                    cbar = fig.colorbar(mappable, ax=top_axes, location="right", fraction=0.02, pad=0.01)
                    cbar.set_label(f"{name.upper()} value")

            bottom = outer[1].subgridspec(1, 2, width_ratios=[3.5, 1.5], wspace=0.12)

            ax_hist = fig.add_subplot(bottom[0, 0])
            if hist_tuple is not None:
                counts, bins = hist_tuple
                ax_hist.hist(bins[:-1], bins=bins, weights=counts, edgecolor="black", alpha=0.75)
                ax_hist.set_xlabel("Index value")
                ax_hist.set_ylabel("Frequency")
                ax_hist.set_title(f"{name.upper()} Distribution")
                ax_hist.grid(alpha=0.25)
            else:
                ax_hist.text(0.5, 0.5, "No histogram data", ha="center", va="center")
                ax_hist.set_axis_off()

            ax_text = fig.add_subplot(bottom[0, 1])
            ax_text.axis("off")
            pct = stats.get("percentiles", None)
            pct_txt = "N/A"
            if pct is not None and len(pct) >= 5:
                pct_txt = ", ".join([f"{v:.3f}" for v in pct[:5]])

            text = (
                f"{name.upper()} summary\n\n"
                f"count: {stats.get('count', 'N/A')}\n"
                f"mean: {stats.get('mean', float('nan')):.4f}\n"
                f"std:  {stats.get('std', float('nan')):.4f}\n"
                f"min:  {stats.get('min', float('nan')):.4f}\n"
                f"max:  {stats.get('max', float('nan')):.4f}\n"
                f"p[1,25,50,75,99]:\n{pct_txt}"
            )
            ax_text.text(0.01, 0.99, text, va="top", ha="left", fontsize=10, family="monospace")

            fig.suptitle(f"Vegetation/Water Proxy Index: {name.upper()}", fontsize=14)
            figs.append(fig)
        
        return figs

    def plot_average_image(self, avg_img):
        if avg_img is None:
            self._warn("No average image to display.")
            return None
        img_rgb = cv2.cvtColor(avg_img, cv2.COLOR_BGR2RGB)
        channels = [
            (img_rgb, None, "RGB"),
            (img_rgb[:, :, 0], "Reds", "Red"),
            (img_rgb[:, :, 1], "Greens", "Green"),
            (img_rgb[:, :, 2], "Blues", "Blue"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle("Average Image", fontsize=16)
        for ax, (data, cmap, title) in zip(axes.flat, channels):
            ax.imshow(data, cmap=cmap)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")
        plt.tight_layout()
        return plt.gcf()

    def plot_intensity_heatmap(self, avg_img):
        if avg_img is None:
            self._warn("No intensity heatmap to display")
            return None
        grey = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.imshow(grey, cmap="gray")
        ax.set_aspect("equal")
        plt.title("Average Intensity Heatmap")
        plt.axis("off")
        plt.tight_layout()
        return plt.gcf()

    def plot_contrast_heatmap(self, contrast_heatmap, window_size=25):
        if contrast_heatmap is None:
            self._warn("No contrast heatmap to display")
            return None
        if np.isnan(contrast_heatmap).any():
            self._warn("NaNs detected in contrast heatmap; rendering may be sparse.")
        n_rows, n_cols = contrast_heatmap.shape
        base_size = 2
        figsize = (max(6, n_cols * base_size), max(6, n_rows * base_size))
        square_flag = n_rows / n_cols == 1
        plt.figure(figsize=figsize)
        sns.heatmap(
            contrast_heatmap,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            cbar=True,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 16},
            square=square_flag,
        )
        plt.title(f"Windowed Local Contrast Heatmap (window={window_size})", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        return plt.gcf()


    # --- Object Statistics Plots --- #
    def plot_object_count_distribution(self, counts, bins=30):
        if counts is None or len(counts) == 0:
            self._warn("No object count data to plot.")
            return None
        counts = np.asarray(counts, dtype=np.int32)
        if counts.size == 0:
            self._warn("No object count data to plot.")
            return None
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(counts, bins=bins, alpha=0.75, edgecolor="black")
        plt.xlabel("Objects per image")
        plt.ylabel("Frequency")
        plt.title("Object Count Distribution")
        plt.subplot(1, 2, 2)
        plt.boxplot(counts, vert=True)
        plt.ylabel("Objects per image")
        plt.title("Object Count Summary")
        plt.tight_layout()
        return plt.gcf()

    def plot_object_area_distribution(self, areas):
        if not areas or not any(areas):
            self._warn("No area data to plot.")
            return None
        all_areas = [a for sublist in areas for a in sublist if a > 0]
        if not all_areas:
            self._warn("No positive object areas to plot.")
            return None
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(all_areas, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Object Area (pixels)")
        plt.ylabel("Frequency")
        plt.title("Object Area Distribution")
        plt.subplot(1, 2, 2)
        plt.hist(all_areas, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Object Area (pixels)")
        plt.ylabel("Frequency")
        plt.title("Object Area Distribution (Log Scale)")
        plt.yscale("log")
        plt.tight_layout()
        return plt.gcf()

    def plot_object_centroids(self, centroids, image_shape):
        if not centroids or not any(centroids) or not image_shape:
            self._warn("No centroid data to plot")
            return None
        all_centroids = [c for sublist in centroids for c in sublist if c is not None]
        if not all_centroids:
            self._warn("No valid centroids to plot.")
            return None
        xs, ys = zip(*all_centroids)
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.scatter(xs, ys, alpha=0.6, s=20)
        shape = None
        if isinstance(image_shape, list) and len(image_shape) > 0:
            shape = image_shape[0]
        elif isinstance(image_shape, tuple) and len(image_shape) >= 2:
            shape = image_shape
        if shape is not None:
            h, w = int(shape[0]), int(shape[1])
            plt.xlim(0, w)
            plt.ylim(h, 0)
            ax.set_aspect("equal")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Object Centroids Distribution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

    def plot_pixel_heatmap(self, heatmap, smooth=True, sigma=1.2):
        if heatmap is None:
            self._warn("No heatmap to display")
            return None
        display = heatmap
        if smooth:
            k = max(3, int(6 * sigma) | 1)
            display = cv2.GaussianBlur(heatmap.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.imshow(display, cmap="hot", interpolation="nearest")
        ax.set_aspect("equal")
        plt.title("Centroid Heatmap")
        plt.axis("off")
        plt.tight_layout()
        cbar = plt.colorbar(label="Centroid density")
        cbar.ax.tick_params(labelsize=9)
        return plt.gcf()

    def plot_centre_bias_distribution(self, centre_bias):
        if not centre_bias:
            self._warn("No centre bias data to plot.")
            return None
        all_centre_bias = [c for sublist in centre_bias for c in sublist if c is not None]
        if not all_centre_bias:
            self._warn("No valid centre bias to plot.")
            return None
        plt.figure(figsize=(10, 10))
        plt.hist(all_centre_bias, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Centre Bias")
        plt.ylabel("Frequency")
        plt.title("Centre Bias Distribution")
        plt.tight_layout()
        return plt.gcf()

    def plot_edge_touching_distribution(self, edge_touching):
        if edge_touching is None or len(edge_touching) == 0:
            self._warn("No edge-touching data to plot.")
            return None
        edge_touching = np.asarray(edge_touching, dtype=np.int32)
        if edge_touching.size == 0:
            self._warn("No edge-touching data to plot.")
            return None
        unique_vals, freqs = np.unique(edge_touching, return_counts=True)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(unique_vals, freqs, edgecolor="black", alpha=0.8)
        plt.xlabel("Edge-touching objects per image")
        plt.ylabel("Frequency")
        plt.title("Edge-Touching Object Distribution")
        plt.subplot(1, 2, 2)
        ratio = (edge_touching > 0).mean() * 100.0
        plt.hist(edge_touching, bins=min(30, max(5, len(unique_vals))), edgecolor="black", alpha=0.75)
        plt.xlabel("Edge-touching objects per image")
        plt.ylabel("Frequency")
        plt.title(f"Histogram (images with edge-touching objects: {ratio:.1f}%)")
        plt.tight_layout()
        return plt.gcf()

    def plot_multiclass_object_summary(self, class_ids, image_counts, object_counts, total_areas, area_ratios):
        labels = [str(c) for c in class_ids]
        x = np.arange(len(labels))
        panels = [
            ("Images containing each class", "Images per class", image_counts),
            ("Total mask objects per class", "Objects per class", object_counts),
            ("Total object area per class", "Area (px)", total_areas),
            ("Mean object area / image area", "Mean ratio", area_ratios),
        ]
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)), sharex=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, ylabel, values) in zip(axes, panels):
            ax.bar(x, values, width=0.6, edgecolor="black", alpha=0.85)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel("Class ID")
        fig.tight_layout()
        return plt.gcf()
