import logging
from pathlib import Path

import numpy as np
import ast
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.transform import array_bounds
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point
from affine import Affine

from models.metadata import SpatialMetadata


class SpatialMetadataService:
    """
    Service class for extracting geospatial metadata from raster images.
    """

    def __init__(self, config=None):
        self.config = config or {}
        world_shp = Path(__file__).resolve().parent.parent / "data" / "ne_10m_admin_0_countries.shp"
        self.world = gpd.read_file(world_shp)
        self.world_sindex = self.world.sindex

    def _get_country(self, centroid):
        if centroid is None:
            return None
        point = Point(centroid)
        idx = list(self.world_sindex.intersection(point.bounds))
        if not idx:
            return None
        candidates = self.world.iloc[idx]
        hit = candidates[candidates.geometry.covers(point)]
        if hit.empty:
            return None
        return hit.iloc[0]["ADMIN"]

    def _safe_epsg(self, crs):
        """
        Safely extract EPSG code from a CRS object.
        """
        if crs is None:
            return None
        if crs.is_epsg_code:
            return crs.to_epsg()
        try:
            crs_obj = CRS.from_user_input(crs)
            if crs_obj.is_projected:
                code = crs_obj.to_epsg()
                if code is not None:
                    return code
            if crs_obj.is_geographic:
                return crs_obj.to_epsg()
            return crs_obj.to_epsg()
        except Exception:
            logging.exception("_safe_epsg failed for crs=%r", crs)
            return None

    def _centroid_wgs84(self, src, epsg):
        """
        Return centroid in source coordinates and WGS84.
        """
        centroid = (
            (src.bounds.left + src.bounds.right) / 2,
            (src.bounds.top + src.bounds.bottom) / 2,
        )

        if src.crs is None:
            return centroid, None

        try:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(*centroid)
            return centroid, (lon, lat)
        except Exception:
            logging.exception("_centroid_wgs84 failed for src=%r, epsg=%r", src, epsg)
            return centroid, None

    def _cm_per_pixel(self, src, centroid_wgs84):
        """
        Return x_cm_per_pixel, y_cm_per_pixel, mean_cm_per_pixel.
        """
        if src is None or centroid_wgs84 is None:
            return None, None, None
        a = abs(src.transform.a)
        e = abs(src.transform.e)
        try:
            crs_obj = CRS.from_user_input(src.crs)
            is_projected = crs_obj.is_projected
            unit_name = (
                crs_obj.axis_info[0].unit_name.lower()
                if crs_obj.axis_info and crs_obj.axis_info[0].unit_name
                else ""
            )
        except Exception:
            is_projected = False
            unit_name = ""
        if is_projected and "met" in unit_name:
            x_cm = a * 100.0
            y_cm = e * 100.0
            return x_cm, y_cm, (x_cm + y_cm) / 2.0
        lon, lat = centroid_wgs84
        geod = Geod(ellps="WGS84")
        dlon = abs(src.transform.a)
        dlat = abs(src.transform.e)
        _, _, dist_x_m = geod.inv(lon, lat, lon + dlon, lat)
        _, _, dist_y_m = geod.inv(lon, lat, lon, lat + dlat)
        x_cm = dist_x_m * 100.0
        y_cm = dist_y_m * 100.0
        return x_cm, y_cm, (x_cm + y_cm) / 2.0

    def get_spatial_metadata(self, image_path):
        """
        Get spatial metadata from an image.
        NOTE: Country is determined by image centroid.
        """
        with rio.open(image_path) as src:
            epsg = self._safe_epsg(src.crs)
            crs_wkt = src.crs.to_wkt() if src.crs is not None else None
            projection_format = None
            if src.crs is not None:
                try:
                    crs_obj = CRS.from_user_input(src.crs)
                    projection_format = (
                        crs_obj.coordinate_operation.method_name
                        if crs_obj.coordinate_operation
                        else None
                    )
                except Exception:
                    logging.exception(f"Failed CRS parse for {image_path}")
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            transform = tuple(src.transform)[:6]
            centroid_src, centroid_wgs84 = self._centroid_wgs84(src, epsg)
            country = self._get_country(centroid_wgs84)
            x_cm_per_pixel, y_cm_per_pixel, mean_cm_per_pixel = self._cm_per_pixel(src, centroid_wgs84)

            spatial_metadata = SpatialMetadata(
                source_path=image_path,
                epsg=epsg,
                crs_wkt=crs_wkt,
                projection_format=projection_format,
                bounds=bounds,
                transform=transform,
                x_cm_per_pixel=x_cm_per_pixel,
                y_cm_per_pixel=y_cm_per_pixel,
                mean_cm_per_pixel=mean_cm_per_pixel,
                num_bands=src.count,
                centroid_src=centroid_src,
                centroid_wgs84=centroid_wgs84,
                country=country,
            )

        return spatial_metadata

    def get_tile_spatial_metadata(self, image_path, x, y, valid_w, valid_h, padded_w, padded_h):
        with rio.open(image_path) as src:
            parent_transform = src.transform
            tile_transform = parent_transform * Affine.translation(x, y)

            # bounds should use valid size (source-covered area), not padded zeros
            left, bottom, right, top = array_bounds(valid_h, valid_w, tile_transform)
            centroid_src = ((left + right) / 2.0, (bottom + top) / 2.0)

            centroid_wgs84 = None
            if src.crs is not None:
                t = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                centroid_wgs84 = t.transform(*centroid_src)

            # Reuse existing helper for resolution
            x_cm_per_pixel, y_cm_per_pixel, mean_cm_per_pixel = self._cm_per_pixel(src, centroid_wgs84)
            country = self._get_country(centroid_wgs84)

            return {
                "tile_origin_x_px": x,
                "tile_origin_y_px": y,
                "tile_valid_width_px": valid_w,
                "tile_valid_height_px": valid_h,
                "tile_padded_width_px": padded_w,
                "tile_padded_height_px": padded_h,
                "tile_transform": tuple(tile_transform)[:6],
                "tile_bounds": (left, bottom, right, top),
                "tile_centroid_src": centroid_src,
                "tile_centroid_wgs84": centroid_wgs84,
                "tile_x_cm_per_pixel": x_cm_per_pixel,
                "tile_y_cm_per_pixel": y_cm_per_pixel,
                "tile_mean_cm_per_pixel": mean_cm_per_pixel,
                "tile_num_bands": src.count,
                "tile_country": country,
            }

    def _none_if_nan(self, v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return v
    
    def _string_to_tuple(self, v):
        v = self._none_if_nan(v)
        if v is None:
            return None
        if isinstance(v, (tuple, list)):
            return tuple(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            else:
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (tuple, list)):
                        return tuple(parsed)
                    return None
                except Exception:
                    return None
        return None

    def _epsg_string_to_int(self, v):
        v = self._none_if_nan(v)
        if v is None or v=='':
            return None
        if isinstance(v, int):
            return v
        try:
            return int(float(v)) # float() ensures handling of .00 string values like '4326.00'
        except Exception:
            return None
    
    def get_spatial_metadata_from_csv(self, csv_path):
        '''
        Loads spatial metadata from a CSV file.
        '''
        df = pd.read_csv(csv_path)
        df = df[df['deletion_status'] == 'not_deleted']
        
        spatial_metadata_list = []

        for row in df.itertuples(index=False):
            spatial_metadata_list.append(
                SpatialMetadata(
                    source_path=row.output_image,
                    epsg=self._epsg_string_to_int(row.src_epsg),
                    crs_wkt=self._none_if_nan(row.crs_wkt),
                    projection_format=self._none_if_nan(row.projection_format),
                    bounds=self._string_to_tuple(row.bounds),
                    transform=self._string_to_tuple(row.transform),
                    x_cm_per_pixel=self._none_if_nan(row.x_cm_per_pixel),
                    y_cm_per_pixel=self._none_if_nan(row.y_cm_per_pixel),
                    mean_cm_per_pixel=self._none_if_nan(row.mean_cm_per_pixel),
                    num_bands=self._none_if_nan(row.num_bands),
                    centroid_src=self._string_to_tuple(row.centroid_src),
                    centroid_wgs84=self._string_to_tuple(row.centroid_wgs84),
                    country=self._none_if_nan(row.country)
                )
            )
        
        if len(spatial_metadata_list) == 0:
            logging.warning("No spatial metadata found in the CSV file")
            return []
        
        return spatial_metadata_list

