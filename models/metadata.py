#--- IMPORTS ---#
import rasterio as rio
from dataclasses import dataclass
from typing import Tuple

#--- SPATIAL METADATA ---#
@dataclass(frozen=True)
class SpatialMetadata:
    '''
    Dataclass for storing spatial metadata of an image.

    source_path:        Original image source path.
    epsg:               EPSG code of the image.
    crs_wkt:            WKT string of the image CRS.
    projection_format:  projection format of the image.
    bounds:             bounds of the image in the source CRS.
    from GDAL, transform:
                        GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
                        GT(1) w-e pixel resolution / pixel width.
                        GT(2) row rotation (typically zero).
                        GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
                        GT(4) column rotation (typically zero).
                        GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
    x_cm_per_pixel:     x cm per pixel of the image.
    y_cm_per_pixel:     y cm per pixel of the image.
    mean_cm_per_pixel:  mean cm per pixel of the image.
    num_bands:          number of bands in the image.
    centroid_src:       centroid of the image in the source CRS.
    centroid_wgs84:     centroid of the image in WGS84 coordinates.
    country:            country of the image.

    '''
    source_path: (str | None)
    epsg: (int | None)
    crs_wkt: (str | None)
    projection_format: (str | None)
    bounds: (Tuple[float, float, float, float] | None)
    transform: (Tuple[float, float, float, float, float, float] | None)
    x_cm_per_pixel: (float | None)
    y_cm_per_pixel: (float | None)
    mean_cm_per_pixel: (float | None)
    num_bands: (int | None)
    centroid_src: (Tuple[float, float] | None)
    centroid_wgs84: (Tuple[float, float] | None)
    country: (str | None)
    

    
