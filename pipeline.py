#--- IMPORTS ---#

# Standard library imports
import os
import logging

# Third-party imports
import numpy as np
import cv2

#--- Config ---#

from config import config, CLASS_ID_MAP

#--- Libraries ---#
from libs.file_handler import FileHandler
from libs.image_processor import ImageProcessor
from libs.annotation_generator import AnnotationGenerator
from libs.spatial_metadata import SpatialMetadataService
from libs.utils import PipelineUtils

#--- Data Structure ---#
'''
Data/ 
 ├── vegetation/ 
 │   ├── dataset_file_directory.csv # this is the mapping file for the images and masks and labels
 │   ├── data.yaml
 │   ├── images/ 
 │   │   ├── image_0000001.jpg    # this is the original image
 │   │   └── etc...
 │   ├── masks/ 
 │   │   ├── image_0000001_m.png  # this is the image mask for image_0000001.jpg
 │   │   └── etc...
 │   └── labels/ 
 │       ├── image_0000001.txt    # this is the yolo text file for image_0000001.jpg
 │       └── etc...
 ├── buildings/
 │   ├── dataset_file_directory.csv
 │   ├── data.yaml
 │   └── etc/
 ├── etc/
'''

# Ensure output directory exists before logging setup
quarantine_dir = config['quarantine_dir']
os.makedirs(quarantine_dir, exist_ok=True)

# Logging setup
log_file = os.path.join(quarantine_dir, 'pipeline.log')
logging.basicConfig(
    level=getattr(logging, config.get('log_level', 'INFO')),
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

#--- MAIN ---#

# main processing function (to be called from notebook)
def process_dataset_incremental(data_dir, set_name, class_id, config):
    '''
    Incrementally process a dataset: split images/masks, save tiles, annotations, and update CSV.
    User must provide class_id (int) and set_name (str).
    class_name is looked up from CLASS_ID_MAP.
    Optionally, pass custom_mask_filter(mask, class_id, class_colour_map) for dataset-specific mask logic.
    '''
    if class_id not in CLASS_ID_MAP:
        raise ValueError(f"class_id {class_id} not in CLASS_ID_MAP. Available: {list(CLASS_ID_MAP.keys())}")
    class_name = CLASS_ID_MAP[class_id]

    file_handler = FileHandler(config)
    img_processor = ImageProcessor(config)
    spatial_metadata_service = SpatialMetadataService(config)
    ann_generator = AnnotationGenerator(config)
    set_dir = os.path.join(quarantine_dir, set_name)
    utils = PipelineUtils(config=config, output_dir=set_dir)

    class_dir = utils.ensure_class_and_subfolders(class_name)
    images_dir = os.path.join(class_dir, 'images')
    masks_dir = os.path.join(class_dir, 'masks')
    labels_dir = os.path.join(class_dir, 'labels')

    utils.write_yolo_yaml(class_name)

    image_paths, mask_paths = file_handler.load_dataset(data_dir)
    index_width = 7  # Support up to 1,000,000 files
    tile_idx = 0
    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            image = img_processor.load_image(img_path)

            # Conditional normalisation based on image dtype
            if image.dtype == np.uint16:
                logging.info(f"Normalising 16-bit image {os.path.basename(img_path)} before tiling.")
                image = (img_processor.normalise_image(image) * 255).astype(np.uint8)
            elif image.dtype == np.float32 or image.dtype == np.float64:
                if np.max(image) > 1.0 or np.min(image) < 0.0:
                    logging.info(f"Normalising float image {os.path.basename(img_path)} before tiling due to out-of-range values.")
                    image = (img_processor.normalise_image(image) * 255).astype(np.uint8)
                else:
                    logging.info(f"Scaling 0-1 float image {os.path.basename(img_path)} to 0-255 uint8.")
                    image = (image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                # Catch any other unexpected dtypes and convert to uint8
                logging.warning(f"Unexpected image dtype {image.dtype} for {os.path.basename(img_path)}. Converting to uint8.")
                image = image.astype(np.uint8)
            

            raw_mask = img_processor.load_mask(mask_path)
            multiclass = img_processor.is_multiclass(raw_mask)
            mask = img_processor.filter_mask_by_class_id(raw_mask, class_id, set_name)

            # Iterate over image and mask tiles in parallel to ensure correct alignment
            image_tiles = img_processor.split_image_generator(image, img_processor.target_size)
            mask_tiles = img_processor.split_image_generator(mask, img_processor.target_size)
            spatial_metadata = spatial_metadata_service.get_spatial_metadata(img_path)
            
            csv_rows_this_image = []
            ignore_holes = config.get('ignore_holes', False)

            for (tile, y, x), (mask_tile, my, mx) in zip(image_tiles, mask_tiles):
                if (y, x) != (my, mx):
                    raise ValueError(f"Tile offset mismatch: image=({y},{x}), mask=({my},{mx})")

                valid_h = min(img_processor.target_size, image.shape[0] - y)
                valid_w = min(img_processor.target_size, image.shape[1] - x)
                tile_meta = spatial_metadata_service.get_tile_spatial_metadata(
                    img_path,
                    x=x,
                    y=y,
                    valid_w=valid_w,
                    valid_h=valid_h,
                    padded_w=tile.shape[1],
                    padded_h=tile.shape[0],
                )

                # Generate annotation lines for the current tile
                if ignore_holes:
                    polygons = ann_generator.mask_to_polygons(mask_tile)
                else:
                    polygons = ann_generator.mask_to_polygons_with_holes(mask_tile)
                annotation_lines = ann_generator.generate_tile_yolo_annotation(polygons, class_id, tile.shape[1], tile.shape[0])

                # Save image tile, mask tile, and annotation only if annotations exist
                if annotation_lines:
                    logging.info(f"Annotations found for tile {set_name}_{tile_idx:0{index_width}d}. Saving files.")
                    # Save image tile
                    image_tile_name = f"{set_name}_{tile_idx:0{index_width}d}.jpg"
                    image_tile_path = os.path.join(images_dir, image_tile_name)
                    cv2.imwrite(image_tile_path, tile)
                    # Save mask tile
                    mask_tile_name = f"{set_name}_{tile_idx:0{index_width}d}_m.png"
                    mask_tile_path = os.path.join(masks_dir, mask_tile_name)
                    cv2.imwrite(mask_tile_path, mask_tile)
                    # Label file should have the same stem as image, but .txt extension
                    label_tile_name = image_tile_name.replace('.jpg', '.txt')
                    logging.info(f"Attempting to save YOLO annotation for: {label_tile_name} in {labels_dir}")
                    file_handler.save_yolo_annotation(annotation_lines, label_tile_name, labels_dir)
                    deletion_status = 'not_deleted'
                else:
                    logging.info(f"No annotations found for tile {set_name}_{tile_idx:0{index_width}d}. Skipping file saving.")
                    image_tile_name = f"{set_name}_{tile_idx:0{index_width}d}.jpg"
                    mask_tile_name = f"{set_name}_{tile_idx:0{index_width}d}_m.png"
                    label_tile_name = image_tile_name.replace('.jpg', '.txt') # Still generate name for CSV
                    deletion_status = 'deleted'

                # Update CSV
                row = {
                    'output_image': image_tile_name,
                    'output_mask': mask_tile_name,
                    'output_label': label_tile_name,
                    'original_image': os.path.basename(img_path),
                    'original_mask': os.path.basename(mask_path),
                    'category': class_name,
                    'tile_index': tile_idx,
                    'dataset_name': set_name,
                    'deletion_status': deletion_status,
                    'multiclass': multiclass,
                    'src_epsg': spatial_metadata.epsg,
                    'crs_wkt': spatial_metadata.crs_wkt,
                    'projection_format': spatial_metadata.projection_format,
                    'bounds': tile_meta['tile_bounds'],
                    'transform': tile_meta['tile_transform'],
                    'x_cm_per_pixel': tile_meta['tile_x_cm_per_pixel'],
                    'y_cm_per_pixel': tile_meta['tile_y_cm_per_pixel'],
                    'mean_cm_per_pixel': tile_meta['tile_mean_cm_per_pixel'],
                    'num_bands': tile_meta['tile_num_bands'],
                    'centroid_src': tile_meta['tile_centroid_src'],
                    'centroid_wgs84': tile_meta['tile_centroid_wgs84'],
                    'country': tile_meta['tile_country'],
                }
                csv_rows_this_image.append(row)
                tile_idx += 1
            utils.append_csv_rows(class_name, csv_rows_this_image)
            del image
            del mask
            
        except Exception as e:
            print(f"AN UNEXPECTED ERROR OCCURRED: {e}")
            logging.error(f"Error processing {img_path} and {mask_path}: {e}", exc_info=True)
    # Clean up unlabelled images
    utils.clean_unlabelled_images(class_name)
    logging.info(f"Processing complete for set {set_name}, class {class_name}.")


