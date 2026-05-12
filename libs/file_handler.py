#--- IMPORTS ---#
import os
import glob
import csv
from pathlib import Path
import logging
import shutil
import cv2
import numpy as np

from libs.utils import PipelineUtils

#--- FILE HANDLER ---#

class FileHandler:
    '''
    Class for importing, sorting, or writing files.
    '''
    def __init__(self, config=None):
        self.config = config or {}
        self.target_size = self.config.get('target_size', 512)
        self.output_dir = self.config.get('output_dir', os.path.join(os.getcwd(), 'output'))
        self.quarantine_dir = self.config.get('quarantine_dir', os.path.join(os.getcwd(), 'quarantine'))
        self.image_paths = []
        self.mask_paths = []

    def load_dataset(self, data_dir):
        '''
        Load and organise images and masks from directory. Enforces that every image has a corresponding mask and vice versa.
        '''
        self.image_paths.clear()
        self.mask_paths.clear()
        skip_keywords = self.config.get('skip_keywords', ['csv', 'json', 'xml', 'vis', 'md', 'zip', 'pdf', 'txt', 'py', 'yaml'])
        mask_dirs = set(self.config.get('mask_dirs', ['mask', 'masks', 'label', 'labels']))
        mask_prefixes = tuple(self.config.get('mask_prefixes', ['mask_', 'label_']))
        mask_suffixes = tuple(self.config.get('mask_suffixes', ['_mask', '_m.']))
        
        all_filepaths = [f for f in glob.glob(os.path.join(data_dir, '**', '*.*'), recursive=True)]
        
        image_maps = {}
        mask_maps = {}

        for filepath in all_filepaths:
            path = Path(filepath)
            ext = path.suffix.lower()

            if ext.lstrip('.') in skip_keywords:
                continue
            
            stem = path.stem.lower()
            parent_parts = {part.lower() for part in path.parts[:-1]}

            is_mask = (
                any(part in mask_dirs for part in parent_parts)
                or stem.startswith(mask_prefixes)
                or stem.endswith(mask_suffixes)
            )
            key = stem
            for prefix in mask_prefixes:
                if key.startswith(prefix):
                    key = key.removeprefix(prefix)
            for suffix in mask_suffixes:
                if key.endswith(suffix):
                    key = key.removesuffix(suffix)

            target = mask_maps if is_mask else image_maps
            target.setdefault(key, []).append(filepath)
        
        paired_keys = sorted(key for key in (image_maps.keys() & mask_maps.keys()) if len(image_maps[key]) == 1 and len(mask_maps[key]) == 1) # discards if there are multiple/no images that match the same mask or vice versa.

        missing_images = sorted(mask_maps.keys() - image_maps.keys())
        missing_masks = sorted(image_maps.keys() - mask_maps.keys())
        ambiguous_keys = sorted(key for key in (image_maps.keys() & mask_maps.keys()) if len(image_maps[key]) > 1 or len(mask_maps[key]) > 1) # stores the keys that have multiple images or masks that match.

        if missing_images:
            logging.info(f"No image found for mask(s): {missing_images}")
        if missing_masks:
            logging.info(f"No mask found for image(s): {missing_masks}")
        if ambiguous_keys:
            logging.info(f"Ambiguous keys: {ambiguous_keys}")

        self.image_paths = [image_maps[key][0] for key in paired_keys]
        self.mask_paths = [mask_maps[key][0] for key in paired_keys]

        logging.info(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")

        return self.image_paths, self.mask_paths

    def save_processed_mask(self, mask, set_name, tile_idx, output_dir):
        '''
        Save a processed mask in visible format. All masks are single class, black with white masks.
        '''
        mask_save_path = os.path.join(output_dir, f"{set_name}_{tile_idx}_m.png")
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            if np.max(mask) <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        cv2.imwrite(mask_save_path, mask)
        return mask_save_path

    def save_yolo_annotation(self, annotation_lines, label_file_name, output_dir):
        '''
        Save annotation in YOLO format for a single tile.
        '''
        logging.info(f"Inside save_yolo_annotation: label_file_name={label_file_name}, output_dir={output_dir}")
        label_path = os.path.join(output_dir, label_file_name)
        print(label_path)
        with open(label_path, 'w') as f:
            for line in annotation_lines:
                f.write(line + '\n')
        return label_path

    def delete_set_from_class(self, class_name, set_name):
        '''
        Delete all files (images, masks, labels) and CSV records associated with a set_name in a given class_name folder.

        Use case: if a set is no longer desired, or if a mistake was made (wrong class, typo in set_name, etc.)
        '''
        class_dir = os.path.join(self.output_dir, class_name)
        images_dir = os.path.join(class_dir, 'images')
        masks_dir = os.path.join(class_dir, 'masks')
        labels_dir = os.path.join(class_dir, 'labels')
        csv_path = os.path.join(class_dir, 'dataset_file_directory.csv')

        if not os.path.exists(csv_path):
            logging.warning(f"No CSV found at {csv_path} for deletion.")
            return

        # Read all rows and filter out those matching set_name
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            rows = list(reader)

        rows_to_keep = []
        rows_to_delete = []
        for row in rows:
            if row.get('dataset_name') == set_name:
                rows_to_delete.append(row)
            else:
                rows_to_keep.append(row)

        # Delete files for rows to delete
        for row in rows_to_delete:
            image_file = row.get('output_image')
            mask_file = row.get('output_mask')
            label_file = row.get('output_label')
            image_path = os.path.join(images_dir, image_file) if image_file else None
            mask_path = os.path.join(masks_dir, mask_file) if mask_file else None
            label_path = os.path.join(labels_dir, label_file) if label_file else None
            for f in [image_path, mask_path, label_path]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                        logging.info(f"Deleted file: {f}")
                    except Exception as e:
                        logging.error(f"Failed to delete {f}: {e}")

        # Write filtered rows back to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_to_keep:
                writer.writerow(row)
        logging.info(f"Deleted all files and CSV records for set '{set_name}' in class '{class_name}'.")
        
        del rows_to_keep
        del rows_to_delete
        logging.info("rows_to_keep and rows_to_delete have been removed from memory.")


    def get_file_types_in_dataset(self, data_dir, max_examples=3):
        '''
        Returns all file types (extensions) in the dataset folder, with counts and example filenames.
        '''
        file_types = {}
        for root, _, files in os.walk(data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in file_types:
                    file_types[ext] = []
                if len(file_types[ext]) < max_examples:
                    file_types[ext].append(os.path.join(root, f))
        return file_types
    
    def print_file_types_in_dataset(self, data_dir, max_examples=3):
        '''
        Print and returns all file types (extensions) in the dataset folder, with counts and example filenames.
        '''
        file_types = self.get_file_types_in_dataset(data_dir, max_examples)
        print("\n--- File Types in Dataset ---")
        for ext, examples in file_types.items():
            print(f"Extension: '{ext or '[no extension]'}' | Count: {len(examples)} | Examples: {examples}")
        print(f"Total unique file types: {len(file_types)}\n")

    
    def merge_quarantine_sets(self, data_dirs=None, all_sets=False, del_original=False):
        '''
        Merge one or more quarantine set directories into the main output directory.

        Expected source layout:
            quarantine/<set_name>/<class_name>/{data.yaml,dataset_file_directory.csv,images/,masks/,labels/}
        '''
        utils = PipelineUtils(config=self.config, output_dir=self.output_dir)
        overwrite = self.config.get('overwrite', False)

        if all_sets:
            set_dirs = [
                os.path.join(self.quarantine_dir, name)
                for name in os.listdir(self.quarantine_dir)
                if os.path.isdir(os.path.join(self.quarantine_dir, name))
            ]
            if not set_dirs:
                raise ValueError('No quarantine set directories found.')
        elif isinstance(data_dirs, (str, os.PathLike)):
            set_dirs = [os.fspath(data_dirs)]
        elif data_dirs:
            set_dirs = [os.fspath(path) for path in data_dirs]
        else:
            raise ValueError('Provide one or more quarantine set directories, or set all_sets=True.')

        merged = []

        for set_dir in set_dirs:
            set_dir = os.path.abspath(set_dir)
            if not os.path.isdir(set_dir):
                raise ValueError(f'Quarantine set directory does not exist: {set_dir}')

            set_name = os.path.basename(os.path.normpath(set_dir))
            class_dirs = [
                os.path.join(set_dir, name)
                for name in os.listdir(set_dir)
                if os.path.isdir(os.path.join(set_dir, name))
            ]

            if not class_dirs:
                logging.warning(f'No class directories found in {set_dir}. Skipping.')
                continue

            for class_dir in class_dirs:
                class_name = os.path.basename(class_dir)
                dest_class_dir = utils.ensure_class_and_subfolders(class_name)

                src_csv = os.path.join(class_dir, 'dataset_file_directory.csv')
                src_yaml = os.path.join(class_dir, 'data.yaml')
                dst_csv = os.path.join(dest_class_dir, 'dataset_file_directory.csv')
                dst_yaml = os.path.join(dest_class_dir, 'data.yaml')

                if not os.path.exists(src_csv):
                    logging.warning(f'No CSV found for {set_name}/{class_name}. Skipping.')
                    continue

                with open(src_csv, 'r', newline='') as csvfile:
                    rows = list(csv.DictReader(csvfile))

                if not rows:
                    logging.info(f'No rows found in {src_csv}. Skipping.')
                    continue

                if os.path.exists(dst_csv):
                    with open(dst_csv, 'r', newline='') as csvfile:
                        existing_rows = list(csv.DictReader(csvfile))
                    already_present = any(row.get('dataset_name') == set_name for row in existing_rows)

                    if already_present and not overwrite:
                        logging.info(f"Set '{set_name}' already merged into class '{class_name}'. Skipping.")
                        continue

                    if already_present and overwrite:
                        self.delete_set_from_class(class_name, set_name)

                if not os.path.exists(dst_yaml):
                    if os.path.exists(src_yaml):
                        shutil.copy2(src_yaml, dst_yaml)
                    else:
                        utils.write_yolo_yaml(class_name)

                for row in rows:
                    for subdir, column in (
                        ('images', 'output_image'),
                        ('masks', 'output_mask'),
                        ('labels', 'output_label'),
                    ):
                        filename = row.get(column)
                        if not filename:
                            continue

                        src_path = os.path.join(class_dir, subdir, filename)
                        dst_path = os.path.join(dest_class_dir, subdir, filename)

                        if os.path.exists(src_path):
                            shutil.copy2(src_path, dst_path)
                        elif row.get('deletion_status') != 'deleted':
                            raise FileNotFoundError(
                                f"Missing expected file for set '{set_name}', class '{class_name}': {src_path}"
                            )

                utils.append_csv_rows(class_name, rows)
                merged.append((set_name, class_name, len(rows)))

            if del_original:
                shutil.rmtree(set_dir)
                logging.info(f"Deleted merged quarantine set: {set_dir}")

        return merged

                

            

