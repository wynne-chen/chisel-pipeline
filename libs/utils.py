#--- IMPORTS ---#
import os
import glob
import csv
import logging
import yaml

#--- UTILS ---#

DATASET_CSV_FIELDNAMES = [
    'output_image', 'output_mask', 'output_label',
    'original_image', 'original_mask', 'category', 
    'tile_index', 'dataset_name', 'deletion_status',
    'multiclass', 'src_epsg', 'crs_wkt', 'projection_format', 
    'bounds', 'transform', 'x_cm_per_pixel', 
    'y_cm_per_pixel', 'mean_cm_per_pixel', 'num_bands', 
    'centroid_src', 'centroid_wgs84', 'country'
]

class PipelineUtils:
    '''
    Class for utility functions used throughout the pipeline.
    '''
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.class_names = config.get('class_names', [])
        self.train_split = config.get('train_split', 0.8)
        self.val_split = config.get('val_split', 0.2)
        self.overwrite = config.get('overwrite', False)

    def ensure_class_and_subfolders(self, class_name):
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for sub in ['images', 'masks', 'labels']:
            os.makedirs(os.path.join(class_dir, sub), exist_ok=True)
        return class_dir

    def write_yolo_yaml(self, class_name):
        class_dir = os.path.join(self.output_dir, class_name)
        yaml_path = os.path.join(class_dir, 'data.yaml')
        if os.path.exists(yaml_path):
            logging.info(f"YAML already exists at {yaml_path}, skipping write.")
            return
        yaml_data = {
            'names': [class_name],
            'nc': 1,
            'train': 'images/',
            'val': 'images/',
            'test': ' '
        }
        if self.train_split is not None and self.val_split is not None:
            yaml_data['split'] = [
                {'train': self.train_split},
                {'val': self.val_split}
            ]
        with open(yaml_path, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        logging.info(f"Wrote YAML to {yaml_path}")

    def update_csv_mapping(self, class_name, set_name, mapping_rows):
        class_dir = os.path.join(self.output_dir, class_name)
        csv_path = os.path.join(class_dir, 'dataset_file_directory.csv')
        fieldnames = DATASET_CSV_FIELDNAMES
        existing_rows = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if not (self.overwrite and row.get('dataset_name') == set_name):
                        existing_rows.append(row)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
            for row in mapping_rows:
                writer.writerow(row)
        logging.info(f"Updated CSV at {csv_path}")

    def append_csv_row(self, class_name, row):
        '''
        Append a single row to the dataset_file_directory.csv for incremental updates.
        '''
        self.append_csv_rows(class_name, [row])
    
    def append_csv_rows(self, class_name, rows):
        """Append many rows with a single open; no-op if rows is empty."""
        if not rows:
            return
        class_dir = os.path.join(self.output_dir, class_name)
        csv_path = os.path.join(class_dir, 'dataset_file_directory.csv')
        fieldnames = DATASET_CSV_FIELDNAMES
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)


    def is_file_processed(self, class_name, original_image):
        class_dir = os.path.join(self.output_dir, class_name)
        csv_path = os.path.join(class_dir, 'dataset_file_directory.csv')
        if not os.path.exists(csv_path):
            return False
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['original_image'] == original_image:
                    return True
        return False

    def update_csv_deletion_status(self, class_name):
        '''
        Scan the CSV for the given class, check if the image, mask, and label files exist, and update the 'deletion_status' field.
        Sets 'deleted' if any file is missing, 'not_deleted' if all exist. Updates the CSV in place.
        '''
        class_dir = os.path.join(self.output_dir, class_name)
        images_dir = os.path.join(class_dir, 'images')
        masks_dir = os.path.join(class_dir, 'masks')
        labels_dir = os.path.join(class_dir, 'labels')
        csv_path = os.path.join(class_dir, 'dataset_file_directory.csv')
        if not os.path.exists(csv_path):
            logging.warning(f"No CSV found at {csv_path} for update.")
            return
        updated_rows = []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            if 'deletion_status' not in fieldnames:
                fieldnames.append('deletion_status')
            for row in reader:
                image_file = row.get('output_image')
                mask_file = row.get('output_mask')
                label_file = row.get('output_label')
                image_path = os.path.join(images_dir, image_file) if image_file else None
                mask_path = os.path.join(masks_dir, mask_file) if mask_file else None
                label_path = os.path.join(labels_dir, label_file) if label_file else None
                if not (image_file and mask_file and label_file):
                    row['deletion_status'] = 'deleted'
                elif not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(label_path)):
                    row['deletion_status'] = 'deleted'
                    logging.info(f"Marked as deleted in CSV: {image_file}, {mask_file}, {label_file}")
                else:
                    row['deletion_status'] = 'not_deleted'
                updated_rows.append(row)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in updated_rows:
                writer.writerow(row)
        logging.info(f"CSV deletion status updated for class {class_name}.")

    def clean_yolo_labels(self, class_name):
        '''
        Ensures all YOLO label files in the class have only class 0 and contain polygons (not just bounding boxes).
        Deletes files with only degenerate bounding boxes.
        '''
        class_dir = os.path.join(self.output_dir, class_name)
        labels_dir = os.path.join(class_dir, 'labels')

        for filepath in glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True):
            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Removes empty files and files that only contain bounding boxes (bb) without polygons
            only_bb = [line for line in lines if len(line.strip().split(' ')) <= 5]
            if len(lines) == 0 or len(only_bb) == len(lines):
                logging.info(f"No polygons found for {filepath}. Deleting file.")
                os.remove(filepath)
                continue

            # Ensure category is 0
            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) > 0 and parts[0] != '0':
                    parts[0] = '0'
                    changed = True
                new_line = ' '.join(parts)
                new_lines.append(new_line)

            # Only write back if changes were made
            if changed:
                with open(filepath, 'w') as file:
                    for line in new_lines:
                        file.write(line + '\n')
        # Update CSV deletion status after cleaning
        self.update_csv_deletion_status(class_name)

    def clean_unlabelled_images(self, class_name):
        '''
        Remove image and mask files without a corresponding .txt label and update CSV with deletion status.
        '''
        class_dir = os.path.join(self.output_dir, class_name)
        images_dir = os.path.join(class_dir, 'images')
        masks_dir = os.path.join(class_dir, 'masks')
        labels_dir = os.path.join(class_dir, 'labels')
        csv_path = os.path.join(class_dir, 'dataset_file_directory.csv')
        if not os.path.exists(csv_path):
            logging.warning(f"No CSV found at {csv_path} for cleaning.")
            return
        updated_rows = []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            if 'deletion_status' not in fieldnames:
                fieldnames.append('deletion_status')

            for row in reader:
                # Preserve existing deletion_status or default to 'not_deleted'
                current_deletion_status = row.get('deletion_status', 'not_deleted')

                image_file = row['output_image']
                mask_file = row['output_mask']
                label_file = row['output_label']
                image_path = os.path.join(images_dir, image_file)
                mask_path = os.path.join(masks_dir, mask_file)
                label_path = os.path.join(labels_dir, label_file)

                deleted_by_cleaner = False

                # Check if label file is missing
                if not os.path.exists(label_path):
                    if os.path.exists(image_path):
                        logging.info(f"Condition met for deleting image: {image_path}")
                        os.remove(image_path)
                        deleted_by_cleaner = True
                        logging.info(f"Deleted unlabelled image: {image_path}")
                    if os.path.exists(mask_path):
                        logging.info(f"Condition met for deleting mask: {mask_path}")
                        os.remove(mask_path)
                        deleted_by_cleaner = True
                        logging.info(f"Deleted unlabelled mask: {mask_path}")

                    # If this function actually deleted files, update status
                    if deleted_by_cleaner:
                        row['deletion_status'] = 'deleted'
                    else:
                        # If label was missing but no files were found (already skipped/deleted),
                        # preserve existing status (which should be 'deleted' from process_dataset_incremental)
                        row['deletion_status'] = current_deletion_status
                else:
                    # If label file exists, ensure status is 'not_deleted'
                    row['deletion_status'] = 'not_deleted'

                updated_rows.append(row)

        # Write updated CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in updated_rows:
                writer.writerow(r)
        # Update CSV deletion status after cleaning
        self.update_csv_deletion_status(class_name)

    def clean_dataset(self, class_name):
        '''
        Cleans a dataset by:
        - Removing degenerate YOLO label files
        - Removing images/masks without labels
        - Updating the CSV deletion status

        Args:
            class_name (str): The class/dataset name (e.g., 'vegetation')
        '''
    
        print(f"Cleaning YOLO label files for class '{class_name}'...")
        self.clean_yolo_labels(class_name)
        print(f"Removing unlabelled images and updating CSV for class '{class_name}'...")
        self.clean_unlabelled_images(class_name)
        print(f"Updating CSV deletion status for class '{class_name}'...")
        self.update_csv_deletion_status(class_name)
        print("Dataset cleaning complete.")

