#--- IMPORTS ---#

import os

#---- CONFIG ----#

config = {
    'skip_keywords': ['csv', 'json', 'xml', 'vis', 'md', 'zip', 'pdf', 'txt', 'py', 'yaml'],
    'mask_dirs': ['mask', 'masks', 'label', 'labels'],
    'mask_prefixes': ['mask_', 'label_'],
    'mask_suffixes': ['_mask', '_m'],
    'image_prefixes': ['image_', 'img_'],
    'image_suffixes': ['_image', '_img'],
    'mask_alpha': 0.5,
    'quarantine_dir': os.path.join(os.getcwd(), 'quarantine'),
    'output_dir': os.path.join(os.getcwd(), 'output'),
    'target_size': 512,
    'class_names': ['building', 'vegetation', 'water', 'road'],
    'log_level': 'WARNING',
    'train_split': 0.8,
    'val_split': 0.2,
    'overwrite': False,
    'sample_size': 50,   # number of images to sample for EDA, particularly for methods that are more compute intensive.
    'top_k': 10,         # number of top colours to store for colour distribution analysis and top countries to store for country distribution analysis.
    'random_seed': 42,
    'max_bounds_area': 500_000, # maximum area of the bounds to plot on the map in square kilometers. For reference the area of Poland is ~312,700sqkm
    'epsilon': 0.5, #
    'epsilon_merge': 1.0,
    'min_hole_area': 100, # minimum area of a hole to be included in the polygons. Pixels^2.
    'max_holes': 10,
    'merge_holes': True,
    'ignore_holes': False, # if True, holes in the mask will be ignored and not included in the polygons. 
    'overwrite': False # if True, will overwrite the existing files in the output directory if they already exist.
}

# edit if new classes are added

CLASS_ID_MAP = {
    0: 'background',
    1: 'building',
    2: 'vegetation',
    3: 'water',
    4: 'road'
}

