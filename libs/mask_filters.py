#--- IMPORTS ---#

import numpy as np

#--- MASK FILTERS ---#

'''
This file defines mask filters for various datasets.
The filters are methods that take a mask and a target class id and return a binary mask.
Additional mask filters using the template provided under "# Template for future dataset-specific mask filters"
DO NOT CHANGE THE INPUTS OR OUTPUTS OF THESE METHODS.
Copy the template and do not delete it. Leave the template at the bottom of the file.
'''


def default(mask, target_class_id):
    '''
    Generic mask filter for the purposes of dataset EDA when a new dataset is multiclass.

    Args:
        mask (np.array):         The input mask, can be either 2D or 3D
        target_class_id (int):   The class ID to be isolated as per the unique values found in the set

    Returns:
        binary_mask(np.array):   A binary mask (0 for background, 255 for the target class), type np.uint8.
    '''
    if isinstance(target_class_id, int):
        binary_mask = (mask == target_class_id).astype(np.uint8) * 255
    else:
        binary_mask = np.all(mask == target_class_id, axis=-1).astype(np.uint8) * 255
    return binary_mask


def chesapeake(mask):
    '''
    Chesapeake dataset mask filter.
    Combines classes 9 and 12 into a binary mask (255 for those classes, 0 otherwise).
    No input for class_id because there is only one relevant class in this dataset, which is 'road'.
    
    Args:
        mask (np.array):          The input mask (expects rasterio read)
    Returns:
        binary_mask (np.array):   A binary mask (0 for background, 255 for the target class), type np.uint8.
    '''
    band = mask[0] if mask.ndim == 3 else mask
    binary_mask = np.isin(band, [9, 12]).astype(np.uint8) * 255
    return binary_mask


def dubai(mask, target_class_id):
    '''
    Filters the multiclass dubai dataset masks to create binary masks for specific classes

    Args:
        mask (np.array):         The input mask with pixel values representing landcover classes (H, W, 3)
        target_class_id (int):   The class ID to be isolated, as per the CLASS_ID_MAP

    Returns:
        binary_mask(np.array):   A binary mask (0 for background, 255 for the target class), type np.uint8.
    '''
    # Colour mapping (BGR)
    dubai_map = {
        0: np.array([246,  41, 132]), # background
        1: np.array([152,  16,  60]), # building
        2: np.array([ 58, 221, 254]), # vegetation
        3: np.array([ 41, 169, 226]), # water
        4: np.array([228, 193, 110]), # road
    }
    # Unlabelled and padding colours also map to background (0)
    background_colours = [
        np.array([155, 155, 155]), # unlabelled
        np.array([0, 0, 0])        # padding
    ]

    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(f'dubai: Expected 3-channel colour mask, got shape {mask.shape}')

    target_colour = dubai_map.get(target_class_id)
    matches = np.all(mask == target_colour, axis=-1)

    # If background, also include unlabelled and padding
    if target_class_id == 0:
        for bg_col in background_colours:
            matches |= np.all(mask == bg_col, axis=-1)

    binary_mask = matches.astype(np.uint8) * 255
    return binary_mask


def landcover(mask, target_class_id):
    """
    Filters the multiclass landcover.ai.v1 dataset masks to create binary masks for specific classes

    Args:
        mask (np.array):            The input mask with pixel values representing landcover classes
                                    In this case, the classes map perfectly so no change is required.
        target_class_id (int):      The class ID to be isolated, as per the CLASS_ID_MAP defined on line 73

    Returns:
        binary_mask:                A binary mask (0 for background, 255 for the target class), type np.uint8.
    """

    binary_mask = (mask == target_class_id).astype(np.uint8) * 255
    return binary_mask

def oem_full(mask, target_class_id):
    '''
    Filters the multiclass open earth map (full) dataset masks to create binary masks for specific classes

    Args:
        mask (np.array):            The input mask with pixel values representing landcover classes
                                    As per the documentation, the classes are indexed as follows:
                                    1: bareland 
                                    2: rangeland
                                    3: developed space
                                    4: road
                                    5: tree
                                    6: water
                                    7: agriculture land
                                    8: building
        target_class_id (int):      The class ID to be isolated, as per the CLASS_ID_MAP defined on line 73

    Returns:
        binary_mask:                A binary mask (0 for background, 255 for the target class), type np.uint8.
    '''
    oem_map = {
        0: 0, # background
        1: 8, # building
        2: 5, # vegetation
        3: 6, # water
        4: 4, # road
    }
    binary_mask = (mask == oem_map[target_class_id]).astype(np.uint8) * 255
    return binary_mask



# Template for future dataset-specific mask filters
#   
# def your_dataset(mask, target_class_id):
#     '''
#     Description of what this filter does and for which dataset.
    
#     Args:
#     your_input:  description
    
#     Returns
#     binary_mask:  A binary mask (0 for background, 255 for the target class), type np.uint8. <-- IMPT! return as uint8
#     '''
#     # Your logic here
#     return binary_mask
