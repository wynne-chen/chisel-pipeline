#--- IMPORTS ---#
import logging
import cv2
import numpy as np
import rasterio as rio

#--- MASK FILTERS ---#
from libs.mask_filters import chesapeake, dubai, landcover, oem_full, default


#--- IMAGE PROCESSOR ---#

class ImageProcessor:
    '''
    Class for splitting images into tiles and loading images and masks.
    '''
    def __init__(self, config=None):
        self.config = config or {}
        self.target_size = self.config.get('target_size', 512)

    def split_image_generator(self, image, target_size):
        '''
        Generator that yields tiles of the image of the given target size.
        '''
        for y in range(0, image.shape[0], target_size):
            for x in range(0, image.shape[1], target_size):
                tile = image[y:y + target_size, x:x + target_size]
                if tile.shape[0] == target_size and tile.shape[1] == target_size:
                    yield tile, y, x
                else:
                    pad_y = target_size - tile.shape[0]
                    pad_x = target_size - tile.shape[1]
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0) if tile.ndim == 3 else np.pad(tile, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                    yield tile, y, x

    def is_multiclass(self, mask):
        '''
        Returns True if the mask is multiclass, False otherwise.
        '''
        if mask is None:
            return False
        if mask.ndim == 2:
            return np.unique(mask).shape[0] > 2
        elif mask.ndim == 3 and mask.shape[0] == 1:
            return np.unique(np.squeeze(mask, axis=0)).size > 2
        elif mask.ndim == 3 and mask.shape[-1] in (3, 4):
            colors = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0)
            return colors.shape[0] > 2
        elif mask.ndim == 3 and mask.shape[0] > 1:
            pixels = np.moveaxis(mask, 0, -1).reshape(-1, mask.shape[0])
            return np.unique(pixels, axis=0).shape[0] > 2
        else:
            return False

    def load_image(self, image_path):
        if image_path.lower().endswith(('.tif', '.tiff')):

            with rio.open(image_path) as src:
                img = src.read()

            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
                if img.shape[2] == 3: 
                    img = img[..., ::-1] # convert RGB to BGR
                elif img.shape[2] > 3:
                    img = img[..., :3][..., ::-1] # convert 4-channel image to 3-channel image and then to BGR
            elif img.ndim == 2:
                pass
            else:
                img = img.squeeze(0)
        else:
            img = cv2.imread(image_path)

        return img

    def load_mask(self, mask_path):
        if mask_path.lower().endswith(('.tif', '.tiff')):
            with rio.open(mask_path) as src:
                mask = src.read()
                if mask.ndim == 3:
                    if mask.shape[0] == 1:
                        mask = np.squeeze(mask, axis=0)
                    else:
                        mask = np.transpose(mask, (1, 2, 0))
                elif mask.ndim == 2:
                    pass
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        return mask

    def normalise_image(self, image):
        x_min = np.min(image)
        x_max = np.max(image)
        if x_max - x_min == 0:
            logging.warning("Image is constant; returning zeros.")
            return np.zeros_like(image)
        return (image - x_min) / (x_max - x_min)

    def filter_mask_by_class_id(self, mask, class_id, set_name):
        '''
        Filter masks according to set-based rules and returns np array with 0 and 255 values.
        Assumes 2 general cases: 
            maximum 2 values: either indexed {0,1} or visible {0,255}
            more than 2 values: either run unique dataset filters, or normalise
        Ensures output is always a 2D mask
        '''
        unique_values = np.unique(mask)

        # 1. If the mask is single-class, visible, and clean, return mask
        if len(unique_values) <= 2:
            if set(unique_values).issubset({0, 255}):
                mask = mask
                if mask.ndim == 3:
                    if mask.shape[2] == 3:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    elif mask.shape[0] == 1:
                        mask = np.squeeze(mask)
                    else:
                        mask = np.transpose(mask, (1, 2, 0))
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    
            # 2. If the mask is 'invisible' (only 0 and 1), make it visible by multiplying by 255
            elif set(unique_values).issubset({0, 1}):
                mask = (mask * 255).astype(np.uint8)
            # 3. Assume the mask is binary but requires normalisation
            else:
                mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        # 4. If the mask is 'invisible' and indexed with multiple classes, or visible but has multiple classes,
        #    run the dataset-specific function to return the binary mask for the specific class desired.
        elif set_name == 'chesapeake':
            mask = chesapeake(mask)
        elif set_name == 'dubai':
            mask = dubai(mask, class_id)
        elif set_name == 'landcover':
            mask = landcover(mask, class_id)
        elif set_name == 'openearthmap-full':
            mask = oem_full(mask, class_id)
        else:
            # 5. Unknown multiclass: indexed 2D mask -> default(mask, class_id).
            # RGB label PNGs need a dataset-specific filter in mask_filters (see dubai).
            logging.info(
                "Unknown multiclass dataset; using default filter for class_id=%s (set_name=%s).",
                class_id,
                set_name,
            )
            m = np.asarray(mask)
            # 1, H, W -> H, W
            if m.ndim == 3 and m.shape[0] == 1:
                m = np.squeeze(m, axis=0)
            # (C, H, W) detect with small band count -> H, W, C
            if m.ndim == 3 and m.shape[0] <= 4 and m.shape[0] < min(m.shape[1], m.shape[2]):
                m = np.transpose(m, (1, 2, 0))
            # 3-channel colour mask -> 2D indexed mask
            if m.ndim == 3 and m.shape[-1] == 3:
                g0, g1, g2 = m[..., 0], m[..., 1], m[..., 2] 
                if np.array_equal(g0, g1) and np.array_equal(g1, g2):
                    m = np.ascontiguousarray(g0)
                else:
                    raise ValueError(
                        f"Unknown set '{set_name}': 3-channel colour mask needs a "
                        f"dataset-specific entry in mask_filters (shape {m.shape})."
                    )
            if m.ndim != 2:
                raise ValueError(
                    f"Unknown set '{set_name}': expected a 2D indexed mask after "
                    f"normalisation, got shape {m.shape!r}."
                )
            mask = default(m, class_id)
        
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D after filtering, got shape {mask.shape}")
        return mask