#--- IMPORTS ---#
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

#--- VISUALISER ---#

class Visualiser:
    def __init__(self, config=None):
        self.config = config or {}
        self.default_alpha = self.config.get("mask_alpha", 0.5)
        self.logger = logging.getLogger(__name__)

    def return_image_mask_pair(self, image, mask):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        if image is not None and image.ndim == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title('Image')
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title('Mask')
        plt.tight_layout()
        return plt.gcf()

    def display_image_mask_pair(self, image, mask):
        fig = self.return_image_mask_pair(image, mask)
        fig.show()
        print(f'Image size is {image.shape}.')
        print(f'Mask size is {mask.shape}.')
    
    def overlay_mask_on_image(self, image, mask, alpha=None):
        if alpha is None:
            alpha = self.default_alpha
        plt.figure(figsize=(8, 8))
        if image is not None and image.ndim == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.imshow(mask, cmap='gray', alpha=alpha, vmin=-0.5, vmax=4.5)
        plt.axis('off')
        plt.title('Image and Mask Overlaid')
        plt.tight_layout()
        plt.show()

    def display_random_pair(self, image_paths, mask_paths):
        random_idx = np.random.randint(0, len(image_paths))
        img = cv2.imread(image_paths[random_idx])
        mask = cv2.imread(mask_paths[random_idx], cv2.IMREAD_GRAYSCALE)
        self.display_image_mask_pair(img, mask)

    def read_annotations_from_txt(self, txt_path):
        annotations = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    annotations.append({'class_id': class_id, 'coords': coords})
        return annotations

    def overlay_label_on_image(self, image, annotations_or_txt):
        plt.figure(figsize=(8, 8))
        if isinstance(annotations_or_txt, str):
            annotations = self.read_annotations_from_txt(annotations_or_txt)
        else:
            annotations = annotations_or_txt
        if image is not None and image.ndim == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        for ann in annotations:
            coords = ann.get('coords')
            if coords:
                coords = np.array(coords).reshape(-1, 2)
                plt.plot(coords[:, 0], coords[:, 1], linewidth=2)
        plt.axis('off')
        plt.title('Image with Annotations')
        plt.tight_layout()
        plt.show()

    def check_random_triplet(self, class_dir, image_file: str | None = None, show_inline: bool = False):
        '''
        Randomly select and display a processed image, its mask, and the reconstructed mask from the YOLO label file.
        ONLY RUN ON PROCESSED DATASETS.
        
        class_dir: expect path to overall class directory ('vegetation', etc.)
        '''
        images_dir = os.path.join(class_dir, 'images')
        masks_dir = os.path.join(class_dir, 'masks')
        labels_dir = os.path.join(class_dir, 'labels')

        if image_file is None:
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg'))]
            if not image_files:
                print('No images found in', images_dir)
                return None, ['No images found in', images_dir]
            rand_img_file = random.choice(image_files)
        else:
            rand_img_file = image_file

        img_name = os.path.splitext(rand_img_file)[0]
        mask_file = img_name + '_m.png'
        label_file = img_name + '.txt'

        img_path = os.path.join(images_dir, rand_img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        label_path = os.path.join(labels_dir, label_file)

        text_lines = []
        text_lines.append(f'Image: {img_path}')
        text_lines.append(f'Mask: {mask_path}')
        text_lines.append(f'Label: {label_path}')
        
        if show_inline:
            for line in text_lines:
                print(line)
    

        image = cv2.imread(img_path)
        if image is None:
            text_lines.append(f'Could not load image: {img_path}')
            return None, text_lines
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # og image based mask
        if mask is None:
            text_lines.append(f'Could not load mask: {mask_path}')
            return None, text_lines
        
        recon_mask = np.zeros((height, width), dtype=np.uint8) # yolo text mask
        if os.path.exists(label_path):
            with open(label_path, 'r') as fp:
                lines = fp.readlines()
            polygons = {}
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                category = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                polygon = [coords[i] * width if i % 2 == 0 else coords[i] * height for i in range(len(coords))]
                polygon_xy = [polygon[i:i+2] for i in range(0, len(polygon), 2)]
                if category not in polygons:
                    polygons[category] = [polygon_xy]
                else:
                    polygons[category].append(polygon_xy)
            for category in polygons:
                for poly in polygons[category]:
                    cv2.fillPoly(recon_mask, [np.array(poly, dtype=np.int32)], 255)
        else:
            text_lines.append(f'Label file does not exist: {label_path}')

        # Plot
        fig = plt.figure(figsize=(27, 9))
        plt.subplot(1, 3, 1); plt.imshow(image_rgb); plt.title("Image", fontsize=20); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(recon_mask, interpolation=None, vmin=-0.5, vmax=4.5, cmap="gray"); plt.title("YOLO Labels Mapped", fontsize=20); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(mask, interpolation=None, vmin=-0.5, vmax=4.5, cmap="gray"); plt.title("Original Mask", fontsize=20); plt.axis("off")
        plt.suptitle(f"Image, Mask, and YOLO Mask for {img_name}", fontsize=30)
        plt.tight_layout()
        if show_inline:
            plt.show()
        return fig, text_lines

    def display_random_grid(self, image_paths, mask_paths, n=10, grid_shape=None, mask_alpha=0.5, max_dim=256):
        '''
        Display a random grid of image-mask pairs overlaid (mask unfiltered, just made visible if needed).
        Args:
            image_paths: list of image file paths
            mask_paths: list of mask file paths
            n: number of pairs to display (default 10)
            grid_shape: (rows, cols) tuple, or auto-calculated
            mask_alpha: alpha for mask overlay
            max_dim: max display size for each image
        '''
        
        n = min(n, len(image_paths), len(mask_paths))
        if n == 0:
            logging.warning("No image/mask pairs to display.")
            return None
        idxs = np.random.choice(len(image_paths), n, replace=False)
        if grid_shape is None:
            cols = min(5, n)
            rows = math.ceil(n / cols)
        else:
            rows, cols = grid_shape
        plt.figure(figsize=(cols * 3, rows * 3))
        for i, idx in enumerate(idxs):
            img = cv2.imread(image_paths[idx])
            mask = cv2.imread(mask_paths[idx], cv2.IMREAD_UNCHANGED)
            # Resize for display
            if img is not None:
                scale = min(max_dim / img.shape[0], max_dim / img.shape[1], 1.0)
                img_disp = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
            else:
                img_disp = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            if mask is not None:
                if mask.ndim == 3 and mask.shape[2] == 3:
                    mask_disp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask_disp = mask
                mask_disp = cv2.resize(mask_disp, (img_disp.shape[1], img_disp.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_disp = np.zeros(img_disp.shape[:2], dtype=np.uint8)
            
            del img
            del mask
            
            plt.subplot(rows, cols, i+1)
            plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
            plt.imshow(mask_disp, cmap='jet', alpha=mask_alpha)
            plt.axis('off')
            plt.title(f'Pair {i+1}')
        plt.tight_layout()
        return plt.gcf()
