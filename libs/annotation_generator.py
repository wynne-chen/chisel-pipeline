#--- IMPORTS ---#
import os
import logging
import cv2
import numpy as np

#--- ANNOTATION GENERATOR ---#

class AnnotationGenerator:
    '''
    YOLO specific functions. All functions related to generating the necessary intermediate components for YOLO format models.
    '''

    def __init__(self, config=None):
        self.config = config or {}
        self.next_annotation_id = 1

        self.epsilon = self.config.get('epsilon', 0.5)
        self.epsilon_merge = self.config.get('epsilon_merge', 1.0)
        self.min_hole_area = self.config.get('min_hole_area', 100)
        self.max_holes = self.config.get('max_holes', 10)
        self.merge_holes = self.config.get('merge_holes', True)

    def is_clockwise(self, contour):
        value = 0
        num = len(contour)
        for i in range(num):
            x1, y1 = contour[i][0]
            x2, y2 = contour[(i + 1) % num][0]
            value += (x2 - x1) * (y2 + y1)
        return value < 0
    
    def get_merge_pt_idx(self, contour1, contour2):
        idx1, idx2 = 0, 0
        min_dist = float('inf')

        for i, p1 in enumerate(contour1):
            for j, p2 in enumerate(contour2):
                dx = p2[0][0] - p1[0][0]
                dy = p2[0][1] - p1[0][1]
                dist = dx * dx + dy * dy
                if dist < min_dist:
                    min_dist = dist
                    idx1, idx2 = i, j
        return idx1, idx2
    
    def merge_contours(self, contour1, contour2, idx1, idx2):
        merged = []
        
        merged.extend(contour1[:idx1 + 1])  # outer loop start point
        merged.extend(contour2[idx2:])      # inner loop start point 
        merged.extend(contour2[:idx2 + 1])  # inner loop end point
        merged.extend(contour1[idx1:])      # outer loop end point

        return np.array(merged)
    
    def merge_with_parent(self, parent, child):
        if not self.is_clockwise(parent):
            parent = parent[::-1]
        
        if self.is_clockwise(child):
            child = child[::-1]
        
        idx1, idx2 = self.get_merge_pt_idx(parent, child)
        merged = self.merge_contours(parent, child, idx1, idx2)
        return merged
    
    def mask_to_polygons(self, mask):
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask_to_polygons: Mask must be 2D for findContours, got shape {mask.shape}")
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                approx = cv2.approxPolyDP(contour, self.epsilon, True)
                poly = approx.reshape(-1).tolist()
                polygons.append(poly)
            else:
                logging.warning('Contour found with less than 3 points, not sufficient to form a polygon.')
        return polygons

    def mask_to_polygons_with_holes(self, mask):
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"mask_to_polygons: Mask must be 2D for findContours, got shape {mask.shape}")
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # changing to CCOMP to find child polygons (holes)

        if not contours or not hierarchy:
            return []
        
        hierarchy = hierarchy[0]

        contours_approx = []
        for i, contour in enumerate(contours):
            if len(contour) <3:
                contours_approx.append(None) # append None to maintain the index alignment
                continue
            approx  = cv2.approxPolyDP(contour, self.epsilon, True)
            if len(approx) >= 3:
                contours_approx.append(approx)
            else: logging.warning(f"Contour {i} has less than 3 points, not sufficient to form a polygon.")

        parents = {}
        holes_map = {}

        for i, contour in enumerate(hierarchy):
            parent_idx = hierarchy[3] # This looks like an error. This is taking the 4th contour's full [next, prev, child, parent]. parent_idx then
                                      # becomes an array. not a scalar.
            
            if parent_idx == -1:
                if contours_approx[i] is not None:
                    parents[i] = contours_approx[i]
                    holes_map[i] = []
                else:
                    logging.warning(f"Contour {i} has no approximation, skipping.")
            else:
                if contours_approx[i] is not None:
                    holes_map.setdefault(parent_idx, []).append(i)
       
        final_polygons = []

        for parent_idx, parent_contour in parents.items():
            merged = parent_contour
            holes = holes_map.get(parent_idx, [])
            if len(holes) > self.max_holes:
                logging.warning(
                    "Parent contour %s has %s holes (> max_holes=%s); skipping hole merges for this parent.",
                    parent_idx,
                    len(holes),
                    self.max_holes,
                )
                holes = []
            for hole_idx in holes:
                hole_contour = contours_approx[hole_idx]
                if hole_contour is None or len(hole_contour) < 3:
                    continue
                if cv2.contourArea(hole_contour) < self.min_hole_area:
                    continue
                merged = self.merge_with_parent(merged, hole_contour)
            merged = cv2.approxPolyDP(merged, self.epsilon_merge, True)
            if len(merged) < 3:
                logging.warning("Merged contour for parent %s collapsed to < 3 points; skipping.", parent_idx)
                continue
            final_polygons.append(merged.reshape(-1).tolist())
        return final_polygons

    def create_coco_annotations(self, image_id, polygons, category_id):
        annotations = []
        for poly in polygons:
            ann = {
                'id': self.next_annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': [poly],
                'area': cv2.contourArea(np.array(poly).reshape(-1, 2)),
                'bbox': list(cv2.boundingRect(np.array(poly).reshape(-1, 2))),
                'iscrowd': 0
            }
            annotations.append(ann)
            self.next_annotation_id += 1
        return annotations

    def convert_coco_to_yolo(self, coco_input, set_name, output_dir):
        output_labels_dir = os.path.join(output_dir, set_name, 'labels')
        os.makedirs(output_labels_dir, exist_ok=True)
        for img_info in coco_input['images']:
            img_id = img_info['id']
            file_name = img_info['file_name']
            img_ann = [ann for ann in coco_input['annotations'] if ann['image_id'] == img_id]
            img_w, img_h = img_info['width'], img_info['height']
            if img_ann:
                class_id = 0
                with open(os.path.join(output_labels_dir, os.path.splitext(file_name)[0] + '.txt'), 'w') as file_object:
                    for ann in img_ann:
                        polygon = ann['segmentation'][0]
                        normalised_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                        file_object.write(f'{class_id} ' + ' '.join(normalised_polygon) + '\n')

    def generate_tile_yolo_annotation(self, polygons, class_id, img_w, img_h):
        '''
        Generate YOLO annotation lines for a single tile.
        Always writes class 0 for single-class datasets.
        Only writes lines if the polygon has at least 6 coordinates (3 points).
        '''
        annotation_lines = []
        for poly in polygons:
            if len(poly) < 6:
                # Skip degenerate polygons (less than 3 points)
                continue
            normalised_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(poly)]
            annotation_lines.append(f'0 ' + ' '.join(normalised_polygon))
        return annotation_lines