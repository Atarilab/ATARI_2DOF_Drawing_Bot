from drawing_bot_api.trajectory_optimizer.camera import Camera
from drawing_bot_api.logger import Log
import cv2
import os
import numpy as np
from math import exp
from drawing_bot_api.trajectory_optimizer.config_rl import *

class ImageProcessor:
    def __init__(self):
        self.camera = Camera()
        self.call_counter = 0
        self.log = Log(0)
        self.image_counter = SAVE_IMAGE_FREQ

    def save_image(self, image, directory, type, nr):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(_script_dir, f'images/{directory}/{str(nr)}_{type}.jpg')
        cv2.imwrite(_path, image)
        self.log(f'Saved {type} to {_path}')

    def save_images_combined(self, image1, image2, image3, directory, type, nr):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(_script_dir, f'images/{directory}/{str(nr)}_{type}.jpg')

        # Resize the images to have the same height (optional, ensures alignment)
        height = max(image1.shape[0], image2.shape[0], image2.shape[0])
        _image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
        _image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))
        _image3 = cv2.resize(image3, (int(image3.shape[1] * height / image3.shape[0]), height))

        # Concatenate the images horizontally
        _combined_image = np.hstack((_image1, _image2, _image3))
        cv2.imwrite(_path, _combined_image)
    
    def _simplify_image(self, image):
        # Enhance Contrast
        _gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
       # _enhanced_contrast = _clahe.apply(_gray)

        # Convert to Black and White
        _, _black_and_white = cv2.threshold(_gray, 100, 255, cv2.THRESH_BINARY)

        # Invert
        _inverted = cv2.bitwise_not(_black_and_white)

        return _inverted
    
    def _simplify_drawing(self, image): # OLD
        # Blurring for noise reduction
        _blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Enhance Contrast
        _gray = cv2.cvtColor(_blurred, cv2.COLOR_BGR2GRAY)
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _enhanced_contrast = _clahe.apply(_gray)

        # Convert to Black and White
        _, _black_and_white = cv2.threshold(_enhanced_contrast, 80, 255, cv2.THRESH_BINARY)

        # Invert
        _inverted = cv2.bitwise_not(_black_and_white)

        return _inverted
    
    def calculate_defect_score(self, defects, contour):
        if defects is None:  # No defects
            return 0, 0
        total_defect_depth = 0
        for i in range(defects.shape[0]):
            start_idx, end_idx, far_idx, depth = defects[i, 0]
            total_defect_depth += depth * 0.1  # Depth is in fixed-point format (multiply by 0.1 if necessary)

        # Normalize score by contour perimeter or area
        perimeter = cv2.arcLength(contour, True)
        return total_defect_depth, perimeter
    
    def _calculate_average_score(self, contours):
        _defect_depths = []
        _perimeter_lengths = []
        
        for _contour in contours:
            _hull = cv2.convexHull(_contour, returnPoints=False)
            _defects = cv2.convexityDefects(_contour, _hull)
            _defect_depth, _perimeter_length = self.calculate_defect_score(_defects, _contour)
            _defect_depths.append(_defect_depth)
            _perimeter_lengths.append(_perimeter_length)
        
        _score = np.sum(_defect_depths) / np.sum(_perimeter_lengths)
        return _score

    def calc_similarity_via_hu_moments(self, inv_drawing, inv_template):
        return cv2.matchShapes(inv_drawing, inv_template, cv2.CONTOURS_MATCH_I1, 0)
    
    def calc_similarity_via_convex_hull(self, inv_drawing, inv_template):
        try:
            _contour_drawing, _ = cv2.findContours(inv_drawing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            _contour_template, _ = cv2.findContours(inv_template, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  

            _score_drawing = self._calculate_average_score(_contour_drawing)
            _score_template = self._calculate_average_score(_contour_template)

            #print(f'Score drawing: {_score_drawing}')
            #print(f'Score template: {_score_template}')

            difference_in_defects = abs(_score_drawing - _score_template)
            return difference_in_defects
        except:
            return None
        
    def calc_similiarity_via_chamfer_matching(self, inv_drawing, inv_template):
        _distance_transform = cv2.distanceTransform(255 - inv_template, cv2.DIST_L2, 3)
        _mask_drawing = (inv_drawing == 255)
        _distances = _distance_transform[_mask_drawing]
        del _distance_transform
        return np.mean(_distances)
    
    def calc_rewards_for_individual_points(self, _images_of_template_points, drawing):
        _drawing = drawing
        # turn to inverted binary black and white image
        _simpl_drawing = self._simplify_image(_drawing)
        _grey_drawing = _simpl_drawing#cv2.cvtColor(_simpl_drawing, cv2.COLOR_BGR2GRAY)
        _, _inv_drawing = cv2.threshold(_grey_drawing, 127, 255, cv2.THRESH_BINARY)

        _rewards = []
        for _template_point_image in _images_of_template_points:
            _inv_template_point_image = self._simplify_image(_template_point_image)
            _difference = self.calc_similiarity_via_chamfer_matching(_inv_template_point_image, _inv_drawing)
            if _difference > REWARD_DISTANCE_CLIPPING:
                _difference = REWARD_DISTANCE_CLIPPING

            if REWARD_NORMALIZATION_MODE == 'sigmoid':
                _rewards.append(self._invert_and_normalize_sigmoid(_difference, pre_scaling=1.5))
            elif REWARD_NORMALIZATION_MODE == 'linear':
                _rewards.append(self._invert_and_normalize_linear(_difference))
        
        return _rewards

    
    def _invert_and_normalize_sigmoid(self, value, pre_scaling=1):
        # modified sigmoid function
        # since there are no negative values from shapeMatching the sigmoid is scaled and inverted
        # So values close to 1 represent high similarity and values close to 0 represent low similarity
        # Sensitivity is increased by scaling the calulated similarity measure by 40 before applying the normalzation
        new_value = 2 - (2 / (1 + exp(-pre_scaling*value)))
        return new_value
    
    def _invert_and_normalize_linear(self, value):
        return 1 - (value / REWARD_DISTANCE_CLIPPING)

    def __call__(self, template, drawing=None, save_images=True, save_folder='original', return_image=False, crop_drawing=None, crop_template=None):
        # retrieve both images
        _template = template

        _drawing = drawing
        if _drawing is None:
            _drawing = self.camera()
            if crop_drawing is not None:
                #_drawing = _drawing[10:600, 220:1060]
                _drawing = _drawing[crop_drawing[0]:-crop_drawing[1], crop_drawing[2]:-crop_drawing[3]]
            if crop_template is not None:
                _template = _template[crop_template[0]:-crop_template[1], crop_template[2]:-crop_template[3]]
            
            h, w = _template.shape[:2]
            _drawing = cv2.resize(_drawing, (w, h), interpolation=cv2.INTER_LINEAR)

        _save_images = False
        if self.image_counter == SAVE_IMAGE_FREQ:
            _save_images = True
            self.image_counter = 0

        # turn to inverted binary black and white image
        _simpl_drawing = self._simplify_image(_drawing)
        _grey_drawing = _simpl_drawing #cv2.cvtColor(_simpl_drawing, cv2.COLOR_BGR2GRAY)
        _, _inv_drawing = cv2.threshold(_grey_drawing, 127, 255, cv2.THRESH_BINARY)
        _inv_template = self._simplify_image(_template)

        # save edited images
        if _save_images and save_images:
            self.save_image(_inv_drawing, save_folder, 'drawing', self.call_counter)
            self.save_image(_inv_template, save_folder, 'template', self.call_counter)

        # calc scores
        #similarity_hu_moments = self.calc_similarity_via_hu_moments(_inv_drawing, _inv_template)
        #similarity_convex_hull = self.calc_similarity_via_convex_hull(_inv_drawing, _inv_template)
        similarity_chamfer_matching = self.calc_similiarity_via_chamfer_matching(_inv_drawing, _inv_template)

        if similarity_chamfer_matching > REWARD_DISTANCE_CLIPPING:
            similarity_chamfer_matching = REWARD_DISTANCE_CLIPPING

        # normalize scores
        #norm_similarity_hu_moments = self._normalize(similarity_hu_moments, pre_scaling=50)
        #norm_similarity_convex_hull = self._normalize(similarity_convex_hull, pre_scaling=10)
        norm_similarity_chamfer_matching = self._invert_and_normalize_sigmoid(similarity_chamfer_matching, pre_scaling=1)

        #print(f'Hu moments: {norm_similarity_hu_moments}\t\tConvex hull: {norm_similarity_convex_hull}')

        self.call_counter += 1
        self.image_counter += 1

        #if similarity_convex_hull is None:
        #    return None
        if return_image:
            return norm_similarity_chamfer_matching, _inv_drawing, _inv_template
        else:
            return norm_similarity_chamfer_matching


if __name__ == '__main__':

    img_proc = ImageProcessor()
    #img_proc(None)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _path_template = os.path.join(_script_dir, f'images/test/plot_image_1.jpg')
    _path_drawing = os.path.join(_script_dir, f'images/test/real_image_1.jpg')

    image1 = cv2.imread(_path_template)
    image2 = cv2.imread(_path_drawing)

    invert2 = img_proc._simplify_drawing(image2)
    grey1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    _, binary1 = cv2.threshold(grey1, 127, 255, cv2.THRESH_BINARY)
    invert1 = cv2.bitwise_not(binary1)

    contours1, _ = cv2.findContours(invert1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(invert2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I3, 0)
    #print(f"Shape Similarity: {similarity}")

    # Calculate moments for both contours
    moments1 = cv2.moments(contours1[0])
    moments2 = cv2.moments(contours2[0])

    # Calculate Hu Moments
    huMoments1 = cv2.HuMoments(moments1).flatten()
    huMoments2 = cv2.HuMoments(moments2).flatten()

    huMoments1 = np.array(huMoments1, dtype=np.float32)
    huMoments2 = np.array(huMoments2, dtype=np.float32)

    # Compare using correlation
    similarity = cv2.compareHist(huMoments1, huMoments2, cv2.HISTCMP_CHISQR)
    print(f"Hu Moment Similarity: {similarity*100}")

    cv2.drawContours(image1, contours1, -1, (0, 0, 255), 1)
    cv2.drawContours(image2, contours2, -1, (0, 0, 255), 1)
    cv2.imshow("Contours", image1)
    cv2.imshow("Contoours2", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # CURRENT BEST METHOD
    # Compare HuMoments with Chi-Square method
    # Smaller value means more similar