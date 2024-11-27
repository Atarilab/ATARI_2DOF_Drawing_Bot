from drawing_bot_api.trajectory_optimizer.camera import Camera
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

class ImageProcessor:
    def __init__(self):
        self._camera = Camera()
        self._image_counter = 0

    def save_image(self, image, directory, type):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(_script_dir, f'images/{directory}/{type}_{str(self._image_counter)}.png')
        cv2.imwrite(_path, image)
        print(f'Saved {type} to {_path}')
    
    def simplify(self, image):
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
        

    def __call__(self, template):
        # retrieve both images
        _template = template
        _drawing = self._camera()
        self._image_counter += 1

        # save both images in original form
        self.save_image(_drawing, 'original', 'drawing')
        self.save_image(_template, 'original', 'template')

        # turn to inverteed binary black and white image
        _simpl_drawing = self.simplify(_drawing)
        _grey_drawing = _simpl_drawing#cv2.cvtColor(_simpl_drawing, cv2.COLOR_BGR2GRAY)
        _, _inv_drawing = cv2.threshold(_grey_drawing, 127, 255, cv2.THRESH_BINARY)
        _inv_template = self.simplify(_template)

        # save edited images
        self.save_image(_inv_drawing, 'simplified', 'drawing')
        self.save_image(_inv_template, 'simplified', 'template')

        # get contours
        _contours_template, _ = cv2.findContours(_inv_template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _contours_drawing, _ = cv2.findContours(_inv_drawing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # calc moments
        _moments_template = cv2.moments(_contours_template[0])
        _moments_drawing = cv2.moments(_contours_drawing[0])

        # Calculate Hu Moments
        _hu_moments_template = cv2.HuMoments(_moments_template).flatten()
        _hu_moments_drawing = cv2.HuMoments(_moments_drawing).flatten()

        _hu_moments_template = np.array(_hu_moments_template, dtype=np.float32)
        _hu_moments_drawing = np.array(_hu_moments_drawing, dtype=np.float32)

        # Compare using correlation
        similarity = cv2.compareHist(_hu_moments_template, _hu_moments_drawing, cv2.HISTCMP_CHISQR)
        print(f"Hu Moment Similarity: {similarity*100}")
        
        # save images with contours
        cv2.drawContours(_drawing, _contours_drawing, -1, (0, 0, 255), 1)
        cv2.drawContours(_inv_template, _contours_template, -1, (0, 0, 255), 1)
        self.save_image(_drawing, 'analysed', 'drawing')
        self.save_image(_inv_template, 'analysed', 'template')

        return similarity


if __name__ == '__main__':

    img_proc = ImageProcessor()
    #img_proc(None)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _path_template = os.path.join(_script_dir, f'images/test/plot_image_1.jpg')
    _path_drawing = os.path.join(_script_dir, f'images/test/real_image_1.jpg')

    image1 = cv2.imread(_path_template)
    image2 = cv2.imread(_path_drawing)

    invert2 = img_proc.simplify(image2)
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
