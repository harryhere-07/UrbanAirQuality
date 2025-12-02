import cv2
import numpy as np

class HazeEstimator:
    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size

    def get_dark_channel(self, image):
        """
        Calculates the dark channel of the image.
        The dark channel is the minimum value across RGB channels in a local patch.
        """
        # Minimum across RGB channels
        min_channel = np.min(image, axis=2)
        
        # Minimum filter (erosion) to find local minimum in the patch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        dark_channel = cv2.morphologyEx(min_channel, cv2.MORPH_ERODE, kernel)
        
        return dark_channel

    def estimate_atmospheric_light(self, image, dark_channel):
        """
        Estimates the atmospheric light (A) - the brightest pixels in the dark channel.
        This represents the ambient light scattered by the atmosphere.
        """
        h, w = image.shape[:2]
        num_pixels = h * w
        num_brightest = int(max(num_pixels * 0.001, 1)) # Top 0.1%

        dark_vec = dark_channel.reshape(num_pixels)
        image_vec = image.reshape(num_pixels, 3)

        indices = dark_vec.argsort()[::-1] # Sort descending
        brightest_indices = indices[:num_brightest]

        # Mean of the brightest pixels in the original image is the Atmospheric Light
        A = np.mean(image_vec[brightest_indices], axis=0)
        return A

    def get_transmission_map(self, image, A):
        """
        Calculates the Transmission Map (t).
        t = 1 - omega * min(I / A)
        High t = Clear view. Low t = Haze/Pollution.
        """
        omega = 0.95 # Retain a small amount of haze for depth perception logic
        
        norm_img = image.astype(np.float64) / A
        dark_ch_norm = self.get_dark_channel(norm_img)
        
        transmission = 1 - omega * dark_ch_norm
        return transmission

    def calculate_haze_score(self, transmission_map):
        """
        Returns a score between 0 and 100.
        0 = Clear, 100 = Heavy Pollution.
        Based on the inverse mean of the transmission map.
        """
        mean_transmission = np.mean(transmission_map)
        # Invert: Lower transmission means higher pollution
        pollution_score = (1.0 - mean_transmission) * 100
        return max(0, min(100, pollution_score))