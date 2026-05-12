import numpy as np


class EDAIndexAnalysis:
    '''
    Spectral index helpers with safe fallbacks. Assumes that we only have 3-bands, no NRI or SWIR bands.
    '''
    def __init__(self, config=None, logger=None):
        self.config = config or {}
        self.logger = logger

    def _image_to_norm_BGR_float(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Normalises the image to the range [0.0, 1.0] and splits it into B, G, R channels.
        Also converts it to float32.
        Returns:
            B: np.ndarray, float32, normalised B channel
            G: np.ndarray, float32, normalised G channel
            R: np.ndarray, float32, normalised R channel
        '''
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Image must have at least 3 channels.")
        
        image = image[..., :3].astype(np.float32, copy=False)
        if image.max() > 1.0:
            if image.max() <= 255.0:
                image /= 255.0
            elif image.max() <= 65535.0:
                image /= 65535.0
            else:
                image /= image.max() 

        B, G, R = image[..., 0], image[..., 1], image[..., 2]
        
        return B, G, R
    
    def _safe_ratio(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray: 
        '''
        Guards against division by zero and returns NaN for invalid values.
        '''
        eps = 1e-6
        num = numerator.astype(np.float32, copy=False) 
        denom = denominator.astype(np.float32, copy=False) 
        result = np.full_like(num, np.nan, dtype=np.float32) 
        valid = np.abs(denom) > eps 
        result[valid] = num[valid] / denom[valid] 

        return result
    

    # --- Vegetation Indices --- #
    def rgbvi(self, B, G, R):
        '''
        Computes the RGB Vegetation Index.
        '''
        numerator = (G ** 2) - (R * B)
        denominator = (G ** 2) + (R * B)
        rgbvi = self._safe_ratio(numerator, denominator)
        
        return rgbvi

    def vari(self, B, G, R):
        '''
        Computes the Visible Atmospherically Resistant Index.
        '''
        numerator = G - R
        denominator = G + R - B
        vari = self._safe_ratio(numerator, denominator)
        
        return vari
    
    def exg(self, B, G, R):
        '''
        Computes the Excess Green Index.
        '''

        exg = (2 * G - (R + B)).astype(np.float32, copy=False)
        
        return exg
    
    def gli(self, B, G, R):
        '''
        Computes the Green Leaf Index.
        '''
        numerator = (2 * G) - R - B
        denominator = (2 * G) + R + B
        gli = self._safe_ratio(numerator, denominator)
        
        return gli
    
    def mgrvi(self, B, G, R): # opting to retain the B input even though it is unused so that the method signature is consistent with the other indices. same for below.
        '''
        Computes the Modified Green Red Vegetation Index.
        '''
        numerator = (G ** 2) - (R ** 2)
        denominator = (G ** 2) + (R ** 2)
        mgrvi = self._safe_ratio(numerator, denominator)
        
        return mgrvi

    # --- Water Indices --- #

    # NOTE: these are not really true water indices, as my research implies that all water indices require NIR or minimally SWIR bands for analysis. These are just proxies I made that weight the amount of blue in the image against red and green.
    def bri(self, B, G, R):
        '''
        Computes the Blue Red Index.
        '''
        numerator = B - R
        denominator = B + R
        bri = self._safe_ratio(numerator, denominator)
        
        return bri
    
    def bgi(self, B, G, R):
        '''
        Computes the Blue Green Index.
        '''
        numerator = B - G
        denominator = B + G
        bgi = self._safe_ratio(numerator, denominator)
        
        return bgi