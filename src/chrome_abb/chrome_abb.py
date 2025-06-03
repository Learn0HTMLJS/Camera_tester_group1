import cv2
import numpy as np

def detect_chromatic_aberration(image_path, window_size=15, threshold=0.7):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    b, g, r = cv2.split(image)

    def channel_diff(ch1, ch2, shift=1):
        h, w = ch1.shape
        ch1_cropped = ch1[shift:h-shift, shift:w-shift]
        ch2_shifted = ch2[shift:h-shift, shift:w-shift]
        diff = cv2.absdiff(ch1_cropped, ch2_shifted)
        return np.mean(diff)

    rg_diff = channel_diff(r, g)
    bg_diff = channel_diff(b, g)

    total_diff = (rg_diff + bg_diff) / 2
        
    result = [
        {'title':'Red-Green difference', 'data': f"{rg_diff:.2f}", 'cmap': None, 'type':'text'},
        {'title':'Red-Green difference', 'data': f"{bg_diff:.2f}", 'cmap': None, 'type':'text'},
    ]
    
    if total_diff > threshold:
        result.append({'title':'Chromatic aberration detected', 'data': f"Intensity: {total_diff:.2f}", 'cmap': None, 'type':'text'})
    else:
        result.append({'title':'No significant chromatic aberration', 'data': f"Intensity: {total_diff:.2f}", 'cmap': None, 'type':'text'})

    return result