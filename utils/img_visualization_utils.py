import numpy as np  
import cv2  

def draw_anns_on_image(image, anns, draw_polygon=False, line_thickness=1, alpha = 0.3):  
    """  
    Draw instance segmentation masks as semi-transparent polygons on the image.  

    Parameters:  
    - image: The original image to draw on (numpy array).  
    - anns: List of annotation dictionaries, each containing 'area' and 'segmentation'. For e.g.:
    {'segmentation': array([[False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        [False, False, False, ...,  True,  True,  True],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]]),
    'area': 9070}
    - draw_polygon: Boolean flag to draw filled polygons (default is False).  
    - line_thickness: Thickness of the polygon outlines.  
    """  

    if len(anns) == 0:  
        return image  

    # Create a copy of the input image to draw on  
    output_image = image.copy()  

    # Create a temporary mask for the polygons  
    temp_mask = np.zeros_like(image)     

    for ann in anns:  
        m = ann['segmentation']  # Assume this is a binary mask (numpy array with shape (height, width))  
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

        # Generate a random color for the polygon  
        color_mask_bgr = tuple(np.random.randint(0, 256, size=3).tolist())  # Random color in BGR  

        for contour in contours:  
            # Draw outlines (polylines) for all contours  
            cv2.polylines(output_image, [contour], isClosed=True, color=color_mask_bgr, thickness=line_thickness)  

            if draw_polygon:  
                # Fill the polygon in the temporary mask  
                cv2.fillPoly(temp_mask, [contour], color_mask_bgr)  

    # Blend the temporary mask with the output image using cv2.addWeighted  
    if draw_polygon:  
        output_image = cv2.addWeighted(temp_mask, alpha, output_image, 1 - alpha, 0)  

    return output_image  