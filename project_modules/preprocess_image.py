def preprocess_image_for_ocr(image):

    # 그레이스케일 변환
    gray_image = convert_to_grayscale(image)

    # 이미지 노이즈 제거
    denoised_image = remove_noise(gray_image)

    # 대비 개선
    enhanced_conraast_image = enhance_contrast(denoised_image)

    return enhanced_conraast_image

def convert_to_grayscale(image):
    """
    이미지를 그레이스케일로 변환합니다.
    
    Args:
        image: 원본 이미지 (PIL Image 객체)
    
    Returns:
        그레이스케일로 변환된 이미지
    """
    import cv2
    import numpy as np
    
    # PIL Image를 NumPy 배열로 변환
    np_image = np.array(image)
    
    # 그레이스케일로 변환
    return cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

def remove_noise(image):
    """
    이미지의 노이즈를 제거합니다.
    
    Args:
        image: 그레이스케일 이미지
    
    Returns:
        노이즈가 제거된 이미지
    """
    import cv2
    # 가우시안 블러를 사용하여 노이즈 제거

    bg_blur = cv2.GaussianBlur(image, (5, 5), 0)
    normalized = cv2.divide(image, bg_blur, scale = 255)
    return normalized

def enhance_contrast(image):
    """
    이미지의 대비를 개선합니다.
    
    Args:
        image: 노이즈가 제거된 이미지
    
    Returns:
        대비가 개선된 이미지
    """
    import cv2
    import numpy as np
    
    # CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)






