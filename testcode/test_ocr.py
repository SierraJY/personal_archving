import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse
import os

# PyTorch 임포트 추가 (libgomp.so.1 라이브러리 경로 문제 해결)
import torch

from paddleocr import PaddleOCR
from project_modules.preprocess_image import preprocess_image_for_ocr

# 테스트 이미지 디렉토리 설정
TEST_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testing_data')

# 한글 폰트 설정 - 나눔고딕 폰트 하나만 사용
import matplotlib.font_manager as fm
# 나눔고딕 폰트 경로 직접 지정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# 폰트 설정
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    print(f"한글 폰트 설정 완료: {font_path}")
else:
    print(f"경고: 지정된 폰트를 찾을 수 없습니다: {font_path}")

def process_image(image_path, ocr, use_preprocess=True):
    """
    이미지를 처리하고 OCR 결과를 반환합니다.
    
    Args:
        image_path: 이미지 파일 경로
        ocr: PaddleOCR 객체
        use_preprocess: 전처리 적용 여부
        
    Returns:
        원본 이미지, 전처리된 이미지, 원본 OCR 결과, 전처리된 OCR 결과
    """
    # 이미지 로드
    image = Image.open(image_path)
    
    # 원본 이미지에 대한 OCR
    original_result = ocr.ocr(np.array(image))
    
    # 이미지 전처리
    if use_preprocess:
        processed_image = preprocess_image_for_ocr(image)
        # 전처리된 이미지에 대한 OCR
        processed_result = ocr.ocr(np.array(processed_image))
    else:
        processed_image = image
        processed_result = original_result
    
    return image, processed_image, original_result, processed_result

def visualize_ocr_results(image_path, ocr, use_preprocess=True, save_path=None):
    """
    이미지와 OCR 결과를 시각화합니다.
    
    Args:
        image_path: 이미지 파일 경로
        ocr: PaddleOCR 객체
        use_preprocess: 전처리 적용 여부
        save_path: 결과 저장 경로 (None인 경우 화면에 표시)
    """
    # 이미지 처리 및 OCR
    image, processed_image, original_result, processed_result = process_image(image_path, ocr, use_preprocess)
    
    # 결과 시각화
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])
    
    # 원본 이미지
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('원본 이미지')
    ax1.axis('off')
    
    # 원본 이미지 OCR 결과
    ax2 = plt.subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('원본 이미지 OCR 결과')
    
    # OCR 결과 텍스트 표시
    result_text = "=== 원본 이미지 OCR 결과 ===\n"
    if original_result:
        for line in original_result[0]:
            text, confidence = line[1]
            result_text += f"{text} (신뢰도: {confidence:.2f})\n"
    
    ax2.text(0, 0.5, result_text, fontsize=10, verticalalignment='center', wrap=True)
    
    # 전처리된 이미지
    ax3 = plt.subplot(gs[1, 0])
    ax3.imshow(processed_image)
    ax3.set_title('전처리된 이미지')
    ax3.axis('off')
    
    # 전처리된 이미지 OCR 결과
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('전처리된 이미지 OCR 결과')
    
    # OCR 결과 텍스트 표시
    result_text = "=== 전처리된 이미지 OCR 결과 ===\n"
    if processed_result:
        for line in processed_result[0]:
            text, confidence = line[1]
            result_text += f"{text} (신뢰도: {confidence:.2f})\n"
    
    ax4.text(0, 0.5, result_text, fontsize=10, verticalalignment='center', wrap=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"결과가 {save_path}에 저장되었습니다.")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='이미지 OCR 테스트')
    parser.add_argument('image_path', nargs='?', help='처리할 이미지 파일 경로')
    parser.add_argument('--no-preprocess', action='store_true', help='이미지 전처리를 건너뜁니다')
    parser.add_argument('--use-angle-cls', action='store_true', help='텍스트 방향 감지 기능을 활성화합니다')
    parser.add_argument('--lang', type=str, default='korean', help='OCR 언어 설정')
    parser.add_argument('--save', type=str, help='결과를 저장할 파일 경로')
    parser.add_argument('--all', action='store_true', help='테스트 디렉토리의 모든 이미지를 처리합니다')
    
    args = parser.parse_args()
    
    # OCR 모델 로드
    print("OCR 모델 로드 중...")
    ocr = PaddleOCR(use_angle_cls=args.use_angle_cls, lang=args.lang)
    print("OCR 모델 로드 완료")
    
    if args.all:
        # 테스트 디렉토리의 모든 이미지 처리
        if not os.path.exists(TEST_IMAGE_DIR):
            os.makedirs(TEST_IMAGE_DIR)
            print(f"테스트 이미지 디렉토리가 생성되었습니다: {TEST_IMAGE_DIR}")
            print("이미지 파일을 해당 디렉토리에 추가한 후 다시 실행해주세요.")
            return
        
        image_files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"테스트 디렉토리에 이미지 파일이 없습니다: {TEST_IMAGE_DIR}")
            return
        
        # 결과 저장 디렉토리
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        for image_file in image_files:
            image_path = os.path.join(TEST_IMAGE_DIR, image_file)
            print(f"처리 중: {image_file}")
            
            # 결과 저장 경로
            save_path = os.path.join(results_dir, f"{os.path.splitext(image_file)[0]}_result.png")
            
            try:
                visualize_ocr_results(
                    image_path, 
                    ocr, 
                    use_preprocess=not args.no_preprocess,
                    save_path=save_path
                )
            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {e}")
    
    elif args.image_path:
        # 단일 이미지 처리
        visualize_ocr_results(
            args.image_path, 
            ocr, 
            use_preprocess=not args.no_preprocess,
            save_path=args.save
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 