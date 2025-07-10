from streamlit import cache_resource

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForImageClassification, AutoProcessor, 
    AutoTokenizer, AutoModelForSeq2SeqLM,
    VisionEncoderDecoderModel, LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
)
from paddleocr import PaddleOCR

# 모델 로드
# @streamlit.cache_resource
# 함수의 반환값(주로 리소스 객체, 예: 모델, DB 연결 등)을 캐싱
# 앱이 다시 실행되더라도 해당 리소스를 재사용할 수 있게 함
# 이 데코레이터를 사용하면, 함수가 처음 호출될 때만 실제로 실행
# 이후에는 캐시에 저장된 결과를 반환
# 주로 모델 로딩, 대용량 객체 생성 등 비용이 큰 작업에 사용
@cache_resource
def load_models():
    # DiT 문서 분류
    dit_processor = AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    dit_model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    
    # OCR
    ocr = PaddleOCR(lang='korean')
    
    # Donut (영수증 전용)
    donut_processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    
    # LayoutLMv3
    layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    layout_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    
    # 텍스트 요약
    summarizer_tokenizer = AutoTokenizer.from_pretrained("gangyeolkim/kobart-korean-summarizer-v2")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("gangyeolkim/kobart-korean-summarizer-v2")
    
    # 임베딩 모델 (벡터 검색용)
    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    return (dit_processor, dit_model, ocr, donut_processor, donut_model, 
            layout_processor, layout_model, summarizer_tokenizer, summarizer_model,
            embedding_model)