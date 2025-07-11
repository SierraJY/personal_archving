# 한국어 형태소 분석기 Konlpy 통합
# Komoran, Okt 활용용
# 명사, 고유명사 위주의 키워드 추출
# 복합 명사 및 신조어 처리
# TF-IDF 점수 기반 중요도 계산

def extract_keyword_with_morpheme_analysis(text):
    """
    형태소 분석을 통해 키워드를 추출합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        추출된 키워드 목록
    """
    # 형태소 분석
    pos_tagged = morpheme_analyze(text)
    
    # 명사 추출
    nouns = extract_nouns_from_pos_taggged(pos_tagged)
    
    # 복합 명사 생성
    compound_nouns = create_compund_nouns(nouns)
    
    # 모든 키워드 후보 결합
    all_keywords = nouns + compound_nouns
    
    # 중복 제거 및 빈도수 기반 정렬
    from collections import Counter
    keyword_counts = Counter(all_keywords)
    
    # 빈도수 기준 상위 키워드 선택
    top_keywords = select_top_keywords(keyword_counts)
    
    return top_keywords

def morpheme_analyze(text):
    """
    텍스트에 대한 형태소 분석 및 품사 태깅을 수행합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        품사가 태깅된 형태소 목록
    """
    from konlpy.tag import Komoran, Okt
    
    try:
        # 우선 Komoran 사용 시도
        komoran = Komoran()
        pos_tagged = komoran.pos(text)
    except Exception as e:
        # Komoran 실패 시 Okt 사용
        okt = Okt()
        pos_tagged = okt.pos(text, norm=True, stem=True)
    
    return pos_tagged

def extract_nouns_from_pos_taggged(pos_tagged):
    """
    품사 태깅된 결과에서 명사만 추출합니다.
    
    Args:
        pos_tagged: 품사 태깅된 형태소 목록
        
    Returns:
        추출된 명사 목록
    """
    nouns = []
    
    # Komoran 태그: NNG(일반명사), NNP(고유명사)
    # Okt 태그: Noun
    for word, pos in pos_tagged:
        if pos in ['NNG', 'NNP', 'Noun'] and len(word) > 1:  # 1글자 명사는 제외
            nouns.append(word)
    
    return nouns

def create_compund_nouns(nouns):
    """
    연속된 명사를 결합하여 복합 명사를 생성합니다.
    
    Args:
        nouns: 추출된 명사 목록
        
    Returns:
        생성된 복합 명사 목록
    """
    compound_nouns = []
    
    # 연속된 명사 결합
    for i in range(len(nouns) - 1):
        compound_noun = nouns[i] + nouns[i + 1]
        compound_nouns.append(compound_noun)
    
    return compound_nouns

def select_top_keywords(keyword_counts, top_n=15):
    """
    빈도수를 기반으로 상위 키워드를 선택합니다.
    
    Args:
        keyword_counts: 단어별 빈도수 Counter 객체
        top_n: 선택할 상위 키워드 수
        
    Returns:
        선택된 상위 키워드 목록
    """
    # 빈도수 기준 정렬
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 키워드 선택
    selected_keywords = [word for word, count in sorted_keywords[:top_n]]
    
    return selected_keywords



