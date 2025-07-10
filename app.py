import streamlit as st

from sqlmodel import Session, select

from project_modules.init_db import Document, intialize_db
from project_modules.load_models import load_models
from project_modules.process_document import *

engine = intialize_db()

# Streamlit UI
st.title("AI 아카이브 시스템")

# 모델 로드
with st.spinner("AI 모델 로딩 중..."):
    models = load_models()

# 탭 생성
tab1, tab2, tab3 = st.tabs(["문서 업로드", "문서 검색", "문서 목록"])

if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'doc_results' not in st.session_state:
    st.session_state.doc_results = None

# 문서 업로드 탭
with tab1:
    uploaded_file = st.file_uploader("문서를 업로드하세요", type=['png', 'jpg', 'jpeg'])

    # 새 파일인지 확인
    if uploaded_file is not None and uploaded_file != st.session_state.processed_file:
        st.session_state.processed_file = uploaded_file
        st.session_state.processing_complete = False
    
    # 처리되지 않은 파일만 처리
    if uploaded_file is not None and not st.session_state.processing_complete:
        with st.spinner("문서 처리 중..."):
            doc_type, content, summary, keywords, structured_data, img_data, embedding = process_document(
                uploaded_file, models
            )
            st.session_state.processing_complete = True
            # 결과를 세션에 저장
            st.session_state.doc_results = {
                'doc_type': doc_type,
                'content': content,
                'summary': summary,
                'keywords': keywords,
                'structured_data': structured_data,
                'img_data': img_data,
                'embedding': embedding
            }
    
    # 처리 완료된 결과 표시
    if uploaded_file is not None and st.session_state.doc_results is not None:
        results = st.session_state.doc_results
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="업로드된 문서", use_container_width=True)
        
        with col2:
            st.write(f"**문서 유형:** {results['doc_type']}")
            st.write(f"**요약:** {results['summary']}")
            st.write(f"**키워드:** {results['keywords']}")
            
            if results['structured_data']:
                st.write("**추출된 정보:**")
                for key, value in results['structured_data'].items():
                    st.write(f"- {key}: {value}")
            
            if st.button("저장"):
                with Session(engine) as session:
                    doc = Document(
                        filename=uploaded_file.name,
                        doc_type=results['doc_type'],
                        content=results['content'],
                        summary=results['summary'],
                        keywords=results['keywords'],
                        structured_data=json.dumps(results['structured_data'], ensure_ascii=False),
                        image_data=results['img_data'],
                        embedding=json.dumps(results['embedding'])
                    )
                    session.add(doc)
                    session.commit()
                st.success("문서가 저장되었습니다!")
                st.session_state.processed_file = None
                st.session_state.processing_complete = False
                st.session_state.doc_results = None

# 문서 검색 탭
with tab2:
    search_query = st.text_input("검색어를 입력하세요 (예: 커피 영수증)")
    search_method = st.radio("검색 방법", ["벡터 유사도 검색", "키워드 검색"])
    
    if st.button("검색", key='search_button'):
        with Session(engine) as session:
            if search_method == "벡터 유사도 검색":
                results = search_by_similarity(
                    search_query, 
                    models[9],  # embedding_model
                    session
                )
            else:
                # 키워드 검색
                statement = select(Document).where(
                    Document.keywords.contains(search_query) | 
                    Document.summary.contains(search_query) |
                    Document.doc_type.contains(search_query)
                )
                results = session.exec(statement).all()
            
            if results:
                print_result_list(results)
            else:
                st.info("검색 결과가 없습니다.")

# 문서 목록 탭
with tab3:
    with Session(engine) as session:
        statement = select(Document)
        results = session.exec(statement).all()
        if results:
            print_result_list(results)
    