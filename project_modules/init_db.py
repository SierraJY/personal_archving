from datetime import datetime
from sqlmodel import Field, SQLModel, create_engine
from typing import Optional

# 데이터베이스 모델
class Document(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    doc_type: str
    content: str
    summary: str
    keywords: str
    structured_data: str  # JSON 형태로 저장
    upload_date: datetime = Field(default_factory=datetime.now)
    image_data: bytes
    embedding: Optional[str] = None  # 벡터를 JSON으로 저장

def intialize_db():
    # 데이터베이스 초기화
    engine = create_engine("sqlite:///archive.db")
    SQLModel.metadata.create_all(engine)

    return engine
