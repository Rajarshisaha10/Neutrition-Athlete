from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    role = Column(String(64), nullable=True)


class InjuryAssessment(Base):
    __tablename__ = "injury_assessments"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=True)
    questionnaire_json = Column(Text, nullable=False)
    risk_model_output_json = Column(Text, nullable=True)
    prediction_json = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    model_version = Column(String(128), nullable=False)
    insufficient_confidence = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User")
    confirmation = relationship("ConfirmedDiagnosis", back_populates="assessment", uselist=False)


class ConfirmedDiagnosis(Base):
    __tablename__ = "confirmed_diagnoses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    assessment_id = Column(String(64), ForeignKey("injury_assessments.id"), nullable=False, index=True)
    confirmed_diagnosis = Column(String(128), nullable=False)
    clinician_type = Column(String(128), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    assessment = relationship("InjuryAssessment", back_populates="confirmation")


class ModelFeedback(Base):
    __tablename__ = "model_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    assessment_id = Column(String(64), ForeignKey("injury_assessments.id"), nullable=False, index=True)
    rating = Column(Integer, nullable=True)
    helpful = Column(Boolean, nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    assessment_id = Column(String(64), ForeignKey("injury_assessments.id"), nullable=False, index=True)
    reason = Column(String(255), nullable=False)
    priority = Column(Integer, default=5, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version = Column(String(128), primary_key=True)
    model_path = Column(String(512), nullable=False)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
