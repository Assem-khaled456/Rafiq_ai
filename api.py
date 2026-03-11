from fastapi import FastAPI
from pydantic import BaseModel
import json

from pose  import extract_angles, compare_angles

app = FastAPI()


class AnalyzeRequest(BaseModel):
    videoUrl: str
    exerciseType: str
    expectedReps: int


class ExtractReferenceRequest(BaseModel):
    videoUrl: str
    exerciseType: str


class CompareRequest(BaseModel):
    childVideoUrl: str
    referenceJointAnglesJson: list
    referenceRepetitionCount: int


@app.post("/extract-reference")
def extract_reference(data: ExtractReferenceRequest):

    angles = extract_angles(data.videoUrl)

    return {
        "repetitionCount": 1,
        "jointAngles": angles
    }


@app.post("/compare")
def compare(data: CompareRequest):

    reference_angles = data.referenceJointAnglesJson

    patient_angles = extract_angles(data.childVideoUrl)

    accuracy, mistakes, feedback = compare_angles(
        reference_angles,
        patient_angles
    )

    return {
        "accuracyScore": accuracy,
        "repetitionCount": data.referenceRepetitionCount,
        "mistakeCount": mistakes,
        "feedback": feedback,
        "jointAngles": patient_angles
    }


@app.post("/analyze")
def analyze(data: AnalyzeRequest):

    angles = extract_angles(data.videoUrl)

    accuracy, mistakes, feedback = compare_angles(angles, angles)

    return {
        "accuracyScore": accuracy,
        "repetitionCount": data.expectedReps,
        "mistakeCount": mistakes,
        "feedback": feedback,
        "jointAngles": angles
    }