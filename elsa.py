from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class StudentData(BaseModel):
    Marital_status: int
    Application_mode: int
    Application_order: int
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Nacionality: int
    Mothers_qualification: int
    Fathers_qualification: int
    Mothers_occupation: int
    Fathers_occupation: int
    Displaced: int
    Educational_special_needs: int
    Debtor: int
    Tuition_fees_up_to_date: int
    Gender: int
    Scholarship_holder: int
    Age_at_enrollment: int
    International: int
    Curricular_units_1st_sem_credited: int
    Curricular_units_1st_sem_enrolled: int
    Curricular_units_1st_sem_evaluations: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_1st_sem_grade: float
    Curricular_units_1st_sem_without_evaluations: int
    Curricular_units_2nd_sem_credited: int
    Curricular_units_2nd_sem_enrolled: int
    Curricular_units_2nd_sem_evaluations: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_2nd_sem_grade: float
    Curricular_units_2nd_sem_without_evaluations: int
    Unemployment_rate: float
    Inflation_rate: float
    GDP: float
   

class StudentList(BaseModel):
    data: List[StudentData]

@app.post("/predict/")
def predict_dropout(student_list: StudentList):
    if not student_list.data:
        raise HTTPException(status_code=400, detail="Empty student list")
    
    predictions = []
    for student in student_list.data:
        # Dummy prediction logic (replace with model prediction)
        prediction = "Graduate" if student.Curricular_units_1st_sem_grade > 10 else "Dropout"
        predictions.append({"student": student, "prediction": prediction})
    
    return {"predictions": predictions}
