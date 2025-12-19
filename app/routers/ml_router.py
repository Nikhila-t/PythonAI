from fastapi import APIRouter, HTTPException
from app.services.data_service import load_customer_tickets
from app.services.ml_service import train_model, predict_team_to_assign
from app.models.schema import Team_Prediction_Request, Team_Prediction_Response

router = APIRouter(prefix = "/ml", tags = ["Machine Learning and Prediction"])

## Endpoint to train the model
## Train the ML model on sample csv/excel file located in data folder
## Logic -> 1. Retriev the DataFrame composed of cleaned data from the raw file
##          2. Train the model 
##                   a. preprocessors
##                            i. TF-IDF for text vectorisation
##                           ii. OneHotEncoder for categorical fields
##                   b. Logistic Regressor as classifier
## returns metrics from the model training output

@router.post("/train-sample")
def train_on_sampledata_endpoint():
    try:
        try:
            df = load_customer_tickets("data/customer_tickets.xlsx")
        except Exception as e:
            raise HTTPException(status_code = 400, detail = f"Failed to process the excel: {e}")
        
        metrics = train_model(df)

        return {
            "status":"ok",
            "metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Training failed: {e}")
    
## Endpoint to predict the team for the single ticket payload
@router.post("/predict-team", response_model = Team_Prediction_Response)
def predict_team(payload: Team_Prediction_Request):
    try:
        payload_dict = payload.model_dump() 
        predicted = predict_team_to_assign(payload_dict)

        return Team_Prediction_Response(assigned_team = predicted["predicted_team"], confidence_score=predicted["confidence"])
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Prediction failed: {e}")