from pydantic import BaseModel, Field
from typing import Optional

## Schema for to-be-assigned_team prediction request
class Team_Prediction_Request(BaseModel):
    ticket_subject: str = Field(..., example="Unable to login to account")
    ticket_body: str = Field(..., example="User reports login failure with error code 401")
    product_module: Optional[str] = Field("Unknown", example="Authentication")
    customer_tier: Optional[str] = Field("Unknown", example="Gold")
    priority: Optional[str] = Field("Unknown", example="High")

## Schema for to-be-assigned_team prediction response
class Team_Prediction_Response(BaseModel):
    assigned_team: str
    confidence_score : float