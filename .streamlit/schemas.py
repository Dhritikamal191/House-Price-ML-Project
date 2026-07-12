from pydantic import BaseModel
from typing import Dict, Any

class HouseInput(BaseModel):
      features: Dict[str, Any]
