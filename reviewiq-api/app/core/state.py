"""
ReviewIQ — App State
Global singleton holding loaded models and dataframes.
"""

import pandas as pd
from typing import Optional, Any


class AppState:
    ris_df: pd.DataFrame = pd.DataFrame()
    reviews_df: pd.DataFrame = pd.DataFrame()
    sentiment_service: Optional[Any] = None


app_state = AppState()
