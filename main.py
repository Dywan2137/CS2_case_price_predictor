import numpy as np
import pandas as pd

from config import DATASET
from data import resample_weekly, load_daily
from features import build_matrix


daily = load_daily(DATASET)
data_weekly = resample_weekly(daily)
data_weekly = build_matrix(data_weekly)
print(data_weekly)