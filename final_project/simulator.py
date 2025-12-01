import random
from datetime import datetime
import pandas as pd

def generate_vitals(base_hr=75):
    hr = int(random.gauss(base_hr, 4))
    spo2 = round(random.gauss(97, 0.8),1)
    temp = round(random.gauss(36.6,0.3),1)
    syst = int(random.gauss(120,7))
    diast = int(random.gauss(80,5))
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'heart_rate': max(40, min(180, hr)),
        'spo2': max(70, min(100, spo2)),
        'temperature': temp,
        'systolic': syst,
        'diastolic': diast
    }

def generate_stream(n=50, base_hr=75):
    rows = [generate_vitals(base_hr) for _ in range(n)]
    return pd.DataFrame(rows)
PY
