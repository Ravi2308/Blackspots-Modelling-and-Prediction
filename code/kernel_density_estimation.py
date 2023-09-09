
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from collections import Counter
import multiprocessing


def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

data=pd.read_csv(FILE_PATH)

lat_long_data = np.radians(data[['Lat', 'Long']])
kde = KernelDensity(kernel='gaussian', bandwidth=0.000001).fit(lat_long_data)

kde_result = parrallel_score_samples(kde, lat_long_data)

data["log_density"]=kde_result
data.to_csv(SAVE_THE_FILE_WITH_KDE)
