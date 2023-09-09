import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from collections import Counter
import multiprocessing


data = pd.read_csv(FILE_WITH_KDE_LOG_SCORE)

blackspot=pd.read_csv(BLACKSPOT_FILE)
blackspot['Lat'] = blackspot['Lat'].astype(float)
blackspot['Long'] = blackspot['Long'].astype(float)

# Calculating the haversine distance
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

sever = []

for i in range(len(data)):
    if data.iloc[i,7] >=(data['log_density'].quantile(0.75)):
        sever.append([data.iloc[i,Lat_Column], data.iloc[i,Long_Column], data.iloc[i,Date_Time_Column]])

# Segregating data based on Time-Interval

list1 = [] #8-11
list2 = [] #12-15
list3 = [] #16-19
list4 = [] #20-23

for i in range(len(sever)):

  time = int(sever[i][2].split(' ')[1].split(':')[0])
  if time>=8 and time<=11:
    list1.append([sever[i][0], sever[i][1], sever[i][2]])
  elif time>=12 and time<=15:
    list2.append([sever[i][0], sever[i][1], sever[i][2]])
  elif time>=16 and time<=19:
    list3.append([sever[i][0], sever[i][1], sever[i][2]])
  elif time>=20 and time<=23:
    list4.append([sever[i][0], sever[i][1], sever[i][2]])

def calculate_distances(df, points_list):
    distances = [ [] for _ in range(len(df))]
    for i in range(len(df)):
        long1, lat1 = df.iloc[i, 1], df.iloc[i, 0]
        for j in range(len(points_list)):
            long2, lat2 = points_list[j][1], points_list[j][0]
            dis = round(haversine(long1, lat1, long2, lat2), 1)
            distances[i].append(dis)
    return distances

distances1 = calculate_distances(blackspot, list1)
distances2 = calculate_distances(blackspot, list2)
distances3 = calculate_distances(blackspot, list3)
distances4 = calculate_distances(blackspot, list4)

recall_all=[]
def calculate_recall_for_threshold_range(distances, thresholds):
    recalls = []
    for thr in thresholds:
        print('THRESHOLD:', thr)
        counts = [0] * len(distances)
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                dis = distances[i][j]
                if dis < thr:
                    counts[i] += 1

        recall = (sum(i > 0 for i in counts)/47)
        recalls.append(recall)

    return recalls

thresholds = np.arange(0, 1, 0.05).tolist()

recall_all.append(calculate_recall_for_threshold_range(distances1, thresholds))
recall_all.append(calculate_recall_for_threshold_range(distances2, thresholds))
recall_all.append(calculate_recall_for_threshold_range(distances3, thresholds))
recall_all.append(calculate_recall_for_threshold_range(distances4, thresholds))

#################################################### Recall-d Plot ###################################################################

plt.figure()
plt.plot(thresholds, recall_all[0], label="08:00-11:00")
plt.plot(thresholds, recall_all[1], label="12:00-15:00")
plt.plot(thresholds, recall_all[2], label="16:00-19:00")
plt.plot(thresholds, recall_all[3], label="20:00-23:00")

plt.xlabel("Distance(d)")
plt.ylabel("Recall-d")

plt.legend(loc="lower right")
plt.savefig("PLOT_NAME",dpi=200)

my_array = np.array(recall_all)

# Save the NumPy array to a .npy file
np.save('SAVE_FOR_LATER_USE', my_array)
