
# Importing the libray

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from collections import Counter
import multiprocessing


# Multiprocessing code for KDE scores

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


data=pd.read_csv(FILE_PATH)
kde = KernelDensity(kernel='gaussian', bandwidth=0.00001).fit(data)
kde_result = parrallel_score_samples(kde, data)
df["log_density"]=kde_result
df.to_csv(FILE_PATH)
print("done")

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


# The blackspot csv file here
blackspot=pd.read_csv(FILE_PATH)

# The sample data
data = pd.read_csv(FILE_PATH)

# Segregate the data based on KDE value

mild = []
sever = []

for i in range(len(data)):
  if data.iloc[i,8] >9:
    sever.append([data.iloc[i,6], data.iloc[i,7], data.iloc[i,5]])
  elif data.iloc[i,8] =<9 and data.iloc[i,8] >=7::
    mild.append([data.iloc[i,6], data.iloc[i,7], data.iloc[i,5]])


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


#list1 = 8:00-11:00 Finding the disatnce of alert from the all blackspots.

distances1 = [ [] for i in range(len(blackspot))]
for i in range(len(blackspot)):
  #print(str(i))
  long1, lat1 = blackspot.iloc[i,1], blackspot.iloc[i,0]
  for j in range(len(list1)):
    long2, lat2 = list1[j][1], list1[j][0]

    dis = round(haversine(long1, lat1, long2, lat2), 1)
    distances1[i].append(dis)


#list2
distances2 = [ [] for i in range(len(blackspot))]
for i in range(len(blackspot)):
  #print(str(i))
  long1, lat1 = blackspot.iloc[i,1], blackspot.iloc[i,0]
  for j in range(len(list2)):
    long2, lat2 = list2[j][1], list2[j][0]

    dis = round(haversine(long1, lat1, long2, lat2), 1)
    distances2[i].append(dis)

#list3
distances3 = [ [] for i in range(len(blackspot))]
for i in range(len(blackspot)):
  #print(str(i))
  long1, lat1 = blackspot.iloc[i,1], blackspot.iloc[i,0]
  for j in range(len(list3)):
    long2, lat2 = list3[j][1], list3[j][0]

    dis = round(haversine(long1, lat1, long2, lat2), 1)
    distances3[i].append(dis)


#list4
distances4 = [ [] for i in range(len(blackspot))]
for i in range(len(blackspot)):
  #print(str(i))
  long1, lat1 = blackspot.iloc[i,1], blackspot.iloc[i,0]
  for j in range(len(list4)):
    long2, lat2 = list4[j][1], list4[j][0]

    dis = round(haversine(long1, lat1, long2, lat2), 1)
    distances4[i].append(dis)

print(len(list1), len(list2), len(list3), len(list4))

print(list1)



k = np.unique(distances3[0], return_counts=True)
print(k[0], '\n', k[1])

thresholds = np.arange(0, 1, 0.05).tolist()
# thresholds =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
recall1 = []
#distance1-sever
print('sever/8-11')
for thr in thresholds:
  print('THRESHOLD: ', thr)
  counts = [ 0 for i in range(len(distances1))]
  for i in range(len(distances1)):
    for j in range(len(distances1[i])):
      dis = distances1[i][j]
      if dis<thr:
        counts[i] += 1
  print(counts, '\n')     
  recall1.append(sum(i > 0 for i in counts)/27) 


 #distance2-sever

recall2 = []
print('sever/12-15')
for thr in thresholds:
  print('THRESHOLD: ', thr)
  counts = [ 0 for i in range(len(distances2))]
  for i in range(len(distances2)):
    for j in range(len(distances2[i])):
      dis = distances2[i][j]
      if dis<thr:
        counts[i] += 1
  print(counts, '\n') 
  print(sum(i > 0 for i in counts))
  recall2.append(sum(i > 0 for i in counts)/27) 

#distance3-mild
recall3 = []
print('sever/16-19')
for thr in thresholds:
  print('THRESHOLD: ', thr)
  counts = [ 0 for i in range(len(distances3))]
  for i in range(len(distances3)):
    for j in range(len(distances3[i])):
      dis = distances3[i][j]
      if dis<thr:
        counts[i] += 1
  print(counts, '\n') 
  print(sum(i > 0 for i in counts))
  recall3.append(sum(i > 0 for i in counts)/27) 

#distance4-mild
recall4 = []
print('sever/20-23')
for thr in thresholds:
  print('THRESHOLD: ', thr)
  counts = [ 0 for i in range(len(distances4))]
  for i in range(len(distances4)):
    for j in range(len(distances4[i])):
      dis = distances4[i][j]
      if dis<thr:
        counts[i] += 1
  print(counts, '\n') 
  print(sum(i > 0 for i in counts))
  recall4.append(sum(i > 0 for i in counts)/27)

print(recall1,'\n', recall2,'\n', recall3,'\n',recall4)

# Code to make plot recall-d

plt.figure()
plt.plot(thresholds, recall1, label="08:00-11:00")
plt.plot(thresholds, recall2, label="12:00-15:00")
plt.plot(thresholds, recall3, label="16:00-19:00")
plt.plot(thresholds, recall4, label="20:00-23:00")

plt.xlabel("Distance(d)")
plt.ylabel("Recall-d")

plt.legend(loc="lower right")
plt.savefig("mild_step_05.jpg",dpi=200)
plt.show()

###################### Similar code can be used for Mild + severe ##########################
