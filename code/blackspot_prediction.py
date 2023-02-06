import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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

blackspot=pd.read_csv(FILE_PATH_haing_KDE_SCORE)

data = pd.read_csv(FILE_PATH)
len(data)

mild = []
sever = []
df=[]
for i in range(len(data)):
  if data.iloc[i,8] >9:
    sever.append([data.iloc[i,6], data.iloc[i,7], data.iloc[i,5]])
  elif data.iloc[i,8] =<9 and data.iloc[i,8] >=7::
    mild.append([data.iloc[i,6], data.iloc[i,7], data.iloc[i,5]])

print(len(mild), len(sever),len(df))

print(sever[:5])

#Blackspot H Calculation, Time interval wise
#time wise data segregation
list1 = [] #8-11
list2 = [] #12-15
list3 = [] #16-19
list4 = [] #20-23

# List for different time intevals

for i in range(len(data)):
  #time = int(data.iloc[i,4].split(' ')[1].split(':')[0])
  time = int(df[i][2].split(' ')[1].split(':')[0])
  if time>=8 and time<=11:
    list1.append([df[i][0], df[i][1], df[i][2]])
  elif time>=12 and time<=15:
    list2.append([df[i][0], df[i][1], df[i][2]])
  elif time>=16 and time<=19:
    list3.append([df[i][0], df[i][1], df[i][2]])
  elif time>=20 and time<=23:
    list4.append([df[i][0], df[i][1], df[i][2]])

print(len(list1),len(list2),len(list3),len(list4))

def get_lat_long(list_n,blackspot):
  # distances4 = [ [] for i in range(len(blackspot))]
  h_2d=[]
  b_name=[]
  X=[]
  Y=[]
  for i in range(len(blackspot)):
  #print(str(i))
    long1, lat1 = blackspot.iloc[i,1], blackspot.iloc[i,0]
    x=[]
    y=[]
    for j in range(len(list_n)):
      long2, lat2 = list_n[j][1], list_n[j][0]

      dis = round(haversine(long1, lat1, long2, lat2), 1)
      if dis<=0.1:
        x.append(long2)
        y.append(lat2)
    if len(x)!=0:
      b_name.append((long1,lat1))
      X.append(x)
      Y.append(y)
      H, yedges, xedges = np.histogram2d(x, y, bins=10)
      h_2d.append(H)
    #   print(H)

  return (h_2d)

h_2d_1=get_lat_long(list1,blackspot)
h_2d_2=get_lat_long(list2,blackspot)
h_2d_3=get_lat_long(list3,blackspot)
h_2d_4=get_lat_long(list4,blackspot)

######################################Severity#################################



#Blackspot H Calculation, Time interval wise
#time wise data segregation
s_list1 = [] #8-11
s_list2 = [] #12-15
s_list3 = [] #16-19
s_list4 = [] #20-23

for i in range(len(sever)):
  #time = int(data.iloc[i,4].split(' ')[1].split(':')[0])
  time = int(sever[i][2].split(' ')[1].split(':')[0])
  if time>=8 and time<=11:
    s_list1.append([sever[i][0], sever[i][1], sever[i][2]])
  elif time>=12 and time<=15:
    s_list2.append([sever[i][0], sever[i][1], sever[i][2]])
  elif time>=16 and time<=19:
    s_list3.append([sever[i][0], sever[i][1], sever[i][2]])
  elif time>=20 and time<=23:
    s_list4.append([sever[i][0], sever[i][1], sever[i][2]])

print(len(s_list1),len(s_list2),len(s_list3),len(s_list4))

# get latitude and longitude for severe points

def get_lat_long_sever(list_n,blackspot):

  h_2d=[]
  b_name=[]
  X=[]
  Y=[]
  for i in range(len(blackspot)):
  #print(str(i))
    long1, lat1 = blackspot[i][1], blackspot[i][0]
    x=[]
    y=[]
    for j in range(len(list_n)):
      long2, lat2 = list_n[j][1], list_n[j][0]

      dis = round(haversine(long1, lat1, long2, lat2), 1)
      if dis<=0.1:
        x.append(long2)
        y.append(lat2)
    if len(x)!=0:
      b_name.append((long1,lat1))
      X.append(x)
      Y.append(y)
      H, yedges, xedges = np.histogram2d(x, y, bins=10)
      h_2d.append(H)
    
  return (h_2d)

s_list1[0][0]

s_2d_1=get_lat_long_sever(list1,s_list1)
s_2d_2=get_lat_long_sever(list2,s_list2)
s_2d_3=get_lat_long_sever(list3,s_list3)
s_2d_4=get_lat_long_sever(list4,s_list4)


##################### EMD distance and corresponding lat and log ######################################

emd_1=[]
emd_2=[]
emd_3=[]
emd_4=[]
lat_long_1=[]
lat_long_2=[]
lat_long_3=[]
lat_long_4=[]


def emd_lat_long(h_2d,s_2d,thres):

  for b in range(0,len(h_2d)):
    for s in range(0, len(s_2d)):
      d = cdist(h_2d[b], s_2d[s])
      assignment = linear_sum_assignment(d)
      if((d[assignment].sum() / 10)<=thres):
        lat_long_1.append([b,s])
      emd_1.append(d[assignment].sum() / 10)

  return emd1, lat_long_1

emd_1, lat_long_1 =emd_lat_long(h_2d_1,s_2d_1,103.6)
emd_2, lat_long_2 =emd_lat_long(h_2d_1,s_2d_1,149)
emd_3, lat_long_3 =emd_lat_long(h_2d_1,s_2d_1,84.5)
emd_4, lat_long_4 =emd_lat_long(h_2d_1,s_2d_1,22.92)

print(len(emd_1),len(emd_2),len(emd_3),len(emd_4))

print(lat_long_1,'\n',lat_long_2,'\n',lat_long_3,'\n',lat_long_4)

############################## Save the new predicted point in csv ######################################
def existing_predicted_lat_long(s_list,lat_long_interval, file_name):

  black=[]
  new_pre=[]
  for x in lat_long_interval:
    black.append(x[0])
    new_pre.append(x[1])

  print(set(black))
  print(set(new_pre))


  new_data=pd.DataFrame()

  l_lat=[]
  l_long=[]
  for x in new_pre:
    l_lat.append(s_list[x][0])
    l_long.append(s_list[x][1])

  new_data["lat"]=l_lat
  new_data["long"]=l_long

  new_data.to_csv(file_name)

existing_predicted_lat_long(s_list1,lat_long_1,"pred_1.csv")
existing_predicted_lat_long(s_list2,lat_long_2,"pred_2.csv")
existing_predicted_lat_long(s_list3,lat_long_3,"pred_3.csv")
existing_predicted_lat_long(s_list4,lat_long_4,"pred_4.csv")
