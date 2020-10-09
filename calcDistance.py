# %%
print("hello")
import pandas
import numpy as np
def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    dist = 3958 * dist #6367 for distance in KM for miles use 3958
    return dist

MSOA_centroids = pandas.read_csv('~/Downloads/MSOA_2011_EW_PWC_COORD_V2.CSV')
MSOA_pop       = pandas.read_csv('~/Downloads/MSOA_pop.csv')
HE_centroids = pandas.read_csv('~/Downloads/learning-providers-plus.csv')
covidCases = pandas.read_csv('~/Downloads/MSOAs_latest.csv')
HE_lon = HE_centroids['LONGITUDE']
HE_lat = HE_centroids['LATITUDE']

allDist = np.zeros((len(MSOA_centroids.index),len(HE_centroids.index)))
print(len(MSOA_centroids.index))
print(len(HE_centroids.index))

for index, row in MSOA_centroids.iterrows():
    this_MSOA_lon = row['LONGITUDE']
    this_MSOA_lat = row['LATITUDE']

    theseDist = haversine(this_MSOA_lon,this_MSOA_lat,HE_lon,HE_lat)
    allDist[index,] = theseDist

dist2Uni = np.nanmin(allDist,axis=1)
MSOA_centroids.insert(2,"dist2Uni",dist2Uni)
# %%
import matplotlib.pyplot as plt

merged = pandas.merge(left=MSOA_centroids, right=covidCases, left_on='MSOA11CD', right_on='msoa11_cd')
merged = pandas.merge(left=merged,right=MSOA_pop,left_on='MSOA11CD',right_on='MSOA Code')
merged = merged.replace(-99, 0)
merged = merged.sort_values('dist2Uni')

distThresh = .5
close = merged.loc[merged['dist2Uni']<distThresh,:]
far   = merged.loc[merged['dist2Uni']>=distThresh,:]

# nNear = 400;
# close = merged.iloc[1:nNear,:]
# far = merged.iloc[(nNear+1):,:]

closeTotalPop = np.sum(close['All Ages'])
farTotalPop = np.sum(far['All Ages'])
#closeCasePer100 = 1e5*np.nansum(close.filter(like='wk_'), axis=0)/closeTotalPop
#farCasePer100 = 1e5*np.nansum(far.filter(like='wk_'), axis=0)/farTotalPop

closeCasePer100 = np.nanmean(1e5*close.filter(like='wk_').div(close['All Ages'], axis=0), axis=0)
farCasePer100 = np.nanmean(1e5*far.filter(like='wk_').div(far['All Ages'], axis=0), axis=0)

plt.plot(closeCasePer100)
plt.plot(farCasePer100)
plt.ylabel('Cases per 100k population')
plt.xlabel('Week #')
plt.legend(['Areas near Universities', 'Other Areas'])

plt.show()

# %%

merged = merged.sort_values('latest_7_days')

targetWeek = 'All Ages'
tmp = merged.tail(6000)
tmp = tmp.loc[tmp['dist2Uni'] < 10, :]
plt.scatter(tmp['dist2Uni'],tmp[targetWeek])
plt.ylabel('Cases per 100k population')
plt.xlabel('Distance to University (Miles)')
plt.show()