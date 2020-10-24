import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter

def haversine(lon1, lat1, lon2, lat2):
    # Calculate distances between lat/lon in miles.
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon / 2.0) ** 2

    dist = 2 * np.arcsin(np.sqrt(haver_formula))
    dist = 3958 * dist  # 6367 for distance in KM for miles use 3958
    return dist

def commonPlotDecoration():
    plt.xlabel('')
    ax.set_xlim(right=ax.get_xlim()[1] + 1)
    # ax.xaxis.set_major_formatter( ConciseDateFormatter( '%b' ) )

    plt.annotate('Created by Justin Ales, code available: https://github.com/j-ales/covid19-neighborhood',
                 (0, 0), (20, -30), xycoords='axes fraction',
                 textcoords='offset points', va='top',
                 fontsize=8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

datazoneToInterZone = pd.read_csv('scotland-datazone-to-interzone.csv');

# QS603SC
#https://www.scotlandscensus.gov.uk/ods-web/standard-outputs.html?year=2011&table=QS603SC&tableDescription=Economic%20activity%20-%20Full-time%20students
studentPop = pd.read_csv('scotland_student_pop_SNS2011.csv');
studentPop = studentPop.rename(columns={'All full-time students aged 16 to 74': 'studentPop'})

restricted = pd.read_csv('currentRestrictionScotland.csv')

weeklyCases =  pd.read_csv('scotland_weekly_cases_iz.csv', thousands=',');
weeklyCases['dateEnd']=pd.to_datetime(weeklyCases['dateEnd'])
weeklyCases['dateStart']=pd.to_datetime(weeklyCases['dateStart'])
#Replace the obfuscated 1-4, with midpoint 2.5
weeklyCases['cases']=pd.to_numeric(weeklyCases['cases'].replace(to_replace='1-4',value=2.5))

izNames = pd.read_csv('scotland-iz.csv')
izCentroids = pd.read_csv('scotland-iz-centroids.csv')
HE_centroids = pd.read_csv('./learning-providers-plus.csv')
# %% Get student pop to IZ

merged = pd.merge(left=datazoneToInterZone, right=studentPop, left_on='DataZone', right_on='DataZone')
studentPopIz = merged.groupby(['InterZone'])[['studentPop']].agg('sum')

weeklyCases = pd.merge(left=weeklyCases,right=izNames[['IntZone','IntZoneName','CAName']],left_on=['council','IZ'], right_on=['CAName', 'IntZoneName'])
weeklyCases = pd.merge(left=weeklyCases,right=studentPopIz,left_on='IntZone',right_index=True)

weeklyCases['studentPercent'] = weeklyCases['studentPop'] / weeklyCases['pop']
weeklyCases['casePer100k'] = 1e5*weeklyCases['cases'].div(weeklyCases['pop'], axis=0)

# %%

HE_lon = HE_centroids['LONGITUDE']
HE_lat = HE_centroids['LATITUDE']

# Calculate distances. NOTE: Using a for loop like this is a very slow way to do it
# but is generally more clear to read.
allDist = np.zeros((len(izCentroids.index), len(HE_centroids.index)))

for index, row in izCentroids.iterrows():
    this_iz_lon = row['longitude']
    this_iz_lat = row['latitude']

    theseDist = haversine(this_iz_lon, this_iz_lat, HE_lon, HE_lat)
    allDist[index,] = theseDist

#Calculate the distance to nearest university.
uni_distance = np.nanmin(allDist, axis=1)
closestUniIdx = np.nanargmin(allDist, axis=1)
uni_name = HE_centroids['PROVIDER_NAME'].iloc[closestUniIdx].values
izCentroids.insert(1, "uni_distance", uni_distance)
izCentroids.insert(1, "uni_name", uni_name)



# %%

# Separate areas by student concentation.
popThresh = .25
highStudent = weeklyCases.loc[weeklyCases['studentPercent'] >= popThresh]
lowStudent = weeklyCases.loc[weeklyCases['studentPercent'] < popThresh]

highData= highStudent.groupby(['dateEnd'])[['casePer100k']].mean()
lowData= lowStudent.groupby(['dateEnd'])[['casePer100k']].mean()

numberHigh=len(highStudent['IntZone'].unique())
numberLow=len(lowStudent['IntZone'].unique())

fig, ax = plt.subplots(1, 1)
highData.plot(y='casePer100k',ax=ax,linewidth=3)
lowData.plot(y='casePer100k',ax=ax,linewidth=3)

plt.legend(['{} areas with students {:.0%}+ of pop.'.format(numberHigh,popThresh),
        '{} areas with students less then {:.0%} of pop.'.format(numberLow,popThresh)],
           frameon=False)
plt.ylabel('Weekly Cases per 100k')


plt.title('Scotland Intermediate Zone Case Rates ')
commonPlotDecoration()

plt.show()



totalData = weeklyCases.groupby(['dateEnd'])[['cases']].sum()
highData= highStudent.groupby(['dateEnd'])[['cases']].sum()
lowData= lowStudent.groupby(['dateEnd'])[['cases']].sum()

fig, ax = plt.subplots(1, 1)
highData.plot(y='cases',ax=ax)
lowData.plot(y='cases',ax=ax)
#totalData.plot(y='cases',ax=ax)

plt.legend(['{} areas with students {:.0%}+ of pop.'.format(numberHigh,popThresh),
        '{} areas with students less then {:.0%} of pop.'.format(numberLow,popThresh)] )
plt.ylabel('Weekly Cases')
plt.xlabel('')
commonPlotDecoration()
plt.show()

# %% Make plot of different restriction.
restrictedAreas = weeklyCases.loc[weeklyCases['council'].isin(restricted['council'])]
nonRestrictedAreas = weeklyCases.loc[~weeklyCases['council'].isin(restricted['council'])]

numberResticted=len(restrictedAreas['IntZone'].unique())
numberNonRestricted=len(nonRestrictedAreas['IntZone'].unique())

restrictedAreas= restrictedAreas.groupby(['dateEnd'])[['casePer100k']].mean()
nonRestrictedAreas= nonRestrictedAreas.groupby(['dateEnd'])[['casePer100k']].mean()


fig, ax = plt.subplots(1, 1)
restrictedAreas.plot(y='casePer100k',ax=ax,linewidth=3)
nonRestrictedAreas.plot(y='casePer100k',ax=ax,linewidth=3)
plt.title('Scotland Intermediate Zone Case Rates ')

plt.legend(['{} areas in central belt under local restrictions'.format(numberResticted),
        '{} areas outwith central belt'.format(numberNonRestricted)] )

plt.ylabel('Weekly Cases per 100k')
commonPlotDecoration()
plt.show()


# %%
from IPython.display import display, HTML, Markdown
import imgkit
from tableTemplate import css

maxDate = weeklyCases['dateEnd'].max()
mostRecentWeek = weeklyCases.loc[weeklyCases['dateEnd']==maxDate];
mostRecentWeek = pd.merge(left=mostRecentWeek,right=izCentroids[['IntZone','uni_name','uni_distance']],left_on=['IntZone'], right_on=['IntZone'])

mostRecentWeek = mostRecentWeek.sort_values('casePer100k',ascending=False)
mostRecentWeek['studentPercent'] = 100*mostRecentWeek['studentPercent']

top30 = mostRecentWeek.head(30)

top30 = top30[
    ['council','IZ','cases','casePer100k','studentPercent',
     'uni_distance','uni_name']]

di = {'IZ': 'Neighbourhood',
      'council': 'Local Authority',
      'cases': 'Number of Cases',
      'casePer100k': 'Cases per 100k pop',
      'studentPercent': 'Student Percentage',
      'uni_distance': 'Miles to Univeristy',
      'uni_name': 'Nearest University'
     }

top30.rename(di,axis=1,inplace=True)

pd.set_option('precision',2)

fileModString =maxDate.strftime('%b-%d-%Y')

header='<b>Scotland 30 Intermediate Zones with highest cases per 100k population for 7 days ending ' + fileModString + '</b><br><br>'
html=(top30.to_html(formatters={'Number of Cases': '{:,.0f}'.format, 'Cases per 100k pop': '{:,.0f}'.format, 'Student Percentage': '{:,.0f}%'.format},index=False))
footer='<br>Created by Justin Ales, code available: https://github.com/j-ales/covid19-neighborhood'

imgkit.from_string(css+header+html+footer,'test.png')
