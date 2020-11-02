# %%
import pandas as pd

dzDensity = pd.read_csv('scotland-dz-density.csv')
simd=pd.read_csv('simd_zones.csv',thousands=',')
dzToIz = pd.read_csv('scotland-datazone-to-interzone.csv')


simd['simdBinned'] = pd.cut(simd[' Overall_SIMD16_rank '],bins=[0, 1395, 2790, 4185, 5580, 6976],labels=[1,2,3,4,5])

dzDensity = pd.merge(left=dzDensity,right=simd,left_on='DataZone',right_on='Data_Zone')
dzDensity = pd.merge(left=dzDensity,right=dzToIz[['DataZone','InterZone']],left_on='DataZone',right_on='DataZone' )
col2num ={}
for i in range(5):

    thisColName = f'simd{i+1}_pop'
    col2num[thisColName] = i+1
    dzDensity[thisColName] = 0
    dzDensity.loc[dzDensity['simdBinned']==i+1,thisColName] = dzDensity.loc[dzDensity['simdBinned']==i+1,'All people']


#izPop= dzDensity.groupby(['Council_area','Intermediate_Zone'],as_index=False).sum()
izPop= dzDensity.groupby(['InterZone'],as_index=False).sum()

izPop['simdMostPop'] = izPop[["simd1_pop", "simd2_pop","simd3_pop","simd4_pop","simd5_pop"]].idxmax(axis=1)

izPop['simdMostPop'] = izPop['simdMostPop'].replace(col2num)

for i in range(5):
    thisColName = f'simd{i+1}_pop'
    newColName = f'simd{i+1}_percent'

    izPop[newColName] = izPop[thisColName].div(izPop['All people'])

 # izPop2=izPop[
 #     ['IntZone','IntZoneName','CAName','All people','Area (hectares)',
 #      'simd1_pop', 'simd2_pop', 'simd3_pop', 'simd4_pop', 'simd5_pop','simdMostPop',
 #      'simd1_percent','simd2_percent','simd3_percent','simd4_percent','simd5_percent']]

izPop2=izPop[
      ['InterZone', 'All people', 'Area (hectares)',
       'simd1_pop', 'simd2_pop', 'simd3_pop', 'simd4_pop', 'simd5_pop','simdMostPop',
       'simd1_percent', 'simd2_percent','simd3_percent','simd4_percent','simd5_percent']]


