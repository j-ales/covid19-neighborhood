import pandas as pd
import datetime as dt

# cases =  pd.read_csv('cases_from_nov5.csv', thousands=',')
#
# casesOlder = pd.read_csv('cases_from_nov4.csv', thousands=',')
cases =  pd.read_csv('tmp.csv', thousands=',')

casesOlder = pd.read_csv('scotland_weekly_cases_iz.csv', thousands=',')

izNames = pd.read_csv('scotland-iz.csv')

cases['dateEnd']=pd.to_datetime(cases['dateEnd'])
casesOlder['dateEnd']=pd.to_datetime(casesOlder['dateEnd'])
casesOlder=casesOlder.loc[casesOlder['dateEnd']<min(cases['dateEnd'])]
cases = cases.append(casesOlder)
cases = pd.merge(left=cases,right=izNames[['IntZone','IntZoneName','CAName']],left_on=['council','IZ'], right_on=['CAName', 'IntZoneName'])
cases['IZCode'] = cases['IntZone']
cases['dateEnd']=cases['dateEnd'].dt.strftime('%d %B %Y')
cases = cases[['IZCode','council','IZ','dateStart','dateEnd','cases','pop']]