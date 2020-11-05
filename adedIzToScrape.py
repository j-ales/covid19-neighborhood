import pandas as pd
import datetime as dt

cases =  pd.read_csv('scotland-iz-cases-from-aug.csv', thousands=',');

casesOlder = pd.read_csv('scotland_weekly_cases_iz.csv', thousands=',');

izNames = pd.read_csv('scotland-iz.csv')

cases['dateEnd']=pd.to_datetime(cases['dateEnd'])
casesOlder['dateEnd']=pd.to_datetime(casesOlder['dateEnd'])
casesOlder=casesOlder.loc[casesOlder['dateEnd']<dt.datetime(2020,8,1)]
cases=

 cases = pd.merge(left=cases,right=izNames[['IntZone','IntZoneName','CAName']],left_on=['council','IZ'], right_on=['CAName', 'IntZoneName'])
