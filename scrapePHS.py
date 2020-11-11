# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# This script updates the csv file containing the Intermediate zone data. It requires manual
# updating of what the newest date data is available from the PHS website.

# %%
import requests
from bs4 import BeautifulSoup
import json
import re
import urllib
import pandas as pd
import datetime as dt

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def updateWeeklyCaseCsv(oldFile, newFile):
    cases = pd.read_csv(newFile, thousands=',')

    casesOlder = pd.read_csv(oldFile, thousands=',')

    izNames = pd.read_csv('scotland-iz.csv')

    cases['dateEnd'] = pd.to_datetime(cases['dateEnd'])
    casesOlder['dateEnd'] = pd.to_datetime(casesOlder['dateEnd'])
    casesOlder = casesOlder.loc[casesOlder['dateEnd'] < min(cases['dateEnd'])]
    cases = cases.append(casesOlder)
    cases = pd.merge(left=cases, right=izNames[['IntZone', 'IntZoneName', 'CAName']], left_on=['council', 'IZ'],
                     right_on=['CAName', 'IntZoneName'])
    cases['IZCode'] = cases['IntZone']
    cases['dateEnd'] = cases['dateEnd'].dt.strftime('%d %B %Y')
    cases = cases[['IZCode', 'council', 'IZ', 'dateStart', 'dateEnd', 'cases', 'pop']]
    cases.drop_duplicates(inplace=True)
    cases.to_csv(oldFile)

# %% [markdown]
# Change the dates below to the select the dates to scrape from the PHS tableau.   Choose the most current date available for the end_date and 4 days before the previous data grab.  Data comes in slowly so most recent data can be revised in new days.  

# %%

retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS","POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)

# Definitions
outputFile = 'scotland_weekly_cases_iz.csv'
currentData = pd.read_csv(oldFile, thousands=',')
max(currentData['dateEnd'])

#default to recollect previous 4 days.
start_date =pd.to_datetime(max(currentData['dateEnd']))-dt.timedelta(3)
start_date = start_date.date()

#Alternatively manually set date to start from
#start_date = dt.date(2020, 11, 8)

end_date = dt.date(2020, 11, 8)
end_date_fix = end_date
df = pd.DataFrame(columns=["council", "IZ", "dateStart", "dateEnd", "cases", "pop"])
#List of councils to skip
councilsFinished = []
#councilsFinished = [ "Aberdeen City", "Aberdeenshire"]

# %% [markdown]
# The next section  does the scraping proper.  Can take a while.   It will display what it is getting.  At the end it will update the file. 

# %%

errorCount = 0
finished = False

while not finished:
    print("Opening connection to: https://public.tableau.com/views/COVID-19DailyDashboard_15960160643010/Overview")
    try:
        s = requests.Session()
        s.mount("https://", adapter)
        s.mount("http://", adapter)

        data_host = "https://public.tableau.com"

        r = requests.get('https://public.tableau.com/views/COVID-19DailyDashboard_15960160643010/Overview',
            params = {
            ":embed": "y",
            ":showVizHome": "no",
            ":display_count":"y",
            ":display_static_image": "y",
            ":bootstrapWhenNotified": "true",
            ":language": "en-GB",
            "embed":"y",
            ":showVizHome":"n",
            ":apiID":"host0",
        })

        soup = BeautifulSoup(r.text, "html.parser")

        tableauData = json.loads(soup.find("textarea",{"id": "tsConfigContainer"}).text)

        # paramTags = dict([
        #     (t["name"], t["value"])
        #     for t in soup.find("div", {"class":"tableauPlaceholder"}).findAll("param")
        # ])
        #'COVID-19DailyDashboard_15960160643010/Casesbyneighbourhood'



        #https://public.tableau.com/vizql/w/COVID-19DailyDashboard_15960160643010/v/Overview/sessions/2313F4AE03384491BF9A801771048688-0:0/commands/tabsrv/ensure-layout-for-sheet



        #https://public.tableau.com/vizql/w/COVID-19DailyDashboard_15960160643010/v/Overview/sessions/9F3BA709B4024279A38576A3EB91602C-0:0/commands/tabdoc/set-parameter-value

        dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/bootstrapSession/sessions/{tableauData["sessionid"]}'
        r2 = requests.post(dataUrl, data= {
        "sheet_id": "Overview",
        "stickySessionKey": urllib.parse.quote_plus(tableauData["stickySessionKey"]),
        # "worksheetPortSize": "%7B%22w%22%3A1050%2C%22h%22%3A1900%7D",
        #    "vizRegionRect": json.dumps({"r": "viz", "x": 496, "y": 148, "w": 0, "h": 0, "fieldVector": None}),
        # "dashboardPortSize":" %7B%22w%22%3A1050%2C%22h%22%3A1900%7D",
        # "clientDimension": "%7B%22w%22%3A516%2C%22h%22%3A727%7D",
        # "renderMapsClientSide": "true",
        # "isBrowserRendering": "true",
        # "browserRenderingThreshold": "100",
        # "formatDataValueLocally": "false",
        # "clientNum":"",
        # "navType": "Reload",
        # "navSrc": "Parse",
        # "devicePixelRatio": "2",
        # "clientRenderPixelLimit": "25000000",
        # "allowAutogenWorksheetPhoneLayouts": "false",
        # "showParams": "%7B%22checkpoint%22%3Afalse%2C%22refresh%22%3Afalse%2C%22refreshUnmodified%22%3Afalse%7D",
        # #"stickySessionKey": "{\"dataserverPermissions\":\"44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a\",\"featureFlags\":\"{\\\"MetricsAuthoringBeta\\\":false}\",\"isAuthoring\":false,\"isOfflineMode\":false,\"lastUpdatedAt\":1603285164529,\"workbookId\":"+f'{tableauData["current_workbook_id"]}'+"}",
        # "filterTileSize": "200",
        # "locale": "en_GB",
        # "language": "en_GB",
        # "verboseMode": "false",
        # ":session_feature_flags": "%7B%7D",
        # "keychain_version": "1",
        })


        dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/sessions/{tableauData["sessionid"]}/commands/tabsrv/ensure-layout-for-sheet'
        r3 = requests.post(dataUrl, data= {
            "targetSheet": "Cases by neighbourhood"
        })

        iz = pd.read_csv('scotland-iz.csv')
        councilList = iz.groupby("CAName").size().reset_index(name='count')



        for index, row in councilList.iterrows():


            thisCouncil = row["CAName"]
            numIZ = row["count"]
            if (thisCouncil in councilsFinished):
                continue

            dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/sessions/{tableauData["sessionid"]}/commands/tabdoc/set-parameter-value'
            r4 = requests.post(dataUrl, data= {
#                "valueString": urllib.parse.quote_plus(thisCouncil),
                "valueString": thisCouncil,
                "globalFieldName": "[Parameters].[Parameter 1 1]",
                "useUsLocale": "false"
            })

            dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/sessions/{tableauData["sessionid"]}/commands/tabsrv/pane-anchor-zoom-server'
            r5 = requests.post(dataUrl, data= {
            "vizRegionRect": json.dumps({"r":"viz","x":537,"y":440}),
            "zoomAnchorPoint": json.dumps({"x":537,"y":440}),
            "zoomFactor": "1.1810923195715735",
            "visualIdPresModel": json.dumps({"worksheet":"IZ_map","dashboard":"Cases by neighbourhood"})
            })

            allTips = []

            numdays = (end_date - start_date).days +1
            date_list = [end_date - dt.timedelta(days=x) for x in range(numdays)]

            for thisDate in range(0,len(date_list),1):

                end_date = date_list[thisDate]
                print(f'Processing: {date_list[thisDate].strftime("%-d %B %Y")}')
                dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/sessions/{tableauData["sessionid"]}/commands/tabdoc/categorical-filter'
                rChangeTime = requests.post(dataUrl, data= {
                "visualIdPresModel": json.dumps({"worksheet":"LA_map","dashboard":"Cases by neighbourhood"}),
                "membershipTarget": "filter",
                "globalFieldName": "[federated.09570nd02ojji51547xls1hzn4bs].[md:Date:ok]",
                "filterValues": f'[\"{date_list[thisDate].strftime("%-d %B %Y")}\"]',
                "filterUpdateType": "filter-replace",
                "heuristicCommandReinterpretation": "do-not-reinterpret-command",
                })
                rSave = rChangeTime
                # rChangeTime.json()["vqlCmdResponse"]["layoutStatus"]["applicationPresModel"]["dataDictionary"]["dataSegments"]["1"][
                #     "dataColumns"][0]["dataValues"][3]
                #Data supressed when region has less than 4 cases.
               # if ("Disclosure control applied" in rChangeTime.text) or ("dataDictionary" not in rChangeTime.json()["vqlCmdResponse"]["layoutStatus"]["applicationPresModel"]):
            #    if ("Disclosure control applied" in rChangeTime.text):
                #    print(f'Disclosure control applied for date {date_list[thisDate].strftime("%-d %B %Y")}')
                    #dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/sessions/{tableauData["sessionid"]}/commands/tabdoc/categorical-filter'
                   # rReset= requests.post(dataUrl, data={
                    #     "visualIdPresModel": json.dumps({"worksheet": "LA_map", "dashboard": "Cases by neighbourhood"}),
                    #     "membershipTarget": "filter",
                    #     "globalFieldName": "[federated.09570nd02ojji51547xls1hzn4bs].[md:Date:ok]",
                    #     "filterValues": "[\"11 October 2020\"]",
                    #     "filterUpdateType": "filter-replace",
                    #     "heuristicCommandReinterpretation": "do-not-reinterpret-command",
                    # })
                    #
                    # r7July= requests.post(dataUrl, data={
                    #     "visualIdPresModel": json.dumps({"worksheet": "LA_map", "dashboard": "Cases by neighbourhood"}),
                    #     "membershipTarget": "filter",
                    #     "globalFieldName": "[federated.09570nd02ojji51547xls1hzn4bs].[md:Date:ok]",
                    #     "filterValues": "[\"7 July 2020\"]",
                    #     "filterUpdateType": "filter-replace",
                    #     "heuristicCommandReinterpretation": "do-not-reinterpret-command",
                    # })
                    #rSave = rChangeTime
                    #continue

                for i in range(1,numIZ+10):

                    print(f'Requesting zone: {i}')
                    dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/sessions/{tableauData["sessionid"]}/commands/tabsrv/render-tooltip-server'
                    r6 = requests.post(dataUrl, data= {
                    "worksheet": "IZ_map",
                    "dashboard": "Cases by neighbourhood",
                    "tupleIds": f'[{i}]',
                    "vizRegionRect": json.dumps({"r":"viz","x":277,"y":332,"w":0,"h":0}),
                    "allowHoverActions": "false",
                    "allowPromptText": "true",
                    "allowWork": "false",
                    "useInlineImages": "true",
                    })
                    tooltipText=r6.json()["vqlCmdResponse"]["cmdResultList"][0]["commandReturn"]["tooltipText"]
                    if not tooltipText:
                        print('No tooltiptext')
                        break
                    tmp=json.loads(tooltipText)
                    if "htmlTooltip" not in tmp:
                        print('no htmlTooltip')
                        break
                    tooltip = tmp["htmlTooltip"]
                    soup = BeautifulSoup(tooltip, "html.parser")

                    allTips.append(soup.text)

                    thisIzTip = soup.find_all('a')

                    df = df.append({
                        "council": thisIzTip[0].get_text(),
                        "IZ": thisIzTip[1].get_text(),
                        "dateStart": thisIzTip[2].get_text(),
                        "dateEnd": thisIzTip[3].get_text(),
                        "cases": thisIzTip[4].get_text(),
                        "pop": soup.find_all('span')[20].get_text()}, ignore_index=True)

                    print(f'Read: {thisIzTip[0].get_text()} {thisIzTip[3].get_text()} {thisIzTip[1].get_text()} {thisIzTip[4].get_text()}' )
                df.to_csv('tmp.csv')
            end_date = end_date_fix #reset end date.
            councilsFinished.append(thisCouncil)
        print('Finished Scraping')

        finished=True
    except KeyboardInterrupt:
        print("Caught Keyboard Interrupt")
        break
    except:
        errorCount = errorCount +1
        print(f'{errorCount} errors happened. Re-establishing connection')
        continue

print('Updating File: ' + outputFile)
updateWeeklyCaseCsv(outputFile, 'tmp.csv')
# %%
