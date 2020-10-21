import requests
from bs4 import BeautifulSoup
import json
import re
import urllib
import pandas as pd


#from selenium import webdriver
# %%
s = requests.Session()

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

dataUrl = f'https://public.tableau.com{tableauData["vizql_root"]}/sessions/{tableauData["sessionid"]}/commands/tabdoc/set-parameter-value'
r4 = requests.post(dataUrl, data= {
    "valueString": "Aberdeen City",
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
df = pd.DataFrame(columns=["council", "IZ", "dateStart", "dateEnd", "cases", "pop"])

for i in range(1,100):
    print(i)
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
    print(r6.status_code)
    tmp=json.loads(r6.json()["vqlCmdResponse"]["cmdResultList"][0]["commandReturn"]["tooltipText"])
    if "htmlTooltip" not in tmp:
        continue
    tooltip = tmp["htmlTooltip"]
    soup = BeautifulSoup(tooltip, "html.parser")

    allTips.append(soup.text)

    df = df.append({
        "council": soup.find_all('a')[0].get_text(),
        "IZ": soup.find_all('a')[1].get_text(),
        "dateStart": soup.find_all('a')[2].get_text(),
        "dateEnd": soup.find_all('a')[3].get_text(),
        "cases": soup.find_all('a')[4].get_text(),
        "pop": soup.find_all('span')[20].get_text()}, ignore_index=True)

# %%
# globalFieldName: [Parameters].[Parameter 1 1]
# valueString: Aberdeen City
# useUsLocale: false
#
#
# # get xsrf cookie
# session_url = f'{paramTags["host_url"]}trusted/{paramTags["ticket"]}{paramTags["site_root"]}/views/{paramTags["name"]}'
# print(f"GET {session_url}")
# r = s.get(session_url)
#
# config_url = f'{paramTags["host_url"][:-1]}{paramTags["site_root"]}/views/{paramTags["name"]}'
# print(f"GET {config_url}")
# r = s.get(config_url,
#     params = {
#         ":embed": "y",
#         ":showVizHome": "no",
#         ":host_url": "https://interactive.data.illinois.gov/",
#         ":embed_code_version": 2,
#         ":tabs": "yes",
#         ":toolbar": "no",
#         ":showShareOptions": "false",
#         ":display_spinner": "no",
#         ":loadOrderID": 0,
# })
# soup = BeautifulSoup(r.text, "html.parser")
# tableauData = json.loads(soup.find("textarea",{"id": "tsConfigContainer"}).text)
#
# dataUrl = f'{paramTags["host_url"][:-1]}{tableauData["vizql_root"]}/bootstrapSession/sessions/{tableauData["sessionid"]}'
# print(f"POST {dataUrl}")
# r = s.post(dataUrl, data= {
#     "sheet_id": tableauData["sheetId"],
# })
# dataReg = re.search('\d+;({.*})\d+;({.*})', r.text, re.MULTILINE)
# info = json.loads(dataReg.group(1))
# data = json.loads(dataReg.group(2))
