import requests
import json
import time


# POST 访问
def post_run():
    data = {"alerts": [{"labels": {'instance': 'XNSWC39-VM05.amr.corp.intel.com'}, }, ]}
    requests.post('http://10.239.33.3:8002/api/operation_node/remove', data=json.dumps(data))


# 时间转换  2020-03-15 12:12:00  == >  1584334456831000064
def timestr_to_timestamp():
    tss1 = '2020-03-15 12:12:00'
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    timeStamp = time.mktime(timeArray)

