# -*- coding: UTF-8 -*-

# 运行etl，测试运行情况，检查运行结果

import requests
import json
import http.client
import json

analysis_url = "http://116.62.122.132:5000/todo/api/v1.0/tasks"


DEBUG_START_AT = '1546272000000'  # 2019-01-01
DEBUG_END_AT = '1584892800000'  # 2020-03-23


def requests_post(url, params):
    response = requests.post(url=url,data=params,headers={'Content-Type':'application/json'})
    # response.encoding = "utf-8"
    print(response.status_code)
    print(response.text)
    return response.text



def pdf_analysis_test():
    try:
        params = json.dumps({"pdfurl":["http://static.cninfo.com.cn/finalpage/2020-02-18/1207309925.PDF","http://static.cninfo.com.cn/finalpage/2020-02-18/1207309925.PDF"]})
        # log.debug(params)
        headers = {"Content-type": "application/json"}
        conn = http.client.HTTPConnection("116.62.122.132", 5000)
        conn.request('POST', '/todo/api/v1.0/tasks', params, headers)
        response = conn.getresponse()
        code = response.status
        reason=response.reason
        print(reason)
        # log.debug(code)
        # log.debug(reason)
        print(code)
        print(type(response))
        #data = json.loads(response.read().decode('utf-8'))
        print(type(response.read()))
        print(response.read().decode())
       # print(data)
        # text = response.write("text_1.txt")
        conn.close()
        # with open("./data/text_1.txt","wb") as f:
        #     f.write(text)
        # restul = json.dumps(data)
        # with open("./data/test_pdf_1", "wb+") as f:
        #     # f.write(restul)
        #     continue
    except Exception as e:
        data = e
        print(data)
        # log.error(e)
    # log.debug('data:{}，{}'.format(data,type(data)))

# def add_job_1():
#     url = job_url + 'add'
#     param = {"timeField": "",
#              "startAt": "",
#              "endAt": "",
#              "knowledgeType": "",
#              "jobId": "1",
#              "jobName": "dayTask",
#              "timeInterval": 86400}
#     result = requests_get(url, param)
    # print("1")
    # print(result)

def analysis_test():
    params = {
        "pdfurl": ["http://static.cninfo.com.cn/finalpage/2020-02-18/1207309925.PDF",
                   "http://static.cninfo.com.cn/finalpage/2020-02-18/1207309925.PDF"]
    }
    result = requests_post(analysis_url,params)
    with open("./test_pdf","w",encoding="utf-8") as f:
        f.write(json.dumps(result))

if __name__ == "__main__":
    # add_job_1()
    # add_job_2()
    # start()
    # query()
    # stop()
    #pdf_analysis_test()
    analysis_test()
    # run("公司市场排名")
    # print(type('123'))

