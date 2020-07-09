# -*- coding: UTF-8 -*-

# 运行etl，测试运行情况，检查运行结果

import requests
import json
import http.client
import json

analysis_url = "http://116.62.122.132:5000/todo/api/v1.0/tasks"


def requests_post(url, params):
    response = requests.post(url=url,data=params,headers={'Content-Type':'application/x-www-form-urlencoded'})
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
        # log.debug(code)
        # log.debug(reason)
        data = json.loads(response.read().decode('utf-8'))
        print(data)
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


def analysis_test():
    params = {
        "pdfurl": ["http://static.cninfo.com.cn/finalpage/2020-02-18/1207309925.PDF",
                   "http://static.cninfo.com.cn/finalpage/2020-02-18/1207309925.PDF"]
    }
    result = requests_post(analysis_url,params)
    with open("./test_pdf","w",encoding="utf-8") as f:
        f.write(result)

if __name__ == "__main__":

    pdf_analysis_test()

