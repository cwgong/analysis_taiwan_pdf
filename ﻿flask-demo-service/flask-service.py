# -*- coding: utf-8 -*-

# python
from flask import Flask
from flask import request
from flask import abort
import datetime
import requests
import uuid
import json
import os
import io

# business logical
import multiprocessing
from models.v18 import analyzer
import traceback
import logging


app = Flask(__name__)

# 用于暂时性存储根据 pdf_url 下载的 pdf 文件，以进行后续 pdf 解析
PDF_DIR = './pdf'
if not os.path.exists(PDF_DIR):
    os.mkdir(PDF_DIR)

@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def handler():
    try:
        # 根据 pdf_url 下载 pdf 文件，下载容易出现超时，设置次数
        def get_pdf(pdf_url, file_path, c=1):
            try:
                r = requests.get(pdf_url)
                f = io.open(file_path, "wb")
                f.write(r.content)
                f.close()
            except Exception as e:
                if c < 6:
                    c += 1
                    print("get_pdf() again!")
                    get_pdf(pdf_url, file_path, c)
                print("get_pdf() error!")
                logging.exception("get_pdf() error! c: {}".format(c))

        #生成 pdf name，下載pdf至本地
        def download_pdf_to_local(pdf_urls,PDF_DIR):
            download_start = datetime.datetime.now()
            pdf_paths = []
            for pdf_url in pdf_urls:
                # 随机性生成 pdf name
                url_md5 = str(uuid.uuid1()).replace('-', '')
                file_path = PDF_DIR + '/' + url_md5 + '.pdf'
                get_pdf(pdf_url, file_path)
                pdf_paths.append(file_path)
            download_end = datetime.datetime.now()
            logging.info("download time: " + str(download_end - download_start))
            return pdf_paths
        
        # step1、获取 post 参数 pdf_urls = [url1, url2, ..]
        # -------------------------------------------------
        pdf_urls = request.json['pdf_urls']

        # step2、下载 pdf 到本地
        # -------------------------------------------------
        pdf_paths = download_pdf_to_local(pdf_urls,PDF_DIR)

        # step3、多线程处理，调用台湾团队 pdf 解析函数
        # -------------------------------------------------
        with multiprocessing.Pool() as pool:
            pdf_jsons = pool.map(analyzer, pdf_paths)

        # step4、解析后删除本地 pdf 文件
        # -------------------------------------------------
        for pdf_path in pdf_paths:
            os.remove(pdf_path)

        # step5、组装 POST 结果并返回
        # -------------------------------------------------
        # ***注意***
        # 要保证 len(pdf_urls) == len(pdf_jsons)，并且 "顺序" 要与用户传入的一致
        Restful_Result = {"data": pdf_jsons,
                          "message": {"code": 0,
                                      "message": "success"}}
        return Restful_Result

    except Exception as e:
        logging.error(traceback.format_exc())
        Restful_Result = {"data": [],
                          "message": {"code": -1,
                                      "message": str(e)}}
        return json.dumps(Restful_Result, ensure_ascii=False)


if __name__ == '__main__':

    host = '0.0.0.0'
    port = 5000

    app.run(debug=True, host=host, port=port)
