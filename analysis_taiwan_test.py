import http.client
import json
import os
import io
import time
import save_to_mongo
import hashlib
import datetime
import read_json_taiwan

json_file_url = './output_json/'   #json文件目录

def pdf_analysis_single(file_1,file_2,file_3,file_4,file_5,file_6,file_7,file_8):
    try:
        params = json.dumps({"title":[file_1,file_2,file_3,file_4,file_5,file_6,file_7,file_8]})
        # log.debug(params)
        headers = {"Content-type": "application/json"}
        conn = http.client.HTTPConnection("localhost", 5000)
        conn.request('POST', '/todo/api/v1.0/tasks', params, headers)
        response = conn.getresponse()
        code = response.status
        reason=response.reason
        # log.debug(code)
        # log.debug(reason)
        data = json.loads(response.read().decode('utf-8'))
        conn.close()
    except Exception as e:
        data = e
        # log.error(e)
    # log.debug('data:{}，{}'.format(data,type(data)))
    return data

def open_json_file(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as file:
        line = file.readline()
        result = json.loads(line)
    return result

def batch_process_save(batch_data):
    param_list= []
    # with open('all_info_1.json',encoding='utf-8') as f1:
    #     text_info_all = json.load(f1)
    for pdf_data in batch_data:
        if not os.path.exists(json_file_url + str(pdf_data['text_id']) + ".json"):
            continue
        json_data_file = read_json_taiwan.taiwan_to_standard(json_file_url + str(pdf_data['text_id']) + ".json")
        # pdf_data_all = {}
        # for text_info in text_info_all:
        #     if str(text_info['text_id']) == os.path.splitext(pdf_data)[0]:
        #         pdf_data_all = text_info.copy()
        #         break
        #     else:
        #         continue
        # if pdf_data_all == {}:
        #     continue
        now_time = int(time.time()) * 1000

        dateArray = datetime.datetime.utcfromtimestamp(pdf_data['pub_date']/1000)
        otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")

        _id = str(pdf_data['text_id']) + pdf_data['sec_code']
        _id = hashlib.md5(_id.encode('utf8')).hexdigest().replace('-', '')  # 32位

        temp_dic = {'id': _id,
                    'textId': pdf_data['text_id'],  #此处如果是uniqueid的话命名规则会改变，textId也会改变
                    'secCode': pdf_data['sec_code'],
                    'marType': pdf_data['mar_type'],
                    'secName': pdf_data['sec_name'],
                    'pubDate': otherStyleTime,
                    'pubDateAt': pdf_data['pub_date'],
                    'annTitle': pdf_data['ann_title'],
                    'infoType': pdf_data['info_type'],
                    'annUrl': pdf_data['ann_url'],
                    'parseStatus': '1',
                    'infoTypes': pdf_data['info_type'].split('||'),
                    'annGroup': '合作设立产业并购基金公告',
                    'pdfAnalyseType': '台湾',
                    'createAt': now_time,
                    'updateAt': now_time}

        new_dict = temp_dic.copy()    #需要把垚锐接口的所有信息都带进来为item
        new_dict['detail'] = json_data_file
        param_list.append(new_dict)

    print('total length will be inserted: %d' % len(param_list))
    # print(param_list[0])
    for i in range(len(param_list)):
        save_to_mongo.insert_to_mongo(param_list[i])

def mk_pdf_name_1(all_info):
    text_id = all_info['text_id']
    pdf_name = str(text_id) + '.PDF'
    return pdf_name

def iteration_analysis():
    with open('long_1.json','r',encoding='utf-8') as f:
        text_list = json.load(f)
    epochs = int(len(text_list) / 8) + 1
    for i in range(epochs):
        # logging.info("i:%s in epochs:%s" % (str(i), str(epochs - 1)))
        batch_data = text_list[i * 8: (i + 1) * 8]
        data = pdf_analysis_single(mk_pdf_name_1(batch_data[0]),mk_pdf_name_1(batch_data[1]),mk_pdf_name_1(batch_data[2]),mk_pdf_name_1(batch_data[3]),mk_pdf_name_1(batch_data[4]),mk_pdf_name_1(batch_data[5]),mk_pdf_name_1(batch_data[6]),mk_pdf_name_1(batch_data[7]))#最后的一个epoch不会有八篇，所以会报错
        batch_process_save(batch_data)

if __name__ == "__main__":
    iteration_analysis()

    # "1203274310.PDF", "1203691722.PDF", "1204239250.PDF", "1204738456.PDF", "1205647202.PDF", "1206109965.PDF", "1207047213.PDF", "1203121058.PDF"