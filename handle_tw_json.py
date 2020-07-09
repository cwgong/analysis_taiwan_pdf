# -*- coding: utf-8 -*-
import pymysql
import time
import datetime
import io
import json
import os
import hashlib

import save_to_mongo

zgsms_json_path = 'zgsms/json_data/'


# 把datetime类型转为时间戳形式(毫秒)
def datetime_to_timestamp(dt):
    return int(time.mktime(dt.timetuple())*1000)


def get_bond_prospectus(start, end):
    host = '10.0.0.95'
    port = 3306
    db = 'datacenter_pub'
    user = 'product'
    password = 'easy2get'

    connection = pymysql.connect(host=host, port=port, user=user, password=password, db=db, charset='utf8')
    cursor = connection.cursor()

    # sql = "SELECT t.text_id, t.sec_code, t.mar_type, t.sec_name, t.pub_date, t.ann_title, t.info_type, t.ann_url FROM ann_info_tab_news t WHERE t.ann_title LIKE '%说明书%' AND t.ann_title LIKE '%债券募集%' AND t.update_at>='{}' AND t.update_at<'{}'".format(
    #     start, end)
    sql = " SELECT t.textid, t.sec_code, t.mar_type, t.sec_name, t.pub_date, t.ann_title, t.info_type, t.ann_url FROM dw.stk_ann_classify_info t WHERE t.ann_group = '招股说明书' AND t.mod_time >= '{}' and t.mod_time <= '{}' ".format(start, end)

    cursor.execute(sql)
    result = cursor.fetchall()

    y = []
    for data in result:
        pub_date = data[4]  # type = datetime
        # print(pub_date)

        ss = str(pub_date)
        yy = ss[0] + ss[1] + ss[2] + ss[3]
        m = ss[4] + ss[5]
        d = ss[6] + ss[7]
        dd = yy + '-' + m + '-' + d
        # print(dd)
        # print(type(dd))
        pub_date = datetime.date(*map(int, dd.split('-')))
        timestamp = datetime_to_timestamp(pub_date)
        # print(timestamp_to_date(timestamp)) # 已验证

        now_time = int(time.time()) * 1000

        _id = str(data[0]) + data[1]
        _id = hashlib.md5(_id.encode('utf8')).hexdigest().replace('-', '')  # 32位

        temp_dic = {'id': _id,
                    'textId': data[0],
                    'secCode': data[1],
                    'marType': data[2],
                    'secName': data[3],
                    'pubDate': ss,
                    'pubDateAt': timestamp,
                    'annTitle': data[5],
                    'infoType': data[6],
                    'annUrl': data[7],
                    'parseStatus': '1',
                    'infoTypes': data[6].split('||'),
                    'annGroup': '招股说明书',
                    'pdfAnalyseType':'台湾',
                    'createAt': now_time,
                    'updateAt': now_time}
        y.append(temp_dic)
    # 关闭数据连接
    connection.close()
    return y


def filter(data_list):
    result_list = []
    for item in data_list:
        sec_name = item.get('secName')
        ann_tittle = item.get('annTitle')
        if len(sec_name) == 0:
            continue
        if '摘要' in ann_tittle:
            continue
        result_list.append(item)
    return result_list


def open_json_file(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as file:
        line = file.readline()
        result = json.loads(line)
    return result


def save(data_list, name_list):
    param_list = []
    for item in data_list:
        text_id = str(item.get('textId'))
        if text_id not in name_list:
            print(text_id)
            continue
        pdf_json = open_json_file(zgsms_json_path + str(text_id) + '.json')
        new_dict = item.copy()
        new_dict['detail'] = pdf_json
        param_list.append(new_dict)

    print('total length will be inserted: %d' % len(param_list))
    # print(param_list[0])
    for i in range(len(param_list)):
        save_to_mongo.insert_to_mongo(param_list[i])


def get_all_json_name():
    file_list = os.listdir(zgsms_json_path)
    file_name_list = []
    for file in file_list:
        prefix = os.path.splitext(file)[0]
        file_name_list.append(prefix)
    return file_name_list


if __name__ == '__main__':
    start = 20190101
    end = 20200101
    mysql_list = get_bond_prospectus(start, end)
    print('total length : %d' % len(mysql_list))
    filter_list = filter(mysql_list)
    print('sec name is not empty and not summary length: %d' % len(filter_list))
    name_list = get_all_json_name()
    print('all names in folder length: %d' % len(name_list))
    save(filter_list, name_list)
    # print(len(open_json_file(zgsms_json_path + '1207048263' + '.json')))

# 近三年：
# total length : 2210
# sec name is not empyt length: 1450

# 近一年：
# total length : 626
# sec name is not empyt length: 330
