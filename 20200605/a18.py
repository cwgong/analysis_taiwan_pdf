#!flask/bin/python
from flask import Flask, jsonify
from flask import request
from flask import abort
from v18 import *
import json
import read_json_taiwan as read_json_taiwan
import read_json_taiwan_new as read_json_taiwan_new
import multiprocessing
app = Flask(__name__)


indir='./pdf_data'
origin_img_dir='./test'
initial(indir)
initial(origin_img_dir)

all_dir=['./FORM','./TXT','./IMAGE','./whole','./whole_txt','./caption','./nonline','./outfile']
initial(all_dir[0])
initial(all_dir[1])
initial(all_dir[2])

show_all=True
if show_all:
    initial(all_dir[3])

show_whole_txt=True
if show_whole_txt:
    initial(all_dir[4])

show_caption=True
if show_caption:
    initial(all_dir[5])
    initial('./clean_whole_txt')

initial(all_dir[6])
initial(all_dir[7])


@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'pdfurl' in request.json:
        abort(400)
    
    file_name=[]
    for file in request.json['pdfurl']:
        file_name.append(os.path.split(file)[-1].split('.')[0])
    print(file_name)
    print(request.json['pdfurl'])
    with multiprocessing.Pool() as pool:
        pool.map(main, request.json['pdfurl'])
    # main(request.json['pdfurl'])
    #result=str()
    result_data_all = []
    for name in file_name:
        result_data = read_json_taiwan_new.taiwan_to_standard("/data/taiwan/20200424/output_json/{}.json".format(name))
        result_data_all.append(result_data)
        #with open('output_json/{}.json'.format(name),'r') as f:
            #result_json=f.read()
            # print(result_json)
            #result+=result_json
    result_dic = {"json_list":result_data_all}
    result_json = json.dumps(result_dic)
    # result.strip()
    # print('---------------------------')
    #temp = {"key":"value"}
    #temps = json.dumps(temp)
    # print(result)
    # return json.dumps(result,ensure_ascii=False), 201
    return result_json, 201

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
