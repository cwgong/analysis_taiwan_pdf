#!flask/bin/python
from flask import Flask, jsonify
from flask import request
from flask import abort
from v18 import main
import os
import json
import read_json_taiwan_new as read_json_taiwan_new
import multiprocessing

app = Flask(__name__)

@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'pdfurl' in request.json:
        abort(400)

    file_name = []
    for file in request.json['pdfurl']:
        file_name.append(os.path.split(file)[-1].split('.')[0])

    with multiprocessing.Pool() as pool:
        pool.map(main, request.json['pdfurl'])

    result_data_all = []
    for name in file_name:
        result_data = read_json_taiwan_new.taiwan_to_standard("/data/taiwan/20200424/output_json/{}.json".format(name))
        result_data_all.append(result_data)

    result_dic = {"json_list": result_data_all}
    result_json = json.dumps(result_dic)

    return result_json, 201


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
