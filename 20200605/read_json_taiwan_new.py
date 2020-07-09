import json
import io

def taiwan_to_standard(filepath):
    file = io.open(filepath, 'r', encoding='utf-8')
    lines=''
    result_data_all = []
    for idx, line in enumerate(file.readlines()):
        # if idx<7:
        #     lines=lines+line
        #     continue
        if line=='}\n':
            # line='}\n'
            lines=lines+line
            # print(lines)
            result_data=json.loads(lines)
            # print(result_data)
            # print('content:')
            # print(result_data['content'])
            # print('header:')
            # print(result_data['header'])
            # print('footer:')
            # print(result_data['footer'])
            # print('type:')
            # print(result_data['type'])
            # print('page_num:')
            # print(result_data['page_num'])
            # print('----------------')
            result_data_all.append(result_data)
            # lines='{\n'
            lines = ''
        else:
            lines=lines+line
    return result_data_all

if __name__ == "__main__":
    result_data_all = taiwan_to_standard("./data/1207309925.json")
    print(result_data_all)