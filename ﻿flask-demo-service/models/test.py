# -*- coding: utf-8 -*-
from models.v18 import analyzer

def test():

    pdf_path = './pdf/xxxx.pdf'
    pdf_json = analyzer(pdf_path, debug=True)
    print(pdf_json)

if __name__ == '__main__':

    test()




