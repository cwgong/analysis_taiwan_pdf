#from kafka import KafkaConsumer
import os 
import subprocess, sys

import numpy as np
import cv2
from matplotlib import pyplot as plt  

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfpage import PDFTextExtractionNotAllowed

from collections import Counter
from pdfminer.pdfinterp import resolve1
import operator
from difflib import SequenceMatcher
from pdfminer.layout import LAParams,LTImage,LTFigure,LTRect

import time
from threading import Thread
import multiprocessing
import json
import io

import pickle
import re
import csv
import shutil
import requests
import subprocess
import glob
# sys.path.append('./Deep/')
# print(sys.path)
# from Deep.ez_eval_final import evaluate
# E=evaluate()

#------------------------------------------------
def countsimilar(lists):
    new_list=[]# [txt,count]
    for i in lists:
        notmatch=True
        for idx,j in enumerate(new_list):
            if SequenceMatcher(None, i,j[0]).ratio()>0.8:
                new_list[idx][1]+=1
                notmatch=False
            # print('new_list:',new_list)
        if notmatch:
            new_list.append([i,1])
    return new_list
#------------------------------------------------------------
def initial(out_dir):# 初始化資料夾 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('build ',out_dir)
    # else:
    #     shutil.rmtree(out_dir)
    #     os.makedirs(out_dir)
#-----------------------------------------------------------------------
class ThreadWithReturnValue(Thread):# 能回傳結果的thread
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return
# -------------------------------------------------------------------
def make_pic(indir,filename,out_dir):# 使用ghostscript 將pdf轉為每頁圖片
    name=filename.split('.')[0]
    initial(out_dir+'/'+name)
    pdf_route='{}/{}.pdf'.format(indir,name)
    gs = 'GSWIN64C' if (sys.platform == 'win32') else 'gs'
    p =subprocess.Popen([gs,\
    '-dBATCH', '-dNOPAUSE', '-sDEVICE=jpeg', '-r144',\
    '-sOutputFile={}/{}/{}_%03d.jpg'.format(out_dir,name,name), pdf_route])
    p.wait()
    print('Extracting Image Done!')
# ------------------------------------------------------------------
def preprocess(src):# 影像之前處理
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    outs = cv2.adaptiveThreshold(255-gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2) 
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(outs, cv2.MORPH_CLOSE, kernel)
    return closing
#----------------------------------------------
def findbox(rct,maps):# 尋找mask之中的物件是否為方形 利用確認上下左右是否都找得到值（若梯形就找不到
    isrect = True
    array=[(rct[0], rct[1]), (rct[0] + rct[2], rct[1]), (rct[0] + rct[2], rct[1] + rct[3]), (rct[0], rct[1] + rct[3])]
    for pt in array:
        fill = False
        jmin=pt[0]-10 if pt[0] > 10 else 0
        jmax=pt[0]+10 if pt[0] + 10 < maps.shape[1] else maps.shape[1]
        imin=pt[1]-10 if pt[1] > 10 else 0
        imax=pt[1]+10 if pt[1] + 10 < maps.shape[0] else maps.shape[0]

        for j in range(jmin,jmax):
            for i in range(imin,imax):
                if maps[i,j]:
                    fill=True
                    break
        if not fill:
            isrect=False
            break
    return isrect
# -------------------------------------------------
def colorful(rct,src):# 物件內彩色像素計算
    ROI=cv2.cvtColor(src[rct[1]:rct[1]+rct[3],rct[0]:rct[0]+rct[2]],cv2.COLOR_BGR2HSV)
    count=0
    for i in range(0,rct[3],2):
        for j in range(0,rct[2],2):
            if ROI[i,j,1]>50 and ROI[i,j,2]>20:
                count+=1
    return count > rct[2]*rct[3]*0.06
# -------------------------------
def extract_table(images,src): # 表格偵測演算法
    height, width = images.shape
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(int(width / 10), 1));
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(height/55)));
    out_1d_hor=cv2.morphologyEx(images, cv2.MORPH_OPEN, horizontalStructure)
    out_1d_ver=cv2.morphologyEx(images, cv2.MORPH_OPEN, verticalStructure)
    mask=out_1d_hor+out_1d_ver
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,2)
    _ , contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lists=[]
    for contour in contours:
        rct=cv2.boundingRect(contour)
        a = cv2.contourArea(contour, False)
        # if not colorful(rct,src):

        if a>0.6*rct[2]*rct[3]:
            if (rct[3]>height*0.02 and rct[2]>width*0.3) or (a>height*width*0.05):
                lists.append(rct)
        else:
            maps=np.zeros([height,width],dtype=np.uint8)
            cv2.drawContours(maps,contour,-1,255,1)
            if findbox(rct,maps):
                if (rct[3]>height*0.02 and rct[2]>width*0.3):
                    lists.append(rct)
    return lists
# ----------------------------------
def isoverlap(lists,rct,threshold): # 查看rct與list內的方框是否重疊
    for i in lists:
        x0=max(i[0],rct[0])
        x1=min(i[0]+i[2],rct[0]+rct[2])
        y0=max(i[1],rct[1])
        y1=min(i[1]+i[3],rct[1]+rct[3])
        if x0>=x1 or y0>=y1:
            continue
        else:
            result=(x1-x0)*(y1-y0)/((i[2]*i[3])+(rct[2]*rct[3])-(x1-x0)*(y1-y0))
            if result>threshold:
                return True
    return False
# ---------------------------------
def extract_some_image(pproc,src,lists):# 圖片偵測演算法
    height, width ,_= src.shape
    kernel = np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(pproc, cv2.MORPH_CLOSE, kernel,2)
    _ , contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    record=[]
    for contour in contours:
        rct=cv2.boundingRect(contour)
        if rct[3]>height*0.1 and rct[2]>width*0.1:
            if not isoverlap(lists, rct, 0.9):
                if colorful(rct,src):
                    record.append(rct)
                elif not isoverlap(lists,rct,0.8):
                    record.append(rct)

    kernel1 = np.ones((3,3),np.uint8)
    mask1=cv2.morphologyEx(pproc, cv2.MORPH_OPEN, kernel1,2)
    _ , contours, _ = cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            rct=cv2.boundingRect(contour)
            if colorful(rct,src) and rct[3]>height*0.1 and rct[2] >0.1*width:
                record.append(rct)
    return record
#-------------------------------------------------------------------
def multi_process_form_image(img_source,images):# 對圖與表偵測執行多執行序
    src=cv2.imread('{}/{}'.format(img_source,images))
    pproc_outs=preprocess(src)
    lists=extract_table(pproc_outs,src)
    img_lists=extract_some_image(pproc_outs,src,lists)
    return lists,img_lists
# -----------------------------------------------------------------
def Form_DCT(indir,file_name,origin_img_dir):# 圖片表格表格偵測多執行序
    make_pic(indir,file_name,origin_img_dir)
    img_source='{}/{}/'.format(origin_img_dir,file_name.split('.')[0])
    page_form_lists=[None for name in os.listdir(img_source)]
    page_img_lists=[None for name in os.listdir(img_source)]
    lst=[]
    for images in os.listdir(img_source):
        res1=multi_process_form_image(img_source,images)
        lst.append([res1,images])
    for i in lst:

        this_page=int(i[1].split('.')[0].split('_')[1])-1
        lists,img_lists=i[0]
        page_form_lists[this_page]=lists
        page_img_lists[this_page]=img_lists
    print('Exteact Form Done!')
    return page_form_lists,page_img_lists
#---------------------------------------------
def htplusimage(pdf_data_path,pdf):# 內文偵測
    print("Manage {} ....".format(pdf))
    print('{}/{}'.format(pdf_data_path,pdf))
    fp = open('{}/{}'.format(pdf_data_path,pdf), 'rb')
    parser = PDFParser(fp)  #創建文檔分析器
    document = PDFDocument(parser)#創建pdf對象除存文檔結構
    # name=pdf.split('.')[0]
    if not document.is_extractable:
        print('NOT EXTRACTABLE')
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr=PDFResourceManager()#建立pdf資源管理器對象除存共享資源
        laparams=LAParams()#進行參數分析
        laparams.char_margin=1.5
        laparams.line_margin=0.05
        laparams.word_margin=0.1
        
        # print(laparams.char_margin)
        # print(laparams.line_margin)
        # print(laparams.word_margin)
        device=PDFPageAggregator(rsrcmgr,laparams=laparams)#建立pdf設備對象
        interpreter=PDFPageInterpreter(rsrcmgr,device)#創見一個pdf解釋器對象

        big_head=[]
        big_head_page=[]
        big_head_L=[]
        big_tail_L=[]
        big_tail_C=0
        page_rec_h=[]
        page_rec_w=[]
        all_record_txt=[]
        all_record_image=[]
        page_all=0

        for page in PDFPage.create_pages(document):
            
            # print('頁長',page.mediabox[3])
            page_rec_h.append(page.mediabox[3])
            page_rec_w.append(page.mediabox[2])
            interpreter.process_page(page)
            layout=device.get_result()#接受此頁面的ltpage對象
            min_height=10000
            all_rec_page_txt=[]
            record_image=[]
            for item in layout:# 獲取頁首以及最尾端之文字位置
                if isinstance(item,LTTextBoxHorizontal):
                    # print('yes')
                    tmp_h=item.get_text().replace("\n","").replace(" ","")
                    # print(tmp_h)
                    # print('---------------')
                    if tmp_h :
                        all_rec_page_txt.append(item)
                        maxxx=max(item.bbox[1],item.bbox[3])
                        minxx=min(item.bbox[1],item.bbox[3])
                        # print('minxx',minxx)
                        # print('page 7/8',page.mediabox[3]*7/8)
                        if minxx > page.mediabox[3]*7/8 :#and len(tmp_h)<30:

                            big_head.append(tmp_h)#1/8上的字
                            big_head_page.append(page_all)
                            big_head_L.append(minxx)#字的位置
                        elif maxxx < page.mediabox[3]/8 and maxxx>0.025*page.mediabox[3] and len(tmp_h)<10:
                            if maxxx < min_height:
                                min_height=maxxx
                elif isinstance(item,LTFigure) or isinstance(item,LTImage):
                    if int(item.bbox[1])>0.05*page.mediabox[3]:
                        record_image.append(item)

            page_all+=1
            # print('yse',page_all)
            all_record_image.append(record_image)
            all_record_txt.append(all_rec_page_txt)

            # print('min_height')
            # print(min_height)
            if min_height < 10000:
                big_tail_L.append(min_height+2)
                big_tail_C+=1
            else:
                big_tail_L.append(0)



        #頁眉位置
        print('page all:',page_all)
        # print('big head:',big_head)
        # print('len big head page',len(big_head_page))
        # print('big head line:',big_head_L)
        # print('---------------------')
        # print('height in every page:',page_rec_h)
        # print(Counter(page_rec_h).most_common(2))
        # common_page_h=Counter(page_rec_h).most_common(2)# find most common resolution height
        # print('common_page_h',common_page_h)
        # print('several page high:',len(common_page_h))


        # [(top_hh,_)]=Counter(page_rec_h).most_common(1)
        common_head_txt=Counter(big_head).most_common(3)
        # print('common_head_txt:',common_head_txt)

        max_head_txt=[]

        for i in common_head_txt:# find frequent head
            if page_all>5:
                if i[1] and i[1]>page_all-5:
                    max_head_txt.append(i[0])


        # print('max_head_txt:',max_head_txt)
        head_page=[]
        head_L=[]
        if max_head_txt:    
            for idx, txt in enumerate(big_head):
                for head_txt in max_head_txt:
                    if txt and SequenceMatcher(None, txt,head_txt).ratio()==1:
                        head_page.append(big_head_page[idx])
                        head_L.append(big_head_L[idx])
            delete=[]
            for idx,page in enumerate(head_page):
                if idx==len(head_page)-1: 
                    continue
                if page==head_page[idx+1]:
                    delete.append(idx)
            delete.reverse()
            for dele in delete:
                del head_page[dele]
                del head_L[dele]
            count=0
            for idx,page in enumerate(head_page):
                while count!=page:
                    head_L.insert(count,page_rec_h[count])
                    count+=1
                if count==page:
                    count=page+1

        # print('head page:',head_page)
        # print('head_L:',head_L)

        head=[]
        head_scale=[]
        if head_L:
            for idx, L in enumerate(head_L):
                head.append(page_rec_h[idx]-L)
                head_scale.append((page_rec_h[idx]-L)/page_rec_h[idx])


        if(len(head)<page_all):
            num=page_all-len(head)
            while num!=0:
                head.append(0)
                head_scale.append(0)
                num-=1

        #tail
        # print('tail:',big_tail_L)
        tail=[]
        tail_scale=[]
        if page_all<=3:
            num=page_all
            while num!=0:
                big_tail_L[page_all-num]=0
                tail.append(page_rec_h[page_all-num])
                tail_scale.append(1)
                num-=1

        else:
            for idx,big_tail in enumerate(big_tail_L):
                tail.append(page_rec_h[idx]-big_tail)
                tail_scale.append((page_rec_h[idx]-big_tail)/page_rec_h[idx])

        pdf_txt=[]
        pdf_txt_txt=[]
        head_txt=[]
        head_txt_page=[]
        tail_txt=[]
        tail_txt_page=[]
        # mid_word=(minus_avg_head+avg_tail)/2

        for idx,pagess in enumerate(all_record_txt):
            mid_word=(head[idx]+big_tail_L[idx])/2
            page_text=[]
            page_txt_txt=[]
            # print('head:',head[idx])
            # print('tail:',big_tail_L[idx])
            # print('mid_word:',mid_word)
            for items in pagess:
                if (items.bbox[1] > big_tail_L[idx] and page_rec_h[idx]-items.bbox[3] > head[idx]) or abs(items.bbox[1]-items.bbox[3])>0.05*page_rec_h[idx]:
                    page_txt_txt.append(items.get_text())
                    if page_rec_w[idx]>items.bbox[2]:
                        page_text.append([items.bbox[0]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[1])/page_rec_h[idx],items.bbox[2]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[3])/page_rec_h[idx]])
                    else:
                        page_text.append([items.bbox[0]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[1])/page_rec_h[idx],1.0,(page_rec_h[idx]-items.bbox[3])/page_rec_h[idx]])
                else:
                    if items.bbox[1]>mid_word:
                        head_txt.append(items.get_text().replace("\n","").replace(" ",""))
                        head_txt_page.append(idx)
                    else:
                        tail_txt.append(items.get_text().replace("\n","").replace(" ",""))
                        tail_txt_page.append(idx)

            pdf_txt.append(page_text)
            pdf_txt_txt.append(page_txt_txt)
        # ----------
        pdf_pic=[]
        for idx,pagess in enumerate(all_record_image):
            page_pic=[]
            for items in pagess:
                    if abs((items.bbox[1]-items.bbox[3]))>50 and abs((items.bbox[2]-items.bbox[0]))>50:
                        page_pic.append([items.bbox[0]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[1])/page_rec_h[idx],items.bbox[2]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[3])/page_rec_h[idx]])
            pdf_pic.append(page_pic)
        # print('pdf_txt_txt:',pdf_txt_txt)
        # print('----------------------')
        # print('pdf_txt:',pdf_txt)
        return pdf_txt,pdf_pic,pdf_txt_txt,head_scale,tail_scale,head_txt,tail_txt,head_txt_page,tail_txt_page,page_all


        
#----------------------------------------
def htplusimage_se(pdf_data_path,pdf):# 偵測表格中的內文
    print("Manage {} ....".format(pdf))
    fp = open('{}/{}'.format(pdf_data_path,pdf), 'rb')
    print('{}/{}'.format(pdf_data_path,pdf))
    parser = PDFParser(fp)  #創建文檔分析器
    document = PDFDocument(parser)#創建pdf對象除存文檔結構
    # name=pdf.split('.')[0]
    if not document.is_extractable:
        print('NOT EXTRACTABLE')
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr=PDFResourceManager()#建立pdf資源管理器對象除存共享資源
        laparams=LAParams()#進行參數分析
        laparams.char_margin=0.5
        laparams.line_margin=0.05
        laparams.word_margin=0.1
        
        device=PDFPageAggregator(rsrcmgr,laparams=laparams)#建立pdf設備對象
        interpreter=PDFPageInterpreter(rsrcmgr,device)#創見一個pdf解釋器對象

        big_head=[]
        big_head_L=[]
        big_tail_L=0
        big_tail_C=0
        page_rec_h=[]
        page_rec_w=[]
        all_record_txt=[]
        all_record_image=[]
        page_all=0

        for page in PDFPage.create_pages(document):
            page_all+=1
            page_rec_h.append(page.mediabox[3])
            page_rec_w.append(page.mediabox[2])
            interpreter.process_page(page)
            layout=device.get_result()#接受此頁面的ltpage對象
            min_height=10000
            all_rec_page_txt=[]
            record_image=[]
            for item in layout:# 獲取頁首以及最尾端之文字位置
                if isinstance(item,LTTextBoxHorizontal):
                    tmp_h=item.get_text().replace("\n","").replace(" ","")
                    if tmp_h :
                        all_rec_page_txt.append(item)
                        maxxx=max(item.bbox[1],item.bbox[3])
                        minxx=min(item.bbox[1],item.bbox[3])
                        if minxx > page.mediabox[3]*7/8 :#and len(tmp_h)<30:
                            big_head.append(tmp_h)
                            big_head_L.append(minxx)
                        elif maxxx < page.mediabox[3]/8 and maxxx>0.025*page.mediabox[3] and len(tmp_h)<30:
                            if maxxx < min_height:
                                min_height=maxxx
                elif isinstance(item,LTFigure) or isinstance(item,LTImage):
                    if int(item.bbox[1])>0.05*page.mediabox[3]:
                        record_image.append(item)

            all_record_image.append(record_image)
            all_record_txt.append(all_rec_page_txt)

            if min_height < 10000:
                big_tail_L+=min_height
                big_tail_C+=1

        pdf_txt=[]
        pdf_txt_txt=[]

        for idx,pagess in enumerate(all_record_txt):
            page_text=[]
            page_txt_txt=[]
            for items in pagess:
                page_txt_txt.append(items.get_text())
                if page_rec_w[idx]>items.bbox[2]:
                    page_text.append([items.bbox[0]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[1])/page_rec_h[idx],items.bbox[2]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[3])/page_rec_h[idx]])
                else:
                    page_text.append([items.bbox[0]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[1])/page_rec_h[idx],1.0,(page_rec_h[idx]-items.bbox[3])/page_rec_h[idx]])

            pdf_txt.append(page_text)
            pdf_txt_txt.append(page_txt_txt)
        # ----------
        pdf_pic=[]
        for idx,pagess in enumerate(all_record_image):
            page_pic=[]
            for items in pagess:
                    if abs((items.bbox[1]-items.bbox[3]))>50 and abs((items.bbox[2]-items.bbox[0]))>50:
                        page_pic.append([items.bbox[0]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[1])/page_rec_h[idx],items.bbox[2]/page_rec_w[idx],(page_rec_h[idx]-items.bbox[3])/page_rec_h[idx]])
            pdf_pic.append(page_pic)

        return pdf_txt,pdf_pic,pdf_txt_txt
#----------------------------------------
def checks(mask,rct,threshold):# 物件重疊之評估
    cnts=0
    maxx=max(rct[3],rct[1])
    minx=min(rct[3],rct[1])
    maxy=max(rct[0],rct[2])
    miny=min(rct[0],rct[2])
    for i in range(minx,maxx,2):
        for j in range(miny,maxy,2):
            if mask[i,j]:
                cnts+=1
    return cnts < (maxy-miny)*(maxx-minx)*threshold*0.125
#-----
def checks1(mask,rct,threshold):# 是否與mask重疊之評估
    # print('rct:',rct)
    cnts=0
    maxx=max(rct[3],rct[1])
    minx=min(rct[3],rct[1])
    maxy=max(rct[0],rct[2])
    miny=min(rct[0],rct[2])
    if maxx>mask.shape[0]:
        maxx=mask.shape[0]
    # print('mask.shape[0]:',mask.shape[0])
    # print('maxx:',maxx)
    if maxy>mask.shape[1]:
        maxy=mask.shape[1]
    # print('mask.shape[1]:',mask.shape[1])
    # print('maxy:',maxx)
    # print('--------------')
    tmp=None
    for i in range(minx,maxx,2):
        for j in range(miny,maxy,2):
            if mask[i,j]:
                tmp=mask[i,j]
                cnts+=1
    if tmp:
        return cnts < (maxy-miny)*(maxx-minx)*threshold*0.125,tmp-1
    else:
        return cnts < (maxy-miny)*(maxx-minx)*threshold*0.125,None
#-----
def chk_all_v1(mask,pthdwn,pthup,w0,w2):#find caption 搜索上下固定範圍
    POS=None #tiltle caption if exist [位置,第幾個物件]
    NEG=None #tail caption if exist[位置,第幾個物件]

    for i in range(w2+80,w0-20,-5):
        if 0<i<mask.shape[1]:
            if not POS:
                for j in range(0,+100,10):
                    if j+pthup < mask.shape[0]:
                        if mask[j+pthup,i]:
                            POS=[j,mask[j+pthup,i]-1]
                            break
            if not NEG:
                for j in range(0,-250,-10):
                    if j+pthdwn > 0:
                        if mask[j+pthdwn,i]:
                            NEG=[j,mask[j+pthdwn,i]-1]
                            break
        if POS and NEG:
            break
    if POS or NEG:
        return True,POS,NEG
    return False,None,None
#---
def merge_horizontal_form_image(h,w,img_lst,f_img_lst,form_lst,dir1,dir3,name,tmp,thresh_h,img3):#合併水平重疊之圖表
    boxes=[]
    # print('img_lst:',img_lst)
    # print('f_img_lst:',f_img_lst)
    # print('form_lst:',form_lst)
    # print('------------')
    # img3=cv2.rectangle(img3,(box[0],box[1]),(box[2],box[3]),(125,0,255),3,8)
    for i_box in img_lst:
        # print(i_box)
        # print('------------')
        notmatch=True
        box_tmp=[int(i_box[0]*w), int(min(i_box[1],i_box[3])*h), int(i_box[2]*w), int(max(i_box[1],i_box[3])*h)]
        if box_tmp[1]<thresh_h:
            continue
        for idx,box in enumerate(boxes):
            if min(box_tmp[2],box[2])<max(box_tmp[0],box[0]) or min(box_tmp[3],box[3])<max(box_tmp[1],box[1]):
                continue
            else:
                boxes[idx]=[min(box_tmp[0],box[0]),min(box_tmp[1],box[1]),max(box_tmp[2],box[2]),max(box_tmp[3],box[3])]
                notmatch=False
                break
        if notmatch:
            boxes.append(box_tmp)

    flags=[0 for _ in boxes]

    for f_i_box in f_img_lst:
        notmatch=True
        box_tmp=[f_i_box[0], f_i_box[1],f_i_box[0]+f_i_box[2],f_i_box[1]+f_i_box[3]]
        # print('box_tmp:',box_tmp)
        # print(box_tmp)
        # if box_tmp[1]<thresh_h:
        #     continue
        for idx,box in enumerate(boxes):
            if min(box_tmp[2],box[2])<max(box_tmp[0],box[0]) or min(box_tmp[3],box[3])<max(box_tmp[1],box[1]):
                continue
            else:
                boxes[idx]=[min(box_tmp[0],box[0]),min(box_tmp[1],box[1]),max(box_tmp[2],box[2]),max(box_tmp[3],box[3])]
                notmatch=False
                break
        if notmatch:
            boxes.append(box_tmp)
            flags.append(1)

    for f_box in form_lst:
        notmatch=True
        box_tmp=[f_box[0], f_box[1],f_box[0]+f_box[2],f_box[1]+f_box[3]]
        # if box_tmp[3]>thresh_h:
        #     continue
        for idx,box in enumerate(boxes):
            if min(box_tmp[2],box[2])<max(box_tmp[0],box[0]) or min(box_tmp[3],box[3])<max(box_tmp[1],box[1]):
                continue
            else:
                boxes[idx]=[min(box_tmp[0],box[0]),min(box_tmp[1],box[1]),max(box_tmp[2],box[2]),max(box_tmp[3],box[3])]
                notmatch=False
                break
        if notmatch:
            boxes.append(box_tmp)
            flags.append(2)

    while 1:
        if len(boxes)>1:
            ischange=False
            for idx,box in enumerate(boxes):
                for idx1,box1 in enumerate(boxes[idx+1:]):
                    if min(box1[2],box[2])<max(box1[0],box[0]) or min(box1[3],box[3])<max(box1[1],box[1]):
                        continue
                    else:
                        boxes[idx]=[min(box[0],box1[0]),min(box[1],box1[1]),max(box[2],box1[2]),max(box[3],box1[3])]
                        # boxes.remove(box1)
                        del boxes[idx+idx1+1]
                        flags[idx]=min(flags[idx],flags[idx+idx1+1])
                        # flags.remove(flags[idx+idx1+1])
                        del flags[idx+idx1+1]
                        ischange=True
                        break
                if ischange:
                    break
            if not ischange:
                break
        else:
            break

    mask=np.zeros((h,w), np.uint8)
    mask_big=np.zeros((h,w), np.uint8)
    #del error boxes
    del_boxes=[]
    for idx, box in enumerate(boxes):
        if (box[0]<0) or (box[1]<0) or (box[2]>w) or (box[3]>h):
            del_boxes.append(idx)
        # print(del_boxes)
        if len(del_boxes)>0:
            for i in sorted(del_boxes, reverse=True):
                del flags[i]
                del boxes[i]
        # print(len(boxes))

    big_lst_name=[]
    for idx,box in enumerate(boxes):
        # print('box:',box)
        # img3=cv2.rectangle(img3,(box[0],box[1]),(box[2],box[3]),(255,125,0),3,8)
        # cv2.imshow('img3',img3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if box[3]-box[1]<0.1*h:
            cv2.rectangle(mask_big, (box[0], box[1]), (box[2], box[3]), idx+1, cv2.FILLED, 8)
        cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), idx+1, cv2.FILLED, 8)
        if flags[idx]<2:
            big_lst_name.append("{}/{}/{}_{}_{}.jpg".format(dir3,name,name,tmp,idx))
        else:
            big_lst_name.append("{}/{}/{}_{}_{}.jpg".format(dir1,name,name,tmp,idx))


    return mask,mask_big,big_lst_name,boxes,flags

#------------------------------------------------
def clean_all_txt_in_item(page,boxes,new_filt_box,new_filt_txts,flags):
    #圖片表格合併後，將併入圖片表格中的文字從txt中修正
    cnt=0
    for idx,txt_box in enumerate(new_filt_box):
        # print(txt_box)
        # print('---------------')
        isov=False
        for idx1,box in enumerate(boxes):
            if min(txt_box[2],box[2])<max(txt_box[0],box[0]) or min(txt_box[3],box[3])<max(txt_box[1],box[1]):
                continue
            else:
                isov=True
                boxes[idx1]=[min(box[0],txt_box[0]),min(box[1],txt_box[1]),max(box[2],txt_box[2]),max(box[3],txt_box[3])]
                # if(flags[idx1]==2):
                #     cv2.rectangle(img,(boxes[idx1][0],boxes[idx1][1]),(boxes[idx1][2],boxes[idx1][3]),(125,0,125),8,8)
                break

        if isov:
            del new_filt_box[idx-cnt]
            del new_filt_txts[idx-cnt]
            cnt+=1
    return boxes,new_filt_box,new_filt_txts

#-------------------------------------------------------------------
def clean_content(page_num,txt_lst,val_txt_lst,all_dir,name,tmp,img1):#清除目錄
    xtf=False
    for idxs,txt_box in enumerate(txt_lst):
        matchObj2=re.search(r'(\.\.\.\.\.\.\.\.\.\s*\d+\s*$)',val_txt_lst[idxs])
        if (matchObj2):
            xtf=True
    if (xtf==True):
        # print(page_num)
        cv2.imwrite("{}/{}/{}_{}.jpg".format(all_dir[3],name,name,tmp), img1)
        with open("{}/{}/{}_{}.txt".format(all_dir[1],name,name,tmp), "w",encoding="utf-8") as text_file:
            text_file.write('')
            text_file.close()
        with open("clean_TXT/{}/{}_{}.txt".format(name,name,tmp), "w",encoding="utf-8") as text_file:
            text_file.write('')
            text_file.close()
        with open("output_json/{}/{}.json".format(name,page_num+1), "w",encoding="utf-8") as list_obj:
            list_obj.write('')
            list_obj.close()
        return True
    else: 
        return False
#-------------------------------------------------
def horizonmerge_find_caption(txt_lst,val_txt_lst,img1,mask_with_txt,title,mask,boxes,flags,title_box,tail,tail_box,all_txt_after_filt,all_txt_num,every_page_min_height,w,h,img3):
    # print('pre tail:',tail)
    # mask2=cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # img3=cv2.addWeighted(img3,0.5,mask2,0.5,0)
    # note_txt=[]
    # note_box=[]
    for idxs,txt_box in enumerate(txt_lst):
        #物件之格式歸一 可能有1 <=> 3的狀況
        # print('------------------')
        # print(txt_box)
        box=[int(min(txt_box[0],txt_box[2])*w), int(min(txt_box[1],txt_box[3])*h) ,int(max(txt_box[0],txt_box[2])*w),int(max(txt_box[1],txt_box[3])*h)]#1 small ,3 big
        # print('img3:',img3.shape)
        # print('mask2:',mask2.shape)
        # img3=cv2.rectangle(img3,(box[0],box[1]),(box[2],box[3]),(125,0,255),3,8)
        # cv2.imshow('img3',img3)

        if every_page_min_height>box[1]:
            every_page_min_height=box[1]
        isov,which=checks1(mask,(box[0],box[1],box[2],box[3]),0.4)# 查看是否已經與mask重疊
        # print(isov)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if isov: #若不重疊
            xxx=''.join(val_txt_lst[idxs].split())
            flag,POS,NEG=chk_all_v1(mask,box[1],box[3],box[0],box[2])# 判斷文字上下範圍左右有無mask=1 若有代表可能為caption
            #一次偵測pos 跟neg 若沒有偵測到則返回null 返回值[位置,第幾個物件]
            # print(flag)
            if flag and len(xxx):
                if POS:# word up image
                    numb=POS[0]
                    TAGS=POS[1]
                    #------
                    if numb<11:
                        if (box[2]-box[0])*(box[3]-box[1])<2*w:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,128,111), 3, 8)
                            if flags[TAGS]==2:
                                if boxes[TAGS][3]-boxes[TAGS][1]>0.1*h:
                                    boxes[TAGS]=[min(boxes[TAGS][0],box[0]),min(boxes[TAGS][1],box[1]),max(boxes[TAGS][2],box[2]),max(boxes[TAGS][3],box[3])]
                            else:
                                boxes[TAGS]=[min(boxes[TAGS][0],box[0]),min(boxes[TAGS][1],box[1]),max(boxes[TAGS][2],box[2]),max(boxes[TAGS][3],box[3])]
                            continue
                    #-------
                    if numb<=80:
                        if xxx[0]=='图':
                            # if (txt_box[2]*w-txt_box[0]*w)<0.7*mask.shape[1]:
                                cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                                cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                                title[TAGS]=xxx
                                title_box[TAGS]=box
                                #print(xxx)
                                continue
                    #-----
                    if numb<=100:
                        if (xxx[0]=='表'  and xxx[-1]!='。') or xxx[-1]=='表':
                            if (box[2]-box[0])<0.65*w or numb<21:
                                cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                                cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                                title[TAGS]=xxx
                                title_box[TAGS]=box
                                continue

                        elif len(xxx)<9 and xxx[:2]=='单位':#找單位開頭且字段小
                            cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                            continue
                        elif (box[2]-box[0])<0.5*w and 7<len(xxx)<50 and  abs((box[0]+box[2]-w)/2)<w/15 and '。' not in xxx:#找附近置中文字段
                            # cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (123,255,0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                            continue
                    if numb<=20:
                        if '单位' in xxx :
                            cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                            continue
                #-------
                if NEG:# word down image
                    numb=NEG[0]
                    TAGS=NEG[1]
                    #------
                    if numb>-11:
                        if (box[2]-box[0])*(box[3]-box[1])<1*w:
                            # cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,128,111), 3, 8)
                            if flags[TAGS]==2:
                                if boxes[TAGS][3]-boxes[TAGS][1]>0.1*h:
                            
                                    boxes[TAGS]=[min(boxes[TAGS][0],box[0]),min(boxes[TAGS][1],box[1]),max(boxes[TAGS][2],box[2]),max(boxes[TAGS][3],box[3])]
                            else:
                                    boxes[TAGS]=[min(boxes[TAGS][0],box[0]),min(boxes[TAGS][1],box[1]),max(boxes[TAGS][2],box[2]),max(boxes[TAGS][3],box[3])]
                            continue

                    if numb >-21:# 較近的可以較鬆散
                        if '来源' in xxx[:7]:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                            # caption_tmp.append([box[0],box[1]])
                            tail[TAGS]=xxx
                            tail_box[TAGS]=box
                            continue

                    if xxx[0]=='图':
                        if box[2]-box[0]<0.7*w:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                            # title[TAGS]=xxx
                            continue

                    if numb>=-250:
                        if len(xxx)>5 and box[2]-box[0]< 0.7*w:
                            if '来源' in xxx[:7]:
                                cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                                cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                                tail[TAGS]=xxx
                                tail_box[TAGS]=box
                                continue

                    # if '注' in xxx[:1]:
                    #     cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                    #     cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                    #     # img2=cv2.resize(img1, (680, 960))
                    #     # cv2.imshow('test',img2)
                    #     # cv2.waitKey(0)
                    #     # cv2.destroyAllWindows()
                    #     # note_box.append(box)
                    #     # note_txt.append(xxx)
                    #     tail[TAGS]=xxx
                    #     tail_box[TAGS]=box
                    #     continue

            if len(xxx)<10 and '单位：' in xxx :
                cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
                cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2],box[3]), 255, cv2.FILLED, 8)
                continue

            
            all_txt_after_filt.append(box)
            all_txt_num.append(val_txt_lst[idxs])

        else:# 若有文字與物件重疊 就把文字與物件合併擴張
            boxes[which]=[min(boxes[which][0],box[0]),min(boxes[which][1],box[1]),max(boxes[which][2],box[2]),max(boxes[which][3],box[3])]

    return img1,mask_with_txt,title,title_box,tail,tail_box,all_txt_after_filt,all_txt_num

#----------------------------------利用剛剛找初步的caption頭尾 來修補無法正確偵測的圖表的範圍
def expand_imgform_by_caption(boxes,title_box,tail_box,all_txt_after_filt,all_txt_num):
    for idx,box in enumerate(boxes):#針對caption首尾內對物件進行文字搜索擴張
        if title_box[idx] and tail_box[idx]:
            tmp_box=box[:]
            del_lst=[]
            for idx1,txt in enumerate(list(all_txt_after_filt)):
                if title_box[idx][1]<txt[1]<tail_box[idx][1]:
                    if title_box[idx][2]-title_box[idx][0]<box[2]-box[0]:
                        if box[0]<(txt[0]+txt[2])>>1 <box[2]:
                            del_lst.append(idx1)
                            tmp_box=[min(txt[0],tmp_box[0]),min(txt[1],tmp_box[1]),max(txt[2],tmp_box[2]),max(txt[3],tmp_box[3])]
                    else:
                        if title_box[idx][0]<(txt[0]+txt[2])>>1 <title_box[idx][2]:
                            del_lst.append(idx1)
                            tmp_box=[min(txt[0],tmp_box[0]),min(txt[1],tmp_box[1]),max(txt[2],tmp_box[2]),max(txt[3],tmp_box[3])]
            for idx1,i in enumerate(del_lst):
                del all_txt_after_filt[i-idx1]
                del all_txt_num[i-idx1]
            boxes[idx]=tmp_box
    return boxes,all_txt_after_filt,all_txt_num

#-------------------------------------------------
def advance_fix_tail_caption(img1,tail_box,new_filt_box):
    for idxs,tailbox in enumerate(tail_box):
        if(tailbox==None):
            continue
        for idx,txtbox in enumerate(new_filt_box):
            if((txtbox[1]-tailbox[3])>0) and ((txtbox[1]-tailbox[3])<10):
                if(tailbox[2]-tailbox[0])>(txtbox[2]-txtbox[0]) and (txtbox[3]-txtbox[1]<20):
                    tailbox[3]=txtbox[3]
                    cv2.rectangle(img1, (tailbox[0], tailbox[1]), (tailbox[2],tailbox[3]), (0,128,0), 3, 8)
    return img1,tail_box

#-------------------------------------------------
def filt_tail_caption_in_txt(tail_box,new_filt_box,new_filt_txts):
    for idx,tailbox in enumerate(tail_box):
        cnt=0
        for idxs,txt_box in enumerate(new_filt_box):
            if(tail_box[idx]==None):
                continue
            if((tailbox[0]<=txt_box[0]) and(tailbox[2]>=txt_box[2]) and(tailbox[1]<=txt_box[1]) and(tailbox[3]>=txt_box[3])):
                del new_filt_box[idxs-cnt]
                del new_filt_txts[idxs-cnt]
                cnt+=1
    return new_filt_box,new_filt_txts

#-----------------------------------------------
def merge_imgform_overlapped(boxes,flags,big_lst_name,title):
    while 1:
        if len(boxes)>1:
            ischange=False
            for idx,box in enumerate(boxes):
                for idx1,box1 in enumerate(boxes[idx+1:]):
                    if min(box1[2],box[2])+5<max(box1[0],box[0]) or min(box1[3],box[3])+15<max(box1[1],box[1]):
                        continue
                    else:
                        boxes[idx]=[min(box[0],box1[0]),min(box[1],box1[1]),max(box[2],box1[2]),max(box[3],box1[3])]
                        del boxes[idx+idx1+1]
                        if flags[idx]>flags[idx+idx1+1]: # 以圖表為優先，相較於無格線表格 flag 1 2是圖表 ，無格現表格3
                            flags[idx]=flags[idx+idx1+1]
                            big_lst_name[idx]=big_lst_name[idx+idx1+1]
                        
                        del flags[idx+idx1+1]
                        
                        if title[idx]=='':
                            title[idx]=title[idx+idx1+1]
                        del title[idx+idx1+1]
                        del big_lst_name[idx+idx1+1]
                        ischange=True
                        break
                if ischange:
                    break
            if not ischange:
                break
        else:
            break
    return boxes,flags,big_lst_name,title

#------------------------------------------------
def correct_wrongimg_to_form(boxes,flags,big_lst_name,title,txt_lst_se,val_txt_lst_se,w,h,img,all_dir,name,tmp):
    for idxs,box in enumerate(boxes):
        all_area=0
        if(flags[idxs]==0 or flags[idxs]==1):
            txt_num=0
            txt_area=0
            all_area=(box[2]-box[0])*(box[3]-box[1])
            for idx,txt_box in enumerate(txt_lst_se):
                if min(txt_box[2]*w,box[2])<max(txt_box[0]*w,box[0]) or min(txt_box[1]*h,box[3])<max(txt_box[3]*h,box[1]):
                    continue
                else:
                    txt_area+=(txt_box[2]-txt_box[0])*(txt_box[1]-txt_box[3])*h*w
                    txt_num+=1
                
            if(txt_area/all_area>0.1):       
                if(txt_num>=10):
                    print('convert img to form')
                    flags[idxs]=2
                    big_lst_name[idxs]=big_lst_name[idxs].replace('IMAGE','FORM')
    return boxes,flags,big_lst_name,title

#-------------------------------------------------
def distingish_form_2column(new_filt_box,boxes,flags,title,title_box,big_lst_name):
    wrong_form=[]
    wrong_form_num=0
    for idxs,box in enumerate(boxes):
        if (flags[idxs]==2):
            txt_area=0
            for idx,txt_box in enumerate(new_filt_box):
                if ((box[0]<=txt_box[0])and(box[2]>=txt_box[2])and(box[1]<=txt_box[1])and(box[3]>=txt_box[3])):
                    txt_area+=(txt_box[2]-txt_box[0])*(txt_box[3]-txt_box[1])
            if (txt_area>((box[2]-box[0])*(box[3]-box[1])*0.4)):#if true 2column
                wrong_form.append(idxs)
                wrong_form_num+=1
    if wrong_form_num!=0:
        for i in sorted(wrong_form, reverse=True):
            del flags[i]
            del boxes[i]
            del title[i]
            del title_box[i]
            del big_lst_name[i]
    return boxes,flags,title,title_box,big_lst_name

#-------------------------------------------------
def del_caption_in_txt(boxes,flags,new_filt_box,new_filt_txts):
    wrong_txt=[]
    wrong_txt_num=0
    for idxs,box in enumerate(boxes):
        if(flags[idxs]!=2 and flags[idxs]!=3):
            for idx,txtbox in enumerate(new_filt_box):
                if(txtbox==None):
                    break
                if((txtbox[2]-txtbox[0])<((box[2]-box[0])/2))and(txtbox[1]>=box[3]):
                    if((txtbox[1]-box[3])<15)and((txtbox[3]-txtbox[1])<20):
                        wrong_txt.append(idx)
                        wrong_txt_num+=1
                        
    if wrong_txt_num!=0:
        for i in sorted(wrong_txt, reverse=True):
            del new_filt_box[i]
            del new_filt_txts[i]
    return new_filt_box,new_filt_txts

#-------------------------------------------------
def distingish_image_2column(new_filt_box,boxes,flags,title,title_box,big_lst_name):
    wrong_img=[]
    wrong_img_num=0
    for idxs,box in enumerate(boxes):
        if (flags[idxs]!=2 and flags[idxs]!=3):
            txt_area=0
            for idx,txt_box in enumerate(new_filt_box):
                if ((box[0]<=txt_box[0])and(box[2]>=txt_box[2])and(box[1]<=txt_box[1])and(box[3]>=txt_box[3])):
                    txt_area+=(txt_box[2]-txt_box[0])*(txt_box[3]-txt_box[1])
            if (txt_area>((box[2]-box[0])*(box[3]-box[1])*0.3)):#if true 2column
                wrong_img.append(idxs)
                wrong_img_num+=1
    if wrong_img_num!=0:
        for i in sorted(wrong_img, reverse=True):
            del boxes[i]
            del flags[i]
            del title[i]
            del title_box[i]
            del big_lst_name[i]
    return boxes,flags,title,title_box,big_lst_name

#---------------------------------------------------
def distingish_nonlineform_2column(boxes,flags,title,title_box,big_lst_name,new_filt_box,new_filt_txts,img1):
    list_ind=[]
    list_ind_num=0
    acut=False
    for idxs,box in enumerate(boxes):
        if flags[idxs]==3:
            ans,LeftorRight,line_postion=dist_nonline_2col(img1[box[1]:box[3],box[0]:box[2]])
            if ans==True:
                list_ind.append(idxs)
                list_ind_num+=1
                cut_postion=line_postion+box[0]
    #----------------
                if acut==False:
                    acut=True
                    if LeftorRight==True:
                        cnt=0
                        for idx,txt_box in enumerate(new_filt_box):
                            if txt_box[2]<cut_postion:
                                del new_filt_box[idx-cnt]
                                del new_filt_txts[idx-cnt]
                                cnt+=1
                    else:
                        cnt=0
                        for idx,txt_box in enumerate(new_filt_box):
                            if txt_box[0]>cut_postion:
                                del new_filt_box[idx-cnt]
                                del new_filt_txts[idx-cnt]
                                cnt+=1
    #-------------------
    if list_ind_num!=0:
        for i in sorted(list_ind, reverse=True):
            del flags[i]
            del boxes[i]
            del title[i]
            del title_box[i]
            del big_lst_name[i]
    return boxes,flags,title,title_box,big_lst_name,new_filt_box,new_filt_txts

#-------------------------------------------------
def expand_image_by_caption(boxes,flags,title_box,tail_box):
    for idxs,box in enumerate(boxes):
        if (flags[idxs]!=2 and flags[idxs]!=3):
            title_long=500
            near_title=0
            tail_long=500
            near_tail=0
            if(title_box==None or tail_box==None):
                break
            for idx,titlebox in enumerate(title_box):
                if (titlebox==None):
                    continue
                if (box[1]>titlebox[3]):
                    if((box[1]-titlebox[3])<title_long):
                        title_long=box[1]-titlebox[3]
                        near_title=idx
            for idx_2,tailbox in enumerate(tail_box):
                if (tailbox==None):
                    continue
                if (tailbox[1]>box[3]):
                    if((tailbox[1]-box[3])<tail_long):
                        tail_long=tailbox[1]-box[3]
                        near_tail=idx_2
                
            if (title_long<250 and tail_long<250):
                box[1]=box[1]-title_long+1
                box[3]=box[3]+tail_long-1
    return boxes

#-------------------------------------------------
def del_firstpage_wrong_imgform(boxes,flags,title,title_box,big_lst_name,h):
    deletboxesnum=0
    deletboxes=[]
    for idx,box in enumerate(boxes):
        # print(box)
        # print('---------------****************************----------------------')
        hd=box[3]-box[1]
        wd=box[2]-box[0]
        if (wd/hd)>9:
            if box[3]<(0.18*h):
                deletboxesnum+=1
                deletboxes.append(idx)
    if deletboxesnum!=0:
        for i in sorted(deletboxes, reverse=True):
            del flags[i]
            del boxes[i]
            del title[i]
            del title_box[i]
            del big_lst_name[i]
    return boxes,flags,title,title_box,big_lst_name    

#-------------------------------------------------
def del_watermark_on_certain_page(page_num,watermark_page,boxes,flags,title,title_box,big_lst_name,h):
    if (page_num==watermark_page):
        deletboxesnum=0
        deletboxes2=[]
        for idx,box in enumerate(boxes):
            # print(box)
            hd=box[3]-box[1]
            wd=box[2]-box[0]
            if (wd/hd)>=9:
                if box[1]>(0.8*h):
                    deletboxesnum+=1
                    deletboxes2.append(idx)
        if deletboxesnum!=0:
            for i in sorted(deletboxes2, reverse=True):
                del flags[i]
                del boxes[i]
                del title[i]
                del title_box[i]
                del big_lst_name[i]
    return boxes,flags,title,title_box,big_lst_name

#-------------------------------------------------
def resize_output_form(boxes,flags,title,big_lst_name,img,w,h,name,tmp,all_dir,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,img3):
    all_form_content=[]
    boxes_new=boxes[:]
    # print('boxes',boxes)
    a=0
    for idx,box in enumerate(boxes):
        
        if flags[idx+a]==0 or flags[idx+a]==1:
            continue
        boxes_old=box[:]
        src=(img[box[1]:box[3],box[0]:box[2]])
        src3=src.copy()
        

        img_h,img_w,_=src3.shape
        for j in range(img_w):
            for i in range(img_h):
                (B,G,R)=src[i,j]
                if B<100 or G<100 or R<100:
                    src3[i,j]=(0,0,0)
                else:
                    src3[i,j]=(255,255,255)


        #改善虛線表格
        gray = cv2.cvtColor(src3, cv2.COLOR_BGR2GRAY)
        # img2=cv2.resize(gray, (680, 960))
        # cv2.imshow('test',img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()      

        #腐蝕後再擴張
        kernel_22 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        eroded = cv2.erode(gray,kernel_22)
        dilated = cv2.dilate(eroded,kernel_22)

        _ ,outs = cv2.threshold(255-dilated,50,255,cv2.THRESH_BINARY) 
        kernel = np.ones((3,3),np.uint8)
        formimages = cv2.morphologyEx(outs, cv2.MORPH_CLOSE, kernel)
        height_f, width_f = formimages.shape
        area = height_f*width_f*0.2
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(int(w / 10), 1));
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h /55)));
        out_1d_hor=cv2.morphologyEx(formimages, cv2.MORPH_OPEN, horizontalStructure)
        out_1d_ver=cv2.morphologyEx(formimages, cv2.MORPH_OPEN, verticalStructure)

        mask=out_1d_hor+out_1d_ver
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,2)
        # cv2.imshow('test1',mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #---------------尋找是否有多個表格被融為一個
        _ , contours_2, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        num_form=0#表格個數
        num_form_box=[]#各表格外框座標
        

        for contour in contours_2:
            rct=cv2.boundingRect(contour)
            if (rct[2]*rct[3])>area:
                cv2.rectangle(mask,(rct[0],rct[1]),(rct[0]+rct[2],rct[1]+rct[3]),(255,255,255),3)
                num_form+=1
                num_form_box.append([rct[0],rct[1],rct[0]+rct[2],rct[1]+rct[3]])

        idx_num_lst=[]
        for idx2,lst_name in enumerate(big_lst_name):
            # print('lst_name',lst_name)
            num=re.sub(r'.jpg','',lst_name)
            num=num[-1]
            if num.isdigit():
                idx_num_lst.append(int(num))
            # print('num',num)
            # print('-----------------')
        # print('idx_num_lst',idx_num_lst)
    
        if num_form>0:
            # print('num_form_box before',num_form_box)
            num_form_box.reverse()
            # print('num_form_box after',num_form_box)
            # print('-------------------------')
            idx_num=max(idx_num_lst)
            # print('idx_num',idx_num)

            # print('old big_lst_name',big_lst_name)
            #分割表格
            new_form_box=[]
            for idxs, form_box in enumerate(num_form_box):
                form_mask=mask[form_box[1]:form_box[3],form_box[0]:form_box[2]]
                if idxs==0:#第一個表格  
                    # print('1')
                    boxes_new[idx+a]=[form_box[0]+box[0]-3,form_box[1]+box[1],form_box[2]+box[0]+3,form_box[3]+box[1]]
                    new_form_box.append(boxes_new[idx+a])
                    form_name=big_lst_name[idx+a]
                    all_form_content=form_output(all_form_content,boxes_new[idx+a],form_mask,form_name,txt_lst_se,val_txt_lst_se,w,h,name,img)
                    # print('idx_num1',idx_num)
                else:#第二個以上表格
                    # print('2')
                    print('double form seperate')
                    cv2.imwrite('{}.jpg'.format(big_lst_name[idx+a].replace('FORM','IMAGE')), mask)
                    second_box=[form_box[0]+box[0]-3,form_box[1]+box[1],form_box[2]+box[0]+3,form_box[3]+box[1]]
                    new_form_box.append(second_box)
                    boxes_new.insert(idx+a+1,second_box)
                    flags.insert(idx+a+1,2)
                    # print('idx_num2',idx_num)
                    big_lst_name.insert(idx+a+1,"{}/{}/{}_{}_{}.jpg".format(all_dir[0],name,name,tmp,idx_num+1))
                    form_name="{}/{}/{}_{}_{}.jpg".format(all_dir[0],name,name,tmp,idx_num+1)
                    all_form_content=form_output(all_form_content,second_box,form_mask,form_name,txt_lst_se,val_txt_lst_se,w,h,name,img)
                    title.insert(idx+a+1,'')
                    a+=1
                    # idx_num+=1 
            new_filt_box,new_filt_txts=restore_txt(boxes_old,new_form_box,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h,img3) 
    # print('new big_lst_name',big_lst_name)
    # print('boxes_new',boxes_new)
    return boxes_new,flags,title,big_lst_name,new_filt_box,new_filt_txts,all_form_content

#-------------------------------------------------
def restore_txt(boxes_old,new_form_box,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h,img3):
    # print('boxes_old',boxes_old)
    # print('new_form_box',new_form_box)
    # print('--------------------------')
    for idxx,txt in enumerate(txt_lst_se):
        if(txt[1]*h<boxes_old[1])|(boxes_old[3]<txt[3]*h):
            continue
        else:   
            # print(val_txt_lst_se[idxx])
            if len(new_form_box)==1:
                if(txt[1]*h<=new_form_box[0][1]):
                    if(re.search(r'(^单位.)',val_txt_lst_se[idxx])):
                        continue
                    # print(val_txt_lst_se[idxx])
                    new_filt_box,new_filt_txts=get_back_txt(idxx,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h)
                elif(new_form_box[0][3]<=txt[3]*h):
                    # print(val_txt_lst_se[idxx])
                    new_filt_box,new_filt_txts=get_back_txt(idxx,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h)
           
            else:#只針對雙表格
                if(txt[1]*h<=new_form_box[0][1]):
                    if(re.search(r'(^单位.)',val_txt_lst_se[idxx])):
                        continue

                    new_filt_box,new_filt_txts=get_back_txt(idxx,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h)

                elif((new_form_box[0][3]<=txt[3]*h)and(txt[1]*h<=new_form_box[1][1])):
                    if(re.search(r'(^单位.)',val_txt_lst_se[idxx])):
                        continue
                    new_filt_box,new_filt_txts=get_back_txt(idxx,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h)

                elif(new_form_box[1][3]<=txt[3]*h):
                    new_filt_box,new_filt_txts=get_back_txt(idxx,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h)

    return new_filt_box,new_filt_txts

#------------------------------------------------
def form_output(all_form_content,form_box,form_mask,form_name,txt_lst_se,val_txt_lst_se,w,h,name,img):
    # print('form_box',form_box)
    # print('------------------------')
    box=form_box
    formtxtboxes=[]#所有方框
    formtxtboxes_txt=[]#所有方框各自的文字段
    form_high_txt=h
    form_low_txt=0
    height_f, width_f = form_mask.shape
    area = height_f*width_f*0.7
    kernel = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(form_mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.dilate(mask,kernel2)
    # cv2.imshow('test1',mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret,thresh2 = cv2.threshold(mask,127,255,cv2.THRESH_BINARY_INV)
    _ , contours2, _ = cv2.findContours(thresh2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    formnum=0
    formboxtxt=str()
    this_form_content=[]
    all_rct=[]
    for contour in contours2:

        rct=(cv2.boundingRect(contour))
        if len(contour)==4:
            all_rct.append(rct)

        if len(contour)>4:
            mask2=mask[rct[1]-3:rct[1]+rct[3]-3,rct[0]-3:rct[0]+rct[2]+3]
            lines=cv2.HoughLinesP(mask2, 1, np.pi/18 ,100, 50, 100)
            # print('lines:',lines)
            # print('len',len(lines))
            if  lines is None :
                all_rct.append(rct)
                continue

            horizon=[]
            vertical=[]
            for _,line in enumerate(lines):
                # print(line[0])
                x1=line[0][0]
                y1=line[0][1]
                x2=line[0][2]
                y2=line[0][3]
                # print('(x1,y1):(',x1,y1,'),(x2,y2):(',x2,y2,')')
                
                if y2==y1:
                    horizon.append([x1,y1,x2,y2])
                if x2==x1:
                    vertical.append([x1,y1,x2,y2])
            horizon=sorted(horizon, key=lambda s: s[1])
            vertical=sorted(vertical, key=lambda s: s[0])
            if len(vertical)<1 or len(horizon)<1:
                all_rct.append(rct)
                continue

            max_hor=horizon[0][2]
            min_hor=horizon[0][0]    
            for idx,hor in enumerate(horizon):
                if idx<len(horizon)-1:
                    if horizon[idx+1][2]>hor[2]:
                        max_hor=horizon[idx+1][2]
                    if horizon[idx+1][0]<hor[0]:
                        min_hor=horizon[idx+1][0]
            

            max_ver=vertical[0][1]
            min_ver=vertical[0][3]    
            for idx,ver in enumerate(vertical):
                if idx<len(vertical)-1:
                    if vertical[idx+1][1]>ver[1]:
                        max_ver=vertical[idx+1][1]
                    if vertical[idx+1][3]<ver[3]:
                        min_ver=vertical[idx+1][3]
            # print(max_ver)
            # print(min_ver)
            new_mask=np.ones(mask2.shape, np.uint8)*255
            for hor in horizon:
                if hor[2]!=max_hor:
                    hor[2]=max_hor
                if hor[0]!=min_hor:
                    hor[0]=min_hor
                cv2.line(new_mask, (hor[0], hor[3]), (hor[2],hor[1] ), 0, 3)
            # print(horizon)

            for ver in vertical:
                if ver[1]!=max_ver:
                    ver[1]=max_ver
                if ver[3]!=min_ver:
                    ver[3]=min_ver
                cv2.line(new_mask, (ver[0], ver[3]), (ver[2],ver[1] ), 0, 3)
            # print(vertical)
            # cv2.imshow('dst_mask2',new_mask)
            # new_mask=cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)
            _ , contours2, _ = cv2.findContours(new_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            # print('all:',rct[2]*rct[3])
            for contour in contours2:
                
                rct2=(cv2.boundingRect(contour))

                # print(rct2[2]*rct2[3])
                # print('-----------')
                if(rct2[2]*rct2[3]>rct[2]*rct[3]*0.05):
                    all_rct.append((rct[0]+rct2[0]-3,rct[1]+rct2[1]-4,rct2[2]+1,rct2[3]+1))
    # print(all_rct)
    for rct in all_rct: 
        # if len(all_rct)==1:
        #     formtxtboxes.append([box[0]+rct[0],box[1]+rct[1],box[0]+rct[0]+rct[2],box[1]+rct[1]+rct[3]])
        #     formtxtboxes_txt.append(formboxtxt)

        if (rct[2]*rct[3])<(area):

            #外框不處理
            if (0 in rct) or (height_f in [rct[1]+rct[3]]) or (width_f in [rct[0]+rct[2]]):
                continue

            #內框找座標
            formnum+=1
            # print([box[0]+rct[0],box[1]+rct[1],box[0]+rct[0]+rct[2],box[1]+rct[1]+rct[3]])
            if (box[0]+rct[0]+rct[2])>box[2]:
                print('wrong')
            if (box[1]+rct[1]+rct[3])>box[3]:
                print('wrong')
            if (box[1]+rct[1])<form_high_txt:
                form_high_txt=(box[1]+rct[1])
            if (box[1]+rct[1]+rct[3])>form_low_txt:
                form_low_txt=(box[1]+rct[1]+rct[3])
            formtxtboxes.append([box[0]+rct[0],box[1]+rct[1],box[0]+rct[0]+rct[2],box[1]+rct[1]+rct[3]])
            formtxtboxes_txt.append(formboxtxt)

    # print('formtxt',formtxtboxes_txt)
    #-----------------------        
    del_formboxlist=[]
    for idxs,formboxes in enumerate(formtxtboxes):
        if idxs in del_formboxlist:
            continue
        for idxs1,formboxes1 in enumerate(formtxtboxes):
            if idxs1 in del_formboxlist:
                continue
            if idxs1 ==idxs:
                continue
            if (min(formboxes[2],formboxes1[2])<max(formboxes[0],formboxes1[0])) or (min(formboxes[3],formboxes1[3])<max(formboxes[1],formboxes1[1])):
                continue
            # print(idxs)
            # print(idxs1)
            if ((formboxes[2]-formboxes[0])*(formboxes[3]-formboxes[1]))>((formboxes1[2]-formboxes1[0])*(formboxes1[3]-formboxes1[1])):
                del_formboxlist.append(idxs)
            else:
                del_formboxlist.append(idxs1)
    if len(del_formboxlist)!=0:
        for i in sorted(del_formboxlist, reverse=True):
            del formtxtboxes[i]
            del formtxtboxes_txt[i]
    #del error box
    del_formboxlist=[]
    for idxs,formboxes in enumerate(formtxtboxes):
        if (formboxes[2]-formboxes[0])<20*0.41:
            del_formboxlist.append(idxs)
            continue
        if (formboxes[3]-formboxes[1])<15*0.41:
            del_formboxlist.append(idxs)
            continue
        if ((formboxes[2]-formboxes[0])/(formboxes[3]-formboxes[1]))>160:
            del_formboxlist.append(idxs)
    if len(del_formboxlist)!=0:
        for i in sorted(del_formboxlist, reverse=True):
    # print(formtxtboxes[i])
            del formtxtboxes[i]
            del formtxtboxes_txt[i]
    #------------------
    
    if len(formtxtboxes)>1:
        imgfmname=re.sub(r'\.','',form_name)
        imgfmname=re.sub(r'jpg','',imgfmname)
        imgfmname=re.sub(r'/','',imgfmname)
        imgfmname=re.sub(name,'',imgfmname)
        #print(imgfmname)
        #表格內的文字塊
        formtxtbox=[]
        formcontent=[]
        
        for idxs,txtbox in enumerate (txt_lst_se):
            if (box[0]<=txtbox[0]*w and box[1]<=txtbox[3]*h) and ((box[2]+10)>=txtbox[2]*w and box[3]>=txtbox[1]*h):
                if (txtbox[1]*h)-10*0.41<form_high_txt:
                    continue
                if (txtbox[3]*h)+10*0.41>form_low_txt:
                    continue
                formtxtbox.append(txtbox)
                formcontent.append(val_txt_lst_se[idxs])
                # print(formcontent)
                # imgcontent.write(val_txt_lst_se[idxs])
        for idxs,txtbox in enumerate(formtxtbox):#文字塊
            everyformbox_txtbox=[]#每個文字塊含的方框
            box_num=0
            box_num_idx=0
            aaa=0
            txtbox_num=[]
            for idxs1,formboxes in reversed(list(enumerate(formtxtboxes))):#方框
                #NO重疊
                if min(txtbox[2]*w-10*0.41,formboxes[2])<max(txtbox[0]*w+10*0.41,formboxes[0]) or min(txtbox[1]*h-10*0.41,formboxes[3])<max(txtbox[3]*h+10*0.41,formboxes[1]):
                    continue
                if box_num!=0:
                    ifspace=re.search(r'(\S+\s+\S+)',formcontent[idxs])
                    if ifspace:
                        if min(txtbox[2]*w-25*0.41,formboxes[2])<max(txtbox[0]*w+15*0.41,formboxes[0]) or min(txtbox[1]*h-10*0.41,formboxes[3])<max(txtbox[3]*h+10*0.41,formboxes[1]):
                            continue
                    else:
                        continue
                #有重疊
                #找出文字塊中所有的方框，並紀錄
                everyformbox_txtbox.append(formboxes)
                txtbox_num.append(idxs1)
                box_num_idx=idxs1
                box_num+=1
            #如果沒有重疊，剛好一對一
            if (box_num==1):
                formcontent[idxs]=re.sub(r'\n','',formcontent[idxs])
                formtxtboxes_txt[box_num_idx]=formtxtboxes_txt[box_num_idx]+formcontent[idxs]
            else:
                if (box_num==0):
                    continue
                wordofform=re.sub(r'\n','',formcontent[idxs])
                formtxt_match=re.findall(r'(\s+)',wordofform)
                formtxt_match_num=len(formtxt_match)

                for idxss, formbox_txtbox in enumerate(txtbox_num):
                    #print(formbox_txtbox)
                    if (formtxt_match_num>0):
                        se_formtxt_match=re.search(r'(\s+)',wordofform)
                        formtxtboxes_txt[formbox_txtbox]=(wordofform[:se_formtxt_match.start(0)])
                        wordofform=wordofform[(se_formtxt_match.end(0)):]
                        formtxt_match_num-=1
                    else:
                        formtxtboxes_txt[formbox_txtbox]=wordofform
        # print('txt:',len(formtxtboxes_txt),formtxtboxes_txt)
        # print('box:',len(formtxtboxes),formtxtboxes)                
        with open("content/{}/{}.csv".format(name,imgfmname), 'w',encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile)
            formboxposition=[]
            formtxtposition=[]
            formtxtboxes2=formtxtboxes[:]
            formtxtboxes_txt2=formtxtboxes_txt[:]
            nowformleft=box[2]
            nowformright=box[0]
            #找單位
            formhigh=box[3]
            # 該表格最多方個的列
            formlist=[]
            formlist_num=0
            for idxs, formboxes in enumerate(formtxtboxes):
                thislist_num=0
                thislist=[]
                for idxs1, formboxes1 in enumerate(formtxtboxes):
                    if formboxes[1]==formboxes1[1]:
                        thislist_num+=1
                        thislist.append(formboxes1)
                if thislist_num>formlist_num:
                    formlist_num=thislist_num
                    formlist=thislist[:]
            for idxs, formboxes in enumerate(formtxtboxes):
                if formboxes[0]<nowformleft:
                    nowformleft=formboxes[0]
                if formboxes[2]>nowformright:
                    nowformright=formboxes[2]
                if formboxes[1]>formhigh:
                    formhigh=formboxes[1]
            error_break=0
            while(1):
                nowformhigh=box[3]
                if len(formtxtboxes2)==0:
                    break

                for idxs, formboxes in enumerate(formtxtboxes2):
                    if formboxes[1]<nowformhigh:
                        nowformhigh=formboxes[1]

                thisformbox=[]
                thisformtxt=[]
                thisformbox_num=[]
                for idxs, formboxes in reversed(list(enumerate(formtxtboxes2))):
                    if (abs(formboxes[1]-nowformhigh))<7*0.41:
                        thisformbox.append(formboxes)
                        thisformtxt.append(formtxtboxes_txt2[idxs])
                        thisformbox_num.append(idxs)
                # print(thisformtxt)
                # print(thisformbox)
                
                for idx, formbox in enumerate(thisformbox):
                    formbox.append(idx)
                # print(thisformbox)
                thisformbox=sorted(thisformbox, key=lambda x:x[0])
                # print(thisformbox)
                thisformtxt_sort=[]
                for idx, formbox in enumerate(thisformbox):
                    thisformtxt_sort.append(thisformtxt[formbox[4]])

                thisformtxt=thisformtxt_sort[:]
                # print(thisformtxt)

                if len(thisformbox)>1:
                    ifchange=1
                    while(1):
                        if ifchange==0:
                            break
                        ifchange=0
                        for idxs, formboxes in enumerate(thisformbox):
                            
                            if len(thisformbox)==(idxs+1):
                                if abs(formboxes[2]-nowformright)>20*0.41:
                                    for idxs1, formboxes1 in enumerate(formtxtboxes):
                                        if abs(formboxes[2]-formboxes1[0])<20*0.41:
                                            if ((formboxes[1]+formboxes[3])/2)>formboxes1[1] and ((formboxes[1]+formboxes[3])/2)<formboxes1[3]:
                                                thisformbox.append(formboxes1)
                                                thisformtxt.append(formtxtboxes_txt[idxs1])
                                                ifchange=1
                                continue
                            if idxs==0:
                                if abs(formboxes[0]-nowformleft)>20*0.41:
                                    for idxs1, formboxes1 in enumerate(formtxtboxes):
                                        if abs(formboxes[0]-formboxes1[2])<20*0.41:
                                            if ((formboxes[1]+formboxes[3])/2)>formboxes1[1] and ((formboxes[1]+formboxes[3])/2)<formboxes1[3]:
                                                thisformbox.insert(idxs,formboxes1)
                                                thisformtxt.insert(idxs,formtxtboxes_txt[idxs1])
                                                ifchange=1
                                continue
                            if abs(formboxes[2]-thisformbox[idxs+1][0])>20*0.41:
                                for idxs1, formboxes1 in enumerate(formtxtboxes):
                                    if abs(formboxes[2]-formboxes1[0])<20*0.41:
                                        if ((formboxes[1]+formboxes[3])/2)>formboxes1[1] and ((formboxes[1]+formboxes[3])/2)<formboxes1[3]:
                                            thisformbox.insert(idxs+1,formboxes1)
                                            thisformtxt.insert(idxs+1,formtxtboxes_txt[idxs1])
                                            ifchange=1
                                error_break+=1
                            if error_break >100000:
                                print('aaaaaaaaaaaaaaaaaaaaaaa')
                                break
                        if error_break >100000:
                            print('bbbbbbbbbbbbbbbbbbbbb')
                            break
                    if error_break >100000:
                        print('cccccccccccccccccccc')
                        thisformbox=[]
                        thisformtxt=[]
                        break
                    # print(thisformtxt)
                    # print(thisformbox)
                    # print('-----------------')

                    thisformbox2=thisformbox[:]
                    thisformtxt2=thisformtxt[:]
                    addnum=0
                    for idxs, formboxes in enumerate(thisformbox2):
                        thisaddnum=0
                        for idxs1, formboxes1 in enumerate(formlist):
                            if abs(formboxes[0]-formboxes1[0])<5*0.41 and abs(formboxes[2]-formboxes1[2])<5*0.41:
                                continue

                            if formboxes[0]<((formboxes1[0]+formboxes1[2])/2) and formboxes[2]>((formboxes1[0]+formboxes1[2])/2):
                                if thisaddnum>0:
                                    thisformbox.insert(idxs+addnum,formboxes)
                                    thisformtxt.insert(idxs+addnum,thisformtxt2[idxs])
                                    addnum+=1
                                thisaddnum+=1
                # print(thisformtxt)
                # print('-----------------')               
                writer.writerow(thisformtxt)
                this_form_content.append(thisformtxt)
                if len(thisformbox_num)!=0:
                    for i in sorted(thisformbox_num, reverse=True):
                        del formtxtboxes2[i]
                        del formtxtboxes_txt2[i]
            #找單位
            for idxs, txtbox in enumerate(txt_lst_se):
                if abs(((txtbox[1]+txtbox[3])*h/2)-formhigh)<(h*0.2):
                    titlesearch4=re.search(r'(^单位：)',val_txt_lst_se[idxs])
                    titlesearch2=re.search(r'(币种：)',val_txt_lst_se[idxs])
                    if titlesearch4:
                        writeform=re.sub(r'\n','',val_txt_lst_se[idxs])
                        writeformcsv=[]
                        writeformcsv.append(writeform)
                        writer.writerow(writeformcsv)
                        this_form_content.append(writeformcsv)
                        continue
                    if titlesearch2:
                        writeform=re.sub(r'\n','',val_txt_lst_se[idxs])
                        writeformcsv=[]
                        writeformcsv.append(writeform)
                        # print(writeform)
                        writer.writerow(writeformcsv)
                        this_form_content.append(writeformcsv)
            csvfile.close()
        # print('this_form_content',this_form_content)
        all_form_content.append(this_form_content)
        # print('all_form_content',all_form_content)
    else:
        formtxtbox=[]
        formcontent=[]
        Lsentence=0
        no_text=0
        for idxs,txtbox in enumerate (txt_lst_se):
            if (box[0]<=txtbox[0]*w and box[1]<=txtbox[3]*h) and (box[2]>=txtbox[2]*w and box[3]>=txtbox[1]*h):
                formtxtbox.append(txtbox)
                formcontent.append(val_txt_lst_se[idxs])
        # print(formcontent)
        if len(formtxtbox)>5:
            no_text=1
        for txtbox in formtxtbox:
            if (txtbox[2]-txtbox[0])*w>0.8*(box[2]-box[0]):
                Lsentence+=1
        this_form_content=''
        if Lsentence>0 and no_text==0:
            for i in formcontent:
                i=i.replace("\n","").replace(" ","")
                this_form_content=this_form_content+i
            all_form_content.append(this_form_content)
            # print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            # print('all_form_content',all_form_content)
        else:
            all_form_content.append(this_form_content)
            # print('ccccccccccccccccccccccccccccccccc')
            # print('all_form_content',all_form_content)
            # print(form_name)
            # print(len(formtxtboxes))
            form_lst_name=form_name.replace('nonline','FORM')
            form_lst_name=form_lst_name.replace('.jpg','_v1.jpg')
            nonlineformimg=img[box[1]:box[3],box[0]:box[2]].copy()
            cv2.imwrite(form_lst_name, nonlineformimg)
            form_w=box[2]-box[0]
            form_h=box[3]-box[1]
            for j in range(0,form_w,1):
                for i in range(0,form_h,1):
                    nonlineformimg[i,j]=255
            for idxs, txtbox in enumerate(txt_lst_se):
                if (txtbox[0]*w)>box[0] and (txtbox[2]*w)<box[2] and (txtbox[1]*h)>box[1] and (txtbox[3]*h)<box[3]:
                    titlesearch=re.search(r'(图+表*\s*\d+)',val_txt_lst_se[idxs])
                    titlesearch2=re.search(r'(表\s*\d+)',val_txt_lst_se[idxs])
                    titlesearch3=re.search(r'(来源)',val_txt_lst_se[idxs])
                    if titlesearch:
                        continue
                    if titlesearch2:
                        continue
                    if titlesearch3:
                        continue
                    cv2.rectangle(nonlineformimg, (round(txtbox[0]*w)-box[0], round(txtbox[1]*h)-box[1]), (round(txtbox[2]*w)-box[0], round(txtbox[3]*h)-box[1]), (0,0,0), -2, 8)
            form_lst_name=form_lst_name.replace('_v1.jpg','_v2.jpg')
            cv2.imwrite(form_lst_name, nonlineformimg)
    # print('dddddddddddddddddddddddddddddddddd')
    # print('all_form_content',all_form_content)
    return all_form_content

#------------------------------------------------
def get_back_txt(idxx,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,w,h):
    new_filt_box.append([int(txt_lst_se[idxx][0]*w),int(txt_lst_se[idxx][3]*h),int(txt_lst_se[idxx][2]*w),int(txt_lst_se[idxx][1]*h)])
    new_filt_txts.append(val_txt_lst_se[idxx])
    return new_filt_box,new_filt_txts

#-------------------------------------------------
# def find_note(note_box,note_txt,new_filt_box,new_filt_txts,img1):
#     index=[]
#     for idx, txt in enumerate(new_filt_txts):
#         if re.search(r'(注+\s*\d*：+)',txt):
#             index.append(idx)
#             note_box.append(new_filt_box[idx])
#             note_txt.append(txt.rstrip())
#     index.reverse()
#     for i in index:
#         del new_filt_box[i]
#         del new_filt_txts[i]
     
#     for idx,notetxt in enumerate(note_txt):
#         cv2.rectangle(img1, (note_box[idx][0], note_box[idx][1]), (note_box[idx][2],note_box[idx][3]), (0,255,0), 3, 8)
#         print('-------------------------------')
#         print('note:',notetxt)
#         flag=False
#         compare=re.search(r'(。+\s*$)',notetxt)

#         while not compare :
#             print('pre note box',note_box[idx])
#             print('pre compare:',compare)
#             print('*****in*******')

#             if flag:    
#                 print('aaaaaaaaaaaaaaaaaaaaaa')  
#                 break
#             for idxs, box in enumerate(new_filt_box):
#                 if 0<box[3]-note_box[idx][3]<=50:
#                     print(new_filt_txts[idxs])
#                     cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,255,0), 3, 8)
#                     notetxt=notetxt+new_filt_txts[idxs].rstrip()
#                     note_box[idx]=box
#                     del new_filt_txts[idxs]
#                     del new_filt_box[idxs]
#                     compare=re.search(r'(。+\s*$)',notetxt)
#                     print('post note:',notetxt)
#                     print('post note box',note_box[idx])
#                     print('post compare:',compare)
#                     break
#                 if idxs==len(new_filt_box)-1:
#                     flag=True 
#             print('flag',flag)    
            

#     return new_filt_box,new_filt_txts,img1

#-------------------------------------------------
def find_notitle_imgform_caption(val_txt_lst,txt_lst,title,boxes,h,w):
    for idx,titletxt in enumerate (title):
        if titletxt == '':
            match_caption_num=0
            match_cpation_box=[]
            match_caption_size=h
            for idxs,txtbox in enumerate (txt_lst):
                titlematch=re.search(r'(^\s*图+表*\s*\d+)',val_txt_lst[idxs])
                titlematch1=re.search(r'(^\s*表\s*\d+)',val_txt_lst[idxs])
                if titlematch:
                    if ((boxes[idx][2])>(txtbox[0]*w)) and ((boxes[idx][0])<(txtbox[2]*w)):
                        new_size = abs((txtbox[1]+txtbox[3])*h/2 - boxes[idx][1])
                        if new_size < (h*0.025):
                            if (new_size < match_caption_size):
                                match_caption_size=new_size
                                match_caption_num=idxs
                                match_cpation_box=txt_lst[idxs]
                if titlematch1:
                    if ((boxes[idx][2])>(txtbox[0]*w)) and ((boxes[idx][0])<(txtbox[2]*w)):
                        new_size = abs((txtbox[1]+txtbox[3])*h/2 - boxes[idx][1])
                        if new_size < (h*0.025):
                            if (new_size < match_caption_size):
                                match_caption_size=new_size
                                match_caption_num=idxs
                                match_cpation_box=txt_lst[idxs]
            if (match_cpation_box!=[]):
                title[idx]=val_txt_lst[match_caption_num]
    return title

#-------------------------------------------------
def sperate_title_caption(val_txt_lst,txt_lst,title,boxes,h,w):
    for idx,titletxt in enumerate (title):
        if titletxt == '':
            match_caption_num=0
            match_cpation_box=[]
            match_caption_size=h
            match_caption_idx=0
            for idxs,txtbox in enumerate (txt_lst):
                titlematch=re.search(r'(图+表*\s*\d+)',val_txt_lst[idxs])
                if titlematch:
                    if ((boxes[idx][2])>(txtbox[0]*w)) and ((boxes[idx][0])<(txtbox[2]*w)):
                        new_size = abs(txtbox[1]*h - boxes[idx][1])
                        if new_size < (h*0.1):
                            if (new_size < match_caption_size):
                                match_caption_size=new_size
                                match_caption_num=idxs
                                match_cpation_box=txt_lst[idxs]
                                 
                    continue
                titlematch1=re.search(r'(表\s*\d+)',val_txt_lst[idxs])
                if titlematch1:
                    if ((boxes[idx][2])>(txtbox[0]*w)) and ((boxes[idx][0])<(txtbox[2]*w)):
                        new_size = abs(txtbox[1]*h - boxes[idx][1])
                        if new_size < (h*0.1):
                            if (new_size < match_caption_size):
                                match_caption_size=new_size
                                match_caption_num=idxs
                                match_cpation_box=txt_lst[idxs]
                                match_caption_idx=titlematch1.start(0)
            if (match_cpation_box!=[]):
                title[idx]=val_txt_lst[match_caption_num][match_caption_idx:]
    return title

#-------------------------------------------------
def find_notitle_in_different_page(val_txt_lst,txt_lst,page_num):
    matchtitle=[]
    matchtitlebox=[]
    finaltitletxt=[]
    finalmatchidx=0
    for idxs,txtbox in enumerate (txt_lst):
        titlematch=re.search(r'(图+表*\s*\d+)',val_txt_lst[idxs])
        if titlematch:
            matchtitlebox.append(txt_lst[idxs])
            finalmatchidx=titlematch.start(0)
            finaltitletxt=val_txt_lst[idxs][finalmatchidx:]
            matchtitle.append(finaltitletxt)
    for idxs,txtbox in enumerate (txt_lst):
        titlematch1=re.search(r'(表\s*\d+)',val_txt_lst[idxs])
        if titlematch1:
            matchtitlebox.append(txt_lst[idxs])
            finalmatchidx=titlematch1.start(0)
            finaltitletxt=val_txt_lst[idxs][finalmatchidx:]
            matchtitle.append(val_txt_lst[idxs])
    title_page_num=[]
    for idx,txtbox in enumerate (matchtitle):
        title_page_num.append(page_num+1)
    return title_page_num,matchtitle,matchtitlebox

#-------------------------------------------------
def del_side_of_firstpage(new_filt_box,new_filt_txts,img):
    col_ans,LeftorRight,line_postion=dist_2col(img)
    if col_ans==True:
        if LeftorRight==True:
            cnt=0
            for idx,txt_box in enumerate(new_filt_box):
                if txt_box[2]<line_postion:
                    del new_filt_box[idx-cnt]
                    del new_filt_txts[idx-cnt]
                    cnt+=1
        else:
            cnt=0
            for idx,txt_box in enumerate(new_filt_box):
                if txt_box[0]>line_postion:
                    del new_filt_box[idx-cnt]
                    del new_filt_txts[idx-cnt]
                    cnt+=1
    return new_filt_box,new_filt_txts

#-------------------------------------------------
def correct_txt_position(val_txt_lst,txt_lst):
    position=txt_lst[:]
    new_position=[]
    # print(position)
    for idx, pos in enumerate(position):
        pos.append(val_txt_lst[idx])
        # print(pos[1])
    new_position=sorted(position, key = lambda s: s[1])

    new_val_txt_lst=[]
    for idx, pos in enumerate(new_position):
        # print(pos[1])
        new_val_txt_lst.append(pos[4])
        del pos[4]
    # print(new_val_txt_lst)
    # print(new_position)
    return new_val_txt_lst,new_position

#-------------------------------------------------
def double_page_form(boxes,flags,new_filt_box,h):#表格跨頁
    remain_form_head=False
    remain_form_tail=False
    form_boxes=[]
    for idx, flag in enumerate(flags):
        if flag==3 or 4:
            form_boxes.append(boxes[idx])
    form_boxes=sorted(form_boxes,key=lambda x:x[1])
    # print(new_filt_box)
    if len(form_boxes)>0:
        if form_boxes[0][1]<h*1/6:
            if len(new_filt_box)==0 or new_filt_box[0][1]>form_boxes[0][3]:
                remain_form_head=True
        if form_boxes[-1][3]>h*5/6:
            if len(new_filt_box)==0 or new_filt_box[-1][3]<form_boxes[-1][1]:
                remain_form_tail=True
    # print(remain_form_head,remain_form_tail)
    return remain_form_head,remain_form_tail

#-------------------------------------------------
def multi_draw(page_num,form_lst,f_img_lst,txt_lst,img_lst,val_txt_lst,origin_img_dir,name,all_dir,show_all,show_caption,thresh_h,thresh_t,txt_lst_se,val_txt_lst_se,head_txt_box,tail_txt_box,head_txt_page,tail_txt_page):# 對所有物件 圖表內文做分析及篩選
 #             頁數    ,表格列   ,　圖片列　,文字座標,圖片列　,文字列     ,文件存放位置　,文件名稱,子位置,                    ,文件上下標位置閥值 ,表格內文字座標, 表格內文字 ,文件上標位置 ,下標位置     ,上標的頁數   ,下標頁數

    # print('val_txt_lst:',val_txt_lst,len(val_txt_lst))
    # print('---------------------')  
    # print('txt_lst:',txt_lst[[3]],len(txt_lst))
    
    head_txt=''
    tail_txt=''
    for idx, i in enumerate(head_txt_page):
        if i==page_num:
            head_txt=head_txt+head_txt_box[idx]
    for idx, i in enumerate(tail_txt_page):
        if i==page_num:
            tail_txt=tail_txt+tail_txt_box[idx]
    route="{}/{}/{}".format(origin_img_dir, name,name)
    tmp = "%03d" % (page_num+1)
    print("{}_{}.jpg".format(route,tmp))
    img = cv2.imread("{}_{}.jpg".format(route,tmp))
    h,w,_=img.shape
    img1=np.copy(img)
    #------------------------------------------------------
    #clean 目錄
    if (page_num != 0):
        xtf=clean_content(page_num,txt_lst,val_txt_lst,all_dir,name,tmp,img1)
        if(xtf):
            return [],[],[],[],[],[],[],False,False,False,False,h,w
    #print(re.search(xxxxx,"....."))
    # if re.search(".....",xxxxx):
    #     print(page_num)
    #print(xxxxx)
    # if (page_num != 5):
    #     cv2.imwrite("{}/{}/{}_{}.jpg".format(all_dir[3],name,name,tmp), img1)
    #     with open("{}/{}/{}_{}.txt".format(all_dir[1],name,name,tmp), "w") as text_file:
    #         text_file.write('')
    #     with open("clean_TXT/{}/{}_{}.txt".format(name,name,tmp), "w") as text_file:
    #         text_file.write('')
    #     with open("output_json/{}/{}.json".format(name,page_num+1), "w") as list_obj:
    #             list_obj.write('')
    #     return [],[],[],[],[],[],[],h,w

    #-----------------------------------------------------------------
    #框出表格中內文
    for idxs,txt_box in enumerate(txt_lst_se):
        cv2.rectangle(img1, (round(txt_box[0]*w), round(txt_box[3]*h)), (round(txt_box[2]*w), round(txt_box[1]*h)), (255,0,255), 3, 8)

    #----------------------------------------------------------------
    #do deeplearning找到圖片、表格位置
    # imgbox,formbox=E.__EvaluateOne__(page_num,img)

    #------------------------------------------------------------------

    # mask=np.zeros((h,w), np.uint8)
    img3=np.copy(img)
    mask,mask_big,big_lst_name,boxes,flags \
    =merge_horizontal_form_image(h,w,img_lst,f_img_lst,form_lst,all_dir[0],all_dir[2],name,tmp,thresh_h,img3)


    # mask:圖片在頁面上的位置圖

    # if page_num==0:
    #     del_boxes=[]
    #     print('------------------')
    #     print(len(mask),len(mask_big),len(big_lst_name),len(boxes),len(flags))
    #     print('------------------')
    #     for idx, box in enumerate(boxes):
    #         if (box[0]<0) or (box[1]<0) or (box[2]>w) or (box[3]>h):
    #             del_boxes.append(idx)
    #     # print(del_boxes)
    #     if len(del_boxes)>0:
    #         for i in sorted(del_boxes, reverse=True):
    #             del flags[i]
    #             del boxes[i]
    #             del big_lst_name[i]
    #     print(len(boxes))

    mask_with_txt=np.zeros((h,w), np.uint8) # get filter txt and form image for non line 
    title=['' for _ in big_lst_name]#紀錄 每張圖片對應的 標題
    title_box=[None for _ in big_lst_name]#紀錄每張圖片的 標題位置
    tail=['' for _ in big_lst_name]#紀錄每張圖片的 尾端文字
    tail_box=[None for _ in big_lst_name]#紀錄每張圖片的 尾端文字方塊位置
    every_page_min_height=h#紀錄文字塊區域的最高處
    all_txt_after_filt=[]#紀錄濾除過後之文字位置
    all_txt_num=[]#紀錄濾除過後之文字

    #------------------------------這部份利用merge_horizontal_form_image 產生出來對於圖表的mask 進行caption的偵測以及濾除
    
    img1,mask_with_txt,title,title_box,tail,tail_box,all_txt_after_filt,all_txt_num \
    =horizonmerge_find_caption(txt_lst,val_txt_lst,img1,mask_with_txt,title,mask,boxes,flags,title_box,tail,tail_box,all_txt_after_filt,all_txt_num,every_page_min_height,w,h,img3)
    # print('note txt:',note_txt)
    # print('note box:',note_box)
    # print('----------------------')




    #--------------------------------------------------
    #del error img
    # del_boxes=[]
    # for idx, box in enumerate(boxes):
    #     if (box[0]<0) or (box[1]<0) or (box[2]>w) or (box[3]>h):
    #         del_boxes.append(idx)
    # # print(del_boxes)
    # if len(del_boxes)>0:
    #     for i in sorted(del_boxes, reverse=True):
    #         del flags[i]
    #         del boxes[i]
    #         del big_lst_name[i]
    #         del title[i]
    #         del title_box[i]

    #-----------------------------利用剛剛找初步的caption頭尾 來修補無法正確偵測的圖表的範圍
    boxes,all_txt_after_filt,all_txt_num \
    =expand_imgform_by_caption(boxes,title_box,tail_box,all_txt_after_filt,all_txt_num)
    # print(all_txt_num)
    # print('----------------------')

    #------------------------nonlinestart
    #------------------------這邊利用先前定義每頁的文字邊界進行區間定義 減少偵測失誤
    every_page_min_height-=5
    # print(int(thresh_t*h))
    # print(int(thresh_h*h))
    cv2.line(img1,(0,int(thresh_t*h)),(w,int(thresh_t*h)),(0,165,255),5)
    cv2.line(img1,(0,int(thresh_h*h)),(w,int(thresh_h*h)),(128,128,128),5)
    # cv2.line(img1,(0,int(every_page_min_height)),(w,int(every_page_min_height)),(203,192,255),5)

    #---------------------------- 利用現有偵測結果之物件匡再做一個mask 目的是要給無格線表格用(覆蓋掉先前的mask)
    mask=np.zeros((h,w), np.uint8)
    for idx,box in enumerate(boxes):
        if box[3]-box[1]>0.1*h:
            cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), idx+1, cv2.FILLED, 8)
   
    #-----------------------------
    # non_line_boxes=[]
    # # non_line_boxes=mg_pdf_non(bbbbz,non_line_boxes0)
    # merge_non_line_boxes=[]
    # # for nlf_box in non_line_boxes:
    # #    if (nlf_box[3]-nlf_box[1])>0.01*h and checks(mask,nlf_box,0.15):
    # #        notmerge=True
    # #        for box in merge_non_line_boxes:
    # #            boxmg=non_line_ovlap(nlf_box,box)
    # #            if boxmg:
    # #                 notmerge=False
    # #                 merge_non_line_boxes.remove(box)
    # #                 merge_non_line_boxes.append(boxmg)
    # #                 break
    # #         if notmerge:
    # #             merge_non_line_boxes.append(nlf_box)

    # #---------------------------- 由於新增了無格線表格至結果 因此也要真對於格線表格的caption進行律除
    # mask_tmp_for_nonline=np.zeros((h,w),np.uint8)
    # new_filt_txts=[]# 律除無格線表格後的文字
    # new_filt_box=[]# 律除無格線表格後的文字的位置
    # non_title=['' for _ in merge_non_line_boxes]# 根據有多少無格線表格 設立多少caption title的空間
    # non_title_box=[None for _ in merge_non_line_boxes]# 根據有多少無格線表格 設立多少caption title位置的空間
    # non_tail_box=[None for _ in merge_non_line_boxes]# 根據有多少無格線表格 設立多少caption tail位置的空間
    # non_big_name=[]#紀錄各個物件的命名


    # for idx,box in enumerate(merge_non_line_boxes):#把這些物件畫到遮罩上 並紀錄檔名
    #     cv2.rectangle(mask_tmp_for_nonline,(box[0],box[1]),(box[2],box[3]),int(idx+1),cv2.FILLED,8)
    #     non_big_name.append("{}/{}/{}_{}_{}.jpg".format(all_dir[6],name,name,tmp,idx))
    # for idx,box in enumerate(all_txt_after_filt):#針對無格線表格再進行一次找caption
    #     isov,_=checks1(mask_tmp_for_nonline,(box[0],box[1],box[2],box[3]),0.4)
    #     # if isov==False:
    #     #     new_filt_txts.append(all_txt_num[idx])
    #     #     new_filt_box.append(all_txt_after_filt[idx])
    #     if isov:
    #         flag,POS,NEG=chk_all_v1(mask_tmp_for_nonline,box[1],box[3],box[0],box[2])
    #         xxx=''.join(all_txt_num[idx].split())
    #         if flag and len(xxx):
    #             if POS:
    #                 numb=POS[0]
    #                 TAGS=POS[1]
    #                 if (xxx[0]=='图') or (xxx[0]=='表' and xxx[-1]!='。') or (xxx[-1]=='表'):
    #                     cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0,255,0), 4, 8)
    #                     non_title[TAGS]=xxx
    #                     non_title_box[TAGS]=box
    #                     continue
    #             if NEG:
    #                 numb=NEG[0]
    #                 TAGS=NEG[1]
    #                 if '来源' in xxx[:5]:
    #                     non_tail_box[TAGS]=box
    #                     cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0,255,0), 4, 8)
    #                     continue
    #         new_filt_txts.append(all_txt_num[idx])
    #         new_filt_box.append(all_txt_after_filt[idx])
    #     else:
    #         new_filt_txts.append(all_txt_num[idx])
    #         new_filt_box.append(all_txt_after_filt[idx])
    # title.extend(non_title)
    # title_box.extend(non_title_box)
    # tail_box.extend(non_tail_box)
    # big_lst_name.extend(non_big_name)
    # boxes.extend(merge_non_line_boxes)
    # flags.extend([3 for _ in merge_non_line_boxes])


    new_filt_box=all_txt_after_filt[:]
    new_filt_txts=all_txt_num[:]

    #------------------------------------
    #advance fix caption for tail
    img1,tail_box=advance_fix_tail_caption(img1,tail_box,new_filt_box)


    #-------------------------------------
    #filt txt with tail caption
    new_filt_box,new_filt_txts=filt_tail_caption_in_txt(tail_box,new_filt_box,new_filt_txts)
    # print(new_filt_txts)
    # print('----------------------')
    #-----------------------------------
    # bbbbz=use_miner_txt_nonline(img,page_num,new_filt_box)#利用pdfminer的文字塊區域做水平匹配 找出可能可以合併的的候選水平方框 目的是修補一些無格線表格的偵測
    # # for boxa in bbbbz:
    # #     cv2.rectangle(img1, (boxa[0], boxa[1]), (boxa[2], boxa[3]), (255,0,255), 3, 8)
    # boxes=mg_pdf_non(bbbbz,boxes,flags)# 合併無格線表格與候選區域
    new_filt_txts_2=new_filt_txts.copy()
    new_filt_box_2=new_filt_box.copy()
    # for box in new_filt_box:
    #    cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (210,200,128), 3, 8) 
    #boxes,new_filt_box_2,new_filt_txts_2=clean_all_txt_in_item(page_num,boxes,new_filt_box_2,new_filt_txts_2)#清除部份區域合併造成文字並未被刪除的狀況

    #----------------------------- 至此可能會有物件重疊之問題 因此需要合併物件
    boxes,flags,big_lst_name,title=merge_imgform_overlapped(boxes,flags,big_lst_name,title)

    #------------------------------
    # distinguish form and 2column
    if (page_num==0):
        boxes,flags,title,title_box,big_lst_name \
        =distingish_form_2column(new_filt_box,boxes,flags,title,title_box,big_lst_name)

    #-------------------------------
    #刪除文字中存在的caption
    new_filt_box,new_filt_txts=del_caption_in_txt(boxes,flags,new_filt_box,new_filt_txts)
    # print(new_filt_txts)
    # print('----------------------')
    #------------------------------
    #分辨 image and 2column
    if (page_num==0):
        boxes,flags,title,title_box,big_lst_name \
        =distingish_image_2column(new_filt_box,boxes,flags,title,title_box,big_lst_name)

        #-------------------------分辨 nonlineform and 2column
        boxes,flags,title,title_box,big_lst_name,new_filt_box,new_filt_txts \
        =distingish_nonlineform_2column(boxes,flags,title,title_box,big_lst_name,new_filt_box,new_filt_txts,img1)
 


    #boxes,new_filt_box,new_filt_txts=clean_all_txt_in_item(page_num,boxes,new_filt_box,new_filt_txts)

    #---------------------------------------------------------------------------------
    #對有上下caption的image進行擴大
    boxes=expand_image_by_caption(boxes,flags,title_box,tail_box)


    #boxes,new_filt_box,new_filt_txts=clean_all_txt_in_item(page_num,boxes,new_filt_box,new_filt_txts)
    #--------------------------
    #-------------------------------------
    #use deep learning result(imgbox & formbox) to analyze 
    # if(imgbox!=[] or imgbox!=None):
    #     long_boxes=len(boxes)
    #     for idx,ibox in enumerate(imgbox):
    #         notmatch=True
    #         for idxs,box in enumerate(boxes):
    #             if (flags[idxs]!=3):
    #                 if min(ibox[2],box[2])<max(ibox[0],box[0]) or min(ibox[3],box[3])<max(ibox[1],box[1]):
    #                     continue
    #                 else:
    #                     boxes[idxs]=[min(ibox[0],box[0]),min(ibox[1],box[1]),max(ibox[2],box[2]),max(ibox[3],box[3])]
    #                     notmatch=False
    #                     if(flags[idxs]==2):
    #                         flags[idxs]=0
    #                         big_lst_name[idxs]=big_lst_name[idxs].replace('FORM','IMAGE')
    #                     break

    #         if notmatch:
    #             boxes.append(imgbox[idx])
    #             flags.append(0)
    #             big_lst_name.append("{}/{}/{}_{}_{}.jpg".format(all_dir[2],name,name,tmp,long_boxes+1))
    #             title.append('')
    #             long_boxes+=1

    # if(imgbox!=[] or formbox!=None):
    #     long_boxes=len(boxes)
    #     for idx,fbox in enumerate(formbox):
    #         notmatch=True
    #         for idxs,box in enumerate(boxes):
    #             if min(fbox[2],box[2])<max(fbox[0],box[0]) or min(fbox[3],box[3])<max(fbox[1],box[1]):
    #                 continue
    #             else:
    #                 boxes[idxs]=[min(fbox[0],box[0]),min(fbox[1],box[1]),max(fbox[2],box[2]),max(fbox[3],box[3])]
    #                 notmatch=False
    #                 if(flags[idxs]!=2 or flags[idxs]!=3):
    #                     flags[idxs]=2
    #                     big_lst_name[idxs]=big_lst_name[idxs].replace('IMAGE','FORM')
    #                 break        
    #         if notmatch:
    #             boxes.append(formbox[idx])
    #             flags.append(2)
    #             big_lst_name.append("{}/{}/{}_{}_{}.jpg".format(all_dir[0],name,name,tmp,long_boxes+1))
    #             title.append('')
    #             long_boxes+=1
    # boxes,new_filt_box,new_filt_txts=clean_all_txt_in_item(page_num,boxes,new_filt_box,new_filt_txts)
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print(new_filt_txts)

    boxes,flags,big_lst_name,title=merge_imgform_overlapped(boxes,flags,big_lst_name,title)

    #-----------------------------------
    boxes,flags,big_lst_name,title=correct_wrongimg_to_form(boxes,flags,big_lst_name,title,txt_lst_se,val_txt_lst_se,w,h,img,all_dir,name,tmp)
   
    #-----------------------------------
    #delete first page wrong img and form(第一頁比較亂)
    if (page_num==0):
        boxes,flags,title,title_box,big_lst_name=del_firstpage_wrong_imgform(boxes,flags,title,title_box,big_lst_name,h)

    #-----------------------------------
    #delet wrong img and form(浮水印特例)
    boxes,flags,title,title_box,big_lst_name \
    =del_watermark_on_certain_page(page_num,10,boxes,flags,title,title_box,big_lst_name,h)

    #---------------------------------------------
    #分割雙表格
    # boxes_old=boxes[:]

    boxes,flags,title,big_lst_name,new_filt_box,new_filt_txts,all_form_content \
    =resize_output_form(boxes,flags,title,big_lst_name,img,w,h,name,tmp,all_dir,txt_lst_se,val_txt_lst_se,new_filt_box,new_filt_txts,img3)

    
    # if page_num==8:
    # print('----------------')
    # print('page:',page_num+1)
    # print(all_form_content)
    # print('---------------')

    # print(new_filt_txts)
    # print('----------------------')


    boxes,new_filt_box,new_filt_txts=clean_all_txt_in_item(page_num,boxes,new_filt_box,new_filt_txts,flags)
    # print('--------------------')
    # print(new_filt_txts)
    # print('--------------------')
    # print(new_filt_txts)

    new_filt_txts,new_filt_box=correct_txt_position(new_filt_txts,new_filt_box)

    #-----------------------------------
    # new_filt_box,new_filt_txts,img1=find_note(note_box,note_txt,new_filt_box,new_filt_txts,img1)


    #-----------------------------------
    #find none title caption
    title=find_notitle_imgform_caption(val_txt_lst,txt_lst,title,boxes,h,w)

    #--------------------------------------------
    # sperate title caption
    title=sperate_title_caption(val_txt_lst,txt_lst,title,boxes,h,w)

    #---------------------------------------------
    #different page caption find no caption
    title_page_num,matchtitle,matchtitlebox=find_notitle_in_different_page(val_txt_lst,txt_lst,page_num)
    # print('--------------------')
    # print(new_filt_txts)
    # print('--------------------')
    # print('new_filt_txts',new_filt_txts)
    # print('-------------------------------------')
    # print('new_filt_box',new_filt_box)

    #--------------------------------------
    remain_form_head,remain_form_tail=double_page_form(boxes,flags,new_filt_box,h)

    #--------------------------------------
    #進行頁首側邊欄濾除
    # if (page_num==0):
    #     new_filt_box,new_filt_txts=del_side_of_firstpage(new_filt_box,new_filt_txts,img)


    # print('--------------------')
    # print(new_filt_txts)
    # print('--------------------')

    #------------------------------
    #以下不用改成副程式
    #txt output 文字輸出順便律除一些輸出上的問題
    
    # print(new_filt_txts)
    # print(new_filt_box)
    # print('-----------------')
    wrong_txtcaption=[]
    wrong_txtcaption_num=0
    for idx,txt in enumerate (new_filt_txts):
        cleancaption=re.search(r'(^\s*图+表*\s*\d+)',txt)
        if cleancaption:
            cv2.rectangle(img1, (new_filt_box[idx][0], new_filt_box[idx][1]), (new_filt_box[idx][2],new_filt_box[idx][3]), (0,255,0), 3, 8)
            wrong_txtcaption.append(idx)
            wrong_txtcaption_num+=1
    if(wrong_txtcaption_num!=0):
        for i in sorted(wrong_txtcaption, reverse=True):
            del new_filt_txts[i]
            del new_filt_box[i]

    txt_flag=[]
    txt_box=[]

    
    with open("{}/{}/{}_{}.txt".format(all_dir[1],name,name,tmp), "w",encoding="utf-8") as text_file:
        # print('--------------------')
        # print(new_filt_txts)
        # print('--------------------')
        for idx,txt in enumerate(new_filt_txts):
            xxx=''.join(txt.split())
            if len(xxx)>0:
                if xxx[0]=='图' and (new_filt_box[idx][2]-new_filt_box[idx][0])<0.6*w and (new_filt_box[idx][3]-new_filt_box[idx][1])<0.1*h:
                    cv2.rectangle(img1, (new_filt_box[idx][0], new_filt_box[idx][1]), (new_filt_box[idx][2],new_filt_box[idx][3]), (0,255,0), 3, 8)
                    continue

                if '来源' in xxx[:5] and (new_filt_box[idx][2]-new_filt_box[idx][0])<0.6*w:
                    cv2.rectangle(img1, (new_filt_box[idx][0], new_filt_box[idx][1]), (new_filt_box[idx][2],new_filt_box[idx][3]), (0,255,0), 3, 8)
                    continue

                txt=re.sub(r'\[(\w*?)\]','',txt)
                txt=re.sub(r'[tT]able_[a-zA-Z]+','',txt)
                txt=re.sub(r'Header','',txt)
                txt=re.sub(r'．?','',txt)
                text_file.write(txt)


                if show_all:
                    txt_box.append(new_filt_box[idx])
                    txt_flag.append(4)
                    cv2.rectangle(img1,(new_filt_box[idx][0],new_filt_box[idx][1]),(new_filt_box[idx][2],new_filt_box[idx][3]),(0,255,255),3,8)
        text_file.close()

    #---------------------------------------------
    #txt output to no \n
    txts_box=str()
    all_word_txt=[]
    all_word_txt_box=[]
    now_word_txt=[]
    remain_txt_head=False
    remain_txt_tail=False
    wordleft=0
    # cv2.line(img1,(0,int(h*1/6)),(w,int(h*1/6)),(0,125,125),3)
    # cv2.line(img1,(int(w*2/3),0),(int(w*2/3),h),(0,125,125),3)
    # cv2.line(img1,(int(w*5/6),0),(int(w*5/6),h),(0,255,0),3)

    with open("clean_TXT/{}/{}_{}.txt".format(name,name,tmp), "w",encoding="utf-8") as text_file2:
        max_right_box=w
        min_left_box=w
        word_high=[]
        word_long=[]
        word_right_box=[]
        word_left_box=[]
        for idx,box in enumerate(new_filt_box):
            word_high.append(box[3]-box[1])
            word_long.append(box[2]-box[0])
            word_right_box.append(box[2])
            word_left_box.append(box[0])
            # print('left:',box[0])
        if len(word_left_box)>0:
            wordleft=min(word_left_box)

        wordlong_num=[]
        wordlongidx=[]
        if len(word_long)>0:
            for idx,wordlong in enumerate(word_long):
                wordmatch=False
                if idx==0:
                    wordlongidx.append(wordlong)
                    wordlong_num.append(1)
                    continue
                for idxs,wordlong_1 in enumerate(wordlongidx):
                    if abs(wordlongidx[idxs]-wordlong)<41:
                        a=wordlong_num[idxs]
                        a+=1
                        wordlong_num[idxs]=a
                        wordmatch=True
                        break
                if wordmatch==True:
                    continue
                else:
                    wordlongidx.append(wordlong)
                    wordlong_num.append(1)
            matchnum=np.max(wordlong_num)
            # matchidx=0
            finalwordlong=0
            matchtimes=0
            # matchtimesbox=[]
            for idx,num in enumerate(wordlong_num):
                if num==matchnum:
                    if matchtimes==0:
                        # matchidx=idx
                        finalwordlong=wordlongidx[idx]
                        matchtimes+=1
                        # matchtimesbox.append(wordlongidx[idx])
                    else:
                        if (finalwordlong<wordlongidx[idx]):
                            finalwordlong=wordlongidx[idx]
                            # matchidx=idx
            sen_long=finalwordlong

        wordright_num=[]
        wordrightidx=[]

        if len(word_right_box)>0:
            if max(word_right_box)<w*5/6:
                max_right_box=w*5/6
            else:
                word_right_box.sort()
                temp=[i for i in word_right_box if i>=w*5/6]
                word_right_box=temp[:]
                # print(word_right_box)
                # print(word_right_box)
                for idx,wordright in enumerate(word_right_box):
                    wordmatch=False
                    # print(wordrightidx)
                    # print(wordright_num)
                    if idx==0:
                        wordrightidx.append(wordright)
                        wordright_num.append(1)
                        continue
                    for idxs,wordlong_1 in enumerate(wordrightidx):
                        if abs(wordrightidx[idxs]-wordright)<30:
                            # a=wordright_num[idxs]
                            # a+=1
                            wordright_num[idxs]+=1
                            wordmatch=True
                            break
                    if wordmatch==True:
                        continue
                    else:
                        wordrightidx.append(wordright)
                        wordright_num.append(1)
                
                
                # matchnum=np.max(wordright_num)
                max_idx=wordright_num.index(max(wordright_num))
                max_right_box=wordrightidx[max_idx]
                # print(max_idx)
                
                # matchidx=0
                # finalwordright=0
                # for idx,num in enumerate(wordright_num):
                #     if num==matchnum:
                #         # matchidx=idx
                #         finalwordright=wordrightidx[idx]
                # max_right_box=finalwordright
                
                # print(max_right_box)
                # print('--------------')
        cv2.line(img1,(int(max_right_box),0),(int(max_right_box),h),(0,255,0),3)
#--------------------------
        # now_word_txt_box=[]
        top=[]
        bottom=[]
        left=[]
        right=[]
        tmpleft=wordleft
        # print(tmpleft)
        # print(wordleft)
        for idx,txt in enumerate(new_filt_txts):

            # print('--------------------')
            # print(all_word_txt_box)
            # print(now_word_txt)
            
            xxxx=''.join(txt.split())
            # print('--------------------')
            # print(xxxx)
            # print(new_filt_box[idx][2])
            # print('########',txts_box)
            
            # print('------------')
            # print(txt)
            # print(new_filt_box[idx])
            # print('------------')
            # now_word_txt_box=new_filt_box[idx][:]
            if len(xxxx)>0:
                # print('k')
                # if xxxx[0]=='图' and (new_filt_box[idx][2]-new_filt_box[idx][0])<0.6*w and (new_filt_box[idx][3]-new_filt_box[idx][1])<0.1*h:
                #     # print('j')
                #     continue
                # if '来源' in xxxx[:5] and (new_filt_box[idx][2]-new_filt_box[idx][0])<0.6*w:
                #     # print('j')
                #     continue
                top.append(new_filt_box[idx][1])
                bottom.append(new_filt_box[idx][3])
                left.append(new_filt_box[idx][0])
                right.append(new_filt_box[idx][2])

                if txts_box==str():
                    # print('f')
                    all_word_txt_box.append(new_filt_box[idx][1])
                 
                txt=re.sub(r'\[(\w*?)\]','',xxxx)
                txt=re.sub(r'[tT]able_[a-zA-Z]+','',xxxx)
                txt=re.sub(r'Header','',xxxx)
                txt=re.sub(r'．?','',xxxx)

                if idx==0:
                    if (new_filt_box[idx][1]<h*1/6) and ((new_filt_box[idx][0]-wordleft)<(30)):
                        answernn3=re.match(r'^[注○（(](\d|一|二|三|四|五|六|七|八|九|十)',xxxx)
                        answernn4=re.match(r'^(\d|一|二|三|四|五|六|七|八|九|十)+[、)）]',xxxx)
                        answernn5=re.match(r'^\d+[.]\D',xxxx)
                        if not (answernn3 or answernn4 or answernn5):
                            remain_txt_head=True
                            cv2.line(img1,(0,new_filt_box[idx][3]),(w,new_filt_box[idx][3]),(255,255,0),3)
                    # print(remain_txt_tail)

                answernn=re.search(r'。\s*$',xxxx)
                if answernn: #if sentence end is 'o' 
                    # print('g')
                    # print(xxxx)

                    if (idx==(len(new_filt_box)-1)): #if is last idx - then output
                        txts_box=txts_box+xxxx
                        # print(txts_box)
                        # print('111111111')
                        # print('txt')
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()  
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()   
                        continue

                    if abs(new_filt_box[idx][2]-max_right_box)>85: #if sentence end is far from max_right_box - then output
                        wordleft=tmpleft
                        txts_box=txts_box+xxxx
                        # print(txts_box)
                        # print('222222222')
                        # print('txt')
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()
                        continue
                    # print(xxxx)
                    # print(new_filt_box[idx])

                    answernn2=re.search(r'^\s{2,}',new_filt_txts[idx+1]) # next sentence start with 2 space

                    if ((new_filt_box[idx+1][0]-wordleft)>(30)) or (answernn2): #if the 2 setense start distane difference is long enough or next setence is new - then output    
                        wordleft=tmpleft
                        txts_box=txts_box+xxxx
                        # print(txts_box)
                        # print('3333333333')
                        # print('txt')
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()
                        continue
                        # print(new_filt_box[idx][0])
                        # print(new_filt_box[idx+1][0])

                    if ((new_filt_box[idx][3]-new_filt_box[idx][1])>73):#if this sentence heigh is too long( pdfminer exceptional condition) - then output
                        wordleft=tmpleft
                        txts_box=txts_box+xxxx
                        # print(txts_box)
                        # print('4444444444')
                        # print('txt')
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()
                        continue

                    if (new_filt_box[idx+1][1]-new_filt_box[idx][3])>(new_filt_box[idx][3]-new_filt_box[idx][1])*1.5:#if space between 2 lines too far - then output
                        wordleft=tmpleft
                        txts_box=txts_box+xxxx
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()
                        continue

                    answernn3=re.match(r'^[注○（(](\d|一|二|三|四|五|六|七|八|九|十)',new_filt_txts[idx+1])
                    answernn4=re.match(r'^(\d|一|二|三|四|五|六|七|八|九|十)+[、)）]',new_filt_txts[idx+1])
                    answernn5=re.match(r'^\d+[.]\D',new_filt_txts[idx+1])

                    if answernn3 or answernn4 or answernn5:
                        wordleft=tmpleft
                        # print('xxxxxxxxxxxxxxxxxxx')
                        txts_box=txts_box+xxxx
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()
                        continue

                    # print('aaaaaaaaaaaa')
                    txts_box=txts_box+xxxx
                    continue

                else: #sentence is not end with 'o'
                    # print('h')
                    if (new_filt_box[idx][3]-new_filt_box[idx][1])>85:#if sentence heigh is too long( pdfminer exceptional condition) - then output
                        wordleft=tmpleft
                        txts_box=txts_box+xxxx
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()   
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()                     
                        continue

                    if ((new_filt_box[idx][2]-new_filt_box[idx][0])>(w*0.6)) and (idx<(len(new_filt_box)-1)):#if sentence is long enough and not the last idx
                        if (new_filt_box[idx+1][1]-new_filt_box[idx][3])<(new_filt_box[idx][3]-new_filt_box[idx][1])*1.5:# if sentence and next are near enough
                            if abs(new_filt_box[idx][2]-max_right_box)<w*0.04:#if sentence is as long as max_right_box!!!!!!!!parameter issue?
                                if (re.match(r'^[注○（(](\d|一|二|三|四|五|六|七|八|九|十)',new_filt_txts[idx+1]))\
                                 or (re.match(r'^(\d|一|二|三|四|五|六|七|八|九|十)+[、)）]',new_filt_txts[idx+1]))\
                                 or (re.match(r'^\d+[.]\D',new_filt_txts[idx+1])):# or (xxxx[-1]==':' or '：'):
                                    # print('yyyyyyyyyyyyyyyyyyy')
                                    wordleft=tmpleft
                                    txts_box=txts_box+xxxx
                                    text_file2.write(txts_box)
                                    now_word_txt.append(txts_box)
                                    text_file2.write('\n')
                                    txts_box=str()  
                                    cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                                    top.clear()
                                    bottom.clear()
                                    left.clear()
                                    right.clear()                      
                                    continue

                                elif abs((new_filt_box[idx+1][0]-wordleft)<(30)) and (xxxx[-1]!= '：'):
                                    # print('bbbbbbbbbbbbb')
                                    # print(xxxx)
                                    txts_box=txts_box+xxxx
                                    # print('back')
                                    continue

                                elif (re.match(r'^[注○（(](\d|一|二|三|四|五|六|七|八|九|十)',xxxx))\
                                 or (re.match(r'^(\d|一|二|三|四|五|六|七|八|九|十)+[、)）]',xxxx))\
                                 or (re.match(r'^\d+[.]\D',xxxx)): 
                                    # print('zzzzzzzzzzzzzzzzzz')
                                    if (-10<(new_filt_box[idx+1][0]-new_filt_box[idx][0])):#>30):
                                        # print('cccccccccccccc')
                                        # print(xxxx)
                                        # print('-------------')
                                        wordleft=new_filt_box[idx+1][0]
                                        txts_box=txts_box+xxxx                    
                                        continue
                                    else:
                                        # print('zzzzzzzzzzzz')
                                        wordleft=tmpleft
                                        txts_box=txts_box+xxxx
                                        text_file2.write(txts_box)
                                        now_word_txt.append(txts_box)
                                        text_file2.write('\n')
                                        txts_box=str()  
                                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                                        top.clear()
                                        bottom.clear()
                                        left.clear()
                                        right.clear()                      
                                        continue

                                else:
                                    # print('555555555555')
                                    wordleft=tmpleft
                                    txts_box=txts_box+xxxx
                                    text_file2.write(txts_box)
                                    now_word_txt.append(txts_box)
                                    text_file2.write('\n')
                                    txts_box=str()  
                                    cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                                    top.clear()
                                    bottom.clear()
                                    left.clear()
                                    right.clear()                      
                                    continue

                            else:
                                wordleft=tmpleft
                                txts_box=txts_box+xxxx
                                # print(txts_box)
                                # print('666666666666')
                                # print('txt')
                                text_file2.write(txts_box)
                                now_word_txt.append(txts_box)
                                text_file2.write('\n')
                                txts_box=str() 
                                cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                                top.clear()
                                bottom.clear()
                                left.clear()
                                right.clear()                       
                                continue

                        else:#sentence and next are far away
                            wordleft=tmpleft
                            txts_box=txts_box+xxxx
                            # print(txts_box)
                            # print('77777777777')
                            # print('txt')
                            text_file2.write(txts_box)
                            now_word_txt.append(txts_box)
                            text_file2.write('\n')
                            txts_box=str()
                            cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                            top.clear()
                            bottom.clear()
                            left.clear()
                            right.clear()                        
                            continue

                    elif ((new_filt_box[idx][2]-new_filt_box[idx][0])>(w*0.6)) and (idx==(len(new_filt_box)-1))\
                     and (len(xxxx)>20) and (new_filt_box[idx][3]>h*5/6): # and not (re.search(r'人：',xxxx)):
                    #next page line break judgement
                    # elif (idx==(len(new_filt_box)-1)):
                        txts_box=txts_box+xxxx
                        # print(txts_box)
                        # print('888888888888')
                        # print(txt)
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()
                        remain_txt_tail=True 
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(0,0,255),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()                     
                        continue

                    else:#sentence is short or the last idx
                        wordleft=tmpleft
                        txts_box=txts_box+xxxx
                        # print('xxxx:',len(xxxx)) 
                        # print('99999999999')
                        # print('txt')
                        text_file2.write(txts_box)
                        now_word_txt.append(txts_box)
                        text_file2.write('\n')
                        txts_box=str()  
                        cv2.rectangle(img1,(min(left)-2,min(top)-2),(max(right)+2,max(bottom)+2),(255,0,0),2,8)                   
                        top.clear()
                        bottom.clear()
                        left.clear()
                        right.clear()                      
                        continue

                    print('??????????????????????')
                    print(xxxx)
                    print('??????????????????????')
        text_file2.close()
            

    # if len(now_word_txt)!=len(all_word_txt_box):
        # print(page_num)
        # print(len(now_word_txt))
        # print(len(all_word_txt_box))
        # print(all_word_txt_box)
        # print('abcdefghijklmnopqrstuvwxyz--------------------------------------------')
    
    #------------------------------------
    #save output in json
    # print('\n','\n')
    # print(now_word_txt)
    # print(all_word_txt_box)
    # print('---------------')
    with io.open("output_json/{}/{}.json".format(name,page_num+1), "w", encoding='utf-8') as list_obj:
        boxes_out=boxes[:]
        boxes_out.sort(key=takeSecond)
        new_idx=[]
        for box in boxes_out:
            new_idx.append(boxes.index(box)),head_txt,tail_txt
        
        if len(all_word_txt_box)==0 or len(now_word_txt)==0:
            for idx, box in enumerate(boxes_out):
                if flags[new_idx[idx]]==0 or flags[new_idx[idx]]==1:
                    Object={"type": "graph", "content": big_lst_name[new_idx[idx]], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                    list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')
                else:
                    img_add=0
                    if not all_form_content:
                        print('content error, page:',page_num+1)
                        continue
                    if (len(all_form_content)-1)<new_idx[idx]:
                        for idxs, box2 in enumerate(boxes):
                            if idxs<new_idx[idx] and (flags[idxs]==0 or flags[idxs]==1):
                                img_add+=1
                    # print('page',page_num+1)
                    # print(new_idx[idx]-img_add) 
                    # print(all_form_content)           
                    # print('all:',all_form_content[new_idx[idx]-img_add])
                    # print('head:',head_txt)
                    # print('foot:',tail_txt)
                    Object={"type": "table", "content": all_form_content[new_idx[idx]-img_add], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                    list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')

        elif len(boxes)==0:
            for idx, txt in enumerate(now_word_txt):
                Object={"type": "text", "content": txt, "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')

        else:
            used_idxs=[]
            for idx, box in enumerate(boxes_out):
                for idxs, box2 in enumerate(all_word_txt_box):
                    if idxs in used_idxs:
                        continue
                    if box[1]<box2:
                        if flags[new_idx[idx]]==0 or flags[new_idx[idx]]==1:
                            Object={"type": "graph", "content": big_lst_name[new_idx[idx]], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                            list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')
                        else:
                            img_add=0
                            if not all_form_content:
                                print('content error, page:',page_num+1)
                                continue
                            if (len(all_form_content)-1)<new_idx[idx]:
                                for idxss, box22 in enumerate(boxes):
                                    if idxss<new_idx[idx] and (flags[idxss]==0 or flags[idxss]==1):
                                        img_add+=1
                            # print(page_num)
                            # print('----------')
                            Object={"type": "table", "content": all_form_content[new_idx[idx]-img_add], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                            list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')
                        break

                    else:
                        # print(idxs)
                        # print('--------------------',now_word_txt[idxs])
                        Object={"type": "text", "content": now_word_txt[idxs], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                        list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')
                        used_idxs.append(idxs)

                if len(used_idxs)==len(all_word_txt_box):
                    if flags[new_idx[idx]]==0 or flags[new_idx[idx]]==1:
                        Object={"type": "graph", "content": big_lst_name[new_idx[idx]], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                        list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')
                    else:
                        img_add=0
                        if not all_form_content:
                            print('content error, page:',page_num+1)
                            continue
                        if (len(all_form_content)-1)<new_idx[idx]:
                            for idxss, box22 in enumerate(boxes):
                                if idxss<new_idx[idx] and (flags[idxss]==0 or flags[idxss]==1):
                                    img_add+=1
                        # print('page:',page_num+1)
                        # print(new_idx[idx]-img_add)
                        # print(all_form_content)
                        Object={"type": "table", "content": all_form_content[new_idx[idx]-img_add], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                        list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')

            if len(used_idxs)<len(all_word_txt_box):
                for idxs, box2 in enumerate(all_word_txt_box):
                    if idxs in used_idxs:
                        continue
                    Object={"type": "text", "content": now_word_txt[idxs], "header": head_txt, "footer": tail_txt, "page_num": page_num+1}
                    list_obj.write(json.dumps(Object, indent=4, sort_keys=True, ensure_ascii=False)+'\n')
                    used_idxs.append(idxs)
        list_obj.close()

    #---------------------------------------------------------------
    #save image form 輸出可視化結果
    page_num_box=[]
    for idxs,box in enumerate(boxes):
        page_num_box.append((page_num+1))
        if flags[idxs]==3:
            big_lst_name[idxs]=big_lst_name[idxs].replace('nonline','FORM')
            cv2.imwrite(big_lst_name[idxs], img[box[1]:box[3],box[0]:box[2]])
            if show_all:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (128,0,128), 3, 8)
        elif flags[idxs]==2:
            big_lst_name[idxs]=big_lst_name[idxs].replace('nonline','FORM')
            cv2.imwrite(big_lst_name[idxs], img[box[1]:box[3],box[0]:box[2]])
            if show_all:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0,0,255), 3, 8)
        else:
            cv2.imwrite(big_lst_name[idxs], img[box[1]:box[3],box[0]:box[2]])
            if show_all:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255,0,0), 3, 8)
    #--------------------------------------------------------------
    if show_all:
        cv2.imwrite("{}/{}/{}_{}.jpg".format(all_dir[3],name,name,tmp), img1)
    # print(remain_txt_head)

    return title,big_lst_name,matchtitle,matchtitlebox,boxes,page_num_box,title_page_num,remain_txt_head,remain_txt_tail,remain_form_head,remain_form_tail,h,w

#---------------------------------------------------
def takeSecond(elem):
    return elem[1]
#----------------------------------
def dist_2col(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    out = cv2.adaptiveThreshold(255-gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    (h,w)=out.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    out2 = cv2.medianBlur(out,3)
    out1 = cv2.dilate(out2,kernel)

    thresh1=out1.copy() 
    thresh2=out1.copy()
    dst=thresh1.copy()
    size_w=10
    #---------------------------------------------------------------
    for j in range(0,w):
        for i in range(0,h):
            dst[i,j]=255
    dst2=dst.copy()
    #-------
    a=[0 for z in range(0,w)]
    b=0
    for j in range(0,w): #遍历一列 
        for i in range(0,h):  #遍历一行
            if thresh1[i,j]==0:  #如果改点为黑点
                a[j]+=1 #该列的计数器加一计数
                thresh1[i,j]=255  #记录完后将其变为白色
        b=b+(h-a[j])
    c=b/w*0.5

    for j in range(0,w): #遍历一列 
        for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i,j]=0   #涂黑
    for j in range(0,w):
        if ((h-a[j]) < c):
            for i in range(0,h):
                dst[i,j]=0

    b=0
    for j in range(0,w):
        if dst[0,j]==0:
            b+=1
        else:
            if b!=0 and b<size_w+1:
                for x in range(j-size_w,j):
                    for y in range(0,h):
                        dst[y,x]=255
            b=0

    word_w=[]
    b=c=white_w=0
    w_max=0
    w_max_l=0
    w_max_r=0
    for j in range(4,w-4):
        if dst[0,j]!=dst[0,j+1]:
            if dst[0,j+1]==255:
                b=j
            if dst[0,j]==255 and b!=0:
                white_w=j-b
                if white_w>w_max:
                    w_max_l=b
                    w_max_r=j
                    w_max=white_w
                word_w.insert(c,white_w)
                c+=1
                b=0
    if ((w_max_r+w_max_l)/2)>(w/2):
        right=True
        line_pos=w_max_l
    else:
        right=False
        line_pos=w_max_r
    word_w.sort()
    word_w.reverse()

    if len(word_w)>1:
        if ((word_w[0]>w*0.4 and word_w[0]<w*0.8) and (word_w[0]>word_w[1]*1.25) and (word_w[1]>w*0.05)):
            return True,right,line_pos
        else:
            return False,right,line_pos
    else:
        return False,right,line_pos
#---------------------------------------------------
def dist_nonline_2col(src_im):
    gray = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
    out = cv2.adaptiveThreshold(255-gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    (h,w)=out.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    out2 = cv2.medianBlur(out,3)
    out1 = cv2.dilate(out2,kernel)
    thresh1=out1.copy() 
    thresh2=out1.copy()
    dst=thresh1.copy()
    size_w=5
    #---------------------------------------------------------------
    for j in range(0,w):
        for i in range(0,h):
            dst[i,j]=255
    dst2=dst.copy()
    #-------
    a=[0 for z in range(0,w)]
    b=0
    for j in range(0,w): #遍历一列 
        for i in range(0,h):  #遍历一行
            if thresh1[i,j]==0:  #如果改点为黑点
                a[j]+=1 #该列的计数器加一计数
                thresh1[i,j]=255  #记录完后将其变为白色
        b=b+(h-a[j])
    c=b/w*0.25

    for j in range(0,w): #遍历一列 
        for i in range((h-a[j]),h):  #从该列应该变黑的最顶部的点开始向最底部涂黑
            thresh1[i,j]=0   #涂黑
    for j in range(0,w):
        if (h-a[j]) < c:
            for i in range(0,h):
                dst[i,j]=0
    #-------------------------------------------------------------
    b=0
    d=[0 for z in range(0,h)]
    for j in range(0,h):
        for i in range(0,w):
            if thresh2[j,i]==255:
                d[j]+=1
                thresh2[j,i]=0
        b=b+d[j]
    c=b/h*0.1
    for j in range(0,h):
        for i in range(0,d[j]):
            thresh2[j,i]=255
    for j in range(0,h):
        if d[j]<c:
            for i in range(0,w):
                dst2[j,i]=0
    #---------------------------------------------------------------------------
    b=0
    for j in range(0,w):
        if dst[0,j]==0:
            b+=1
        else:
            if b!=0 and b<size_w+1:
                for x in range(j-size_w,j):
                    for y in range(0,h):
                        dst[y,x]=255
            b=0
    #------        
    b=0
    for j in range(0,h):
        if dst2[j,0]==0:
            b+=1
        else:
            if b!=0 and b<size_w+1:
                for x in range(j-size_w,j):
                    for y in range(0,w):
                        dst2[x,y]=255
            b=0
    #----------------------
    word_w=[]
    b=c=white_w=0
    w_max=0
    w_max_l=0
    w_max_r=0
    for j in range(4,w-4):
        if dst[0,j]!=dst[0,j+1]:
            if dst[0,j+1]==255:
                b=j
            if dst[0,j]==255 and b!=0:
                white_w=j-b
                if white_w>w_max:
                    w_max_l=b
                    w_max_r=j
                    w_max=white_w
                word_w.insert(c,white_w)
                c+=1
                b=0
    #-----------
    if ((w_max_r+w_max_l)/2)>(w/2):
        right=True
        line_pos=w_max_l
    else:
        right=False
        line_pos=w_max_r
    #-----------
    word_h=[]
    b=c=white_h=0
    for j in range(4,h-4):
        if dst2[j,0]!=dst2[j+1,0]:
            if (dst2[j+1,0]==255):
                b=j
            if dst2[j,0]==255 and b!=0:
                white_h=j-b
                word_h.insert(c,white_h)
                c+=1
                b=0
    word_w.sort()
    word_w.reverse()
    word_h.sort()
    word_h.reverse()
    hmean=np.mean(word_h)
    if len(word_w)>=2:
        if (word_w[0]>w/2 and word_w[0]<w*0.8) or len(word_h)<=2:
            if(len(word_h)<=2):
                return True,right,line_pos
            if((word_h[0]-5)>word_h[1] and (word_h[0]-5)>hmean and (word_h[1]-5)>hmean) or len(word_h)<=2:
                return True,right,line_pos
            else:
                return False,right,line_pos
        else:
            return False,right,line_pos
    else:
        return False,right,line_pos
#----------------------------------------------
def mg_pdf_non(list_pdf,list_non,flags):
    for idx,box in enumerate(list_pdf):
        for idx1,box1 in enumerate(list_non):
            if flags[idx1]!=1:
                if min(box1[2],box[2])+5<max(box1[0],box[0]) or min(box1[3],box[3])+30<max(box1[1],box[1]):
                    continue
                else:
                    list_non[idx1]=[min(box[0],box1[0]),min(box[1],box1[1]),max(box[2],box1[2]),max(box[3],box1[3])]
                    break
    return list_non

#---------------------------------------------
def find_horizontal_group_for_pdfminer(box_list):
    list_tree=[]
    flag=[True for _ in box_list]
    count=[]
    for idx,box in enumerate(box_list):
        for idx1,node in enumerate(list_tree):
            if (box[3]-node[1])*(box[1]-node[3]) <= 0:
                list_tree[idx1]=[min(box[0],node[0]),min(box[1],node[1]),max(box[2],node[2]),max(box[3],node[3])]
                flag[idx]=False
                count[idx1]=True
                break
        if flag[idx]:
            list_tree.append(box)
            flag[idx]=False
            count.append(False)
    new=[i for idx,i in enumerate(list_tree) if count[idx]]
    new1=sorted(new, key=lambda x: x[1])
    return new1

#---------------------------------------------
def use_miner_txt_nonline(src,page_num,txt_lst):
    h , w , _ = src.shape
    list_1=find_horizontal_group_for_pdfminer(txt_lst)
    list_new=[]
    if list_1:
        while 1:
            ischange=False
            for idx,box in enumerate(list_1):
                for boxn in list_1[idx+1:]:
                    if (box[1]-boxn[3])*(box[3]-boxn[1])<0 or min(abs(box[1]-boxn[3]),abs(box[3]-boxn[1]))< 0.07*h:
                        list_1[idx]=[min(box[0],boxn[0]),min(box[1],boxn[1]),max(box[2],boxn[2]),max(box[3],boxn[3])]
                        list_1.remove(boxn)
                        ischange=True
                        break
                if ischange:
                    break
            if not ischange:
                break
        for i in list_1:
            if i[3]-i[1]<0.1*h:
                list_new.append(i)
        return list_new
    return []
#------------------------------------------
def non_line_ovlap(box1,box2):# 另一個查看是否重疊
    minx1=min(box1[0],box1[2])
    miny1=min(box1[1],box1[3])
    maxx1=max(box1[0],box1[2])
    maxy1=max(box1[1],box1[3])
    minx2=min(box2[0],box2[2])
    miny2=min(box2[1],box2[3])
    maxx2=max(box2[0],box2[2])
    maxy2=max(box2[1],box2[3])
    if max(minx1,minx2)>min(maxx1,maxx2) or max(miny1,miny2)>min(maxy1,maxy2)+20:
        return False
    else:
        return [min(minx1,minx2),min(miny1,miny2),max(maxx1,maxx2),max(maxy1,maxy2)]
#--------------------------------------------------
def drawbox_v1(form_lst,f_img_lst,txt_lst,img_lst,val_txt,origin_img_dir,name,all_dir,show_all,show_caption,thresh_h,thresh_t,txt_lst_se,val_txt_se,head_txt,tail_txt,head_txt_page,tail_txt_page):# 多執行序執行統整
    initial('{}/{}'.format(all_dir[0],name))
    initial('{}/{}'.format(all_dir[1],name))
    initial('{}/{}'.format(all_dir[2],name))
    initial('{}/{}'.format(all_dir[6],name))
    initial('{}/{}'.format(all_dir[7],name))
    # initial('clean_whole_txt/{}'.format(name))
    initial('./clean_TXT/{}'.format(name))
    initial('./content/{}'.format(name))
    initial('./findtitle/{}'.format(name))
    initial('./output_json/{}'.format(name))
    if show_all:
        initial('{}/{}'.format(all_dir[3],name))
    
    lst=[]
    for page_num in range(len(form_lst)):
        res1=multi_draw(page_num,form_lst[page_num],f_img_lst[page_num],txt_lst[page_num],img_lst[page_num],val_txt[page_num],origin_img_dir,name,all_dir,show_all,show_caption,thresh_h[page_num],thresh_t[page_num],txt_lst_se[page_num],val_txt_se[page_num],head_txt,tail_txt,head_txt_page,tail_txt_page)
        lst.append(res1)

    lst_name=[]
    lst_caption=[]
    titlecontent=[]
    titlebox=[]
    pagenum=[]
    imgboxes=[]
    titlepagenum=[]
    remain_txt_head=[]
    remain_txt_tail=[]
    remain_form_head=[]
    remain_form_tail=[]
    page_h=0
    page_w=0
    # print(lst)
    for i in lst:
        a,b,c,d,e,f,g,h,j,k,l,m,n=i
        # print(h)
        # print('------------------')
        lst_caption.extend(a)
        lst_name.extend(b)
        titlecontent.extend(c)
        titlebox.extend(d)
        imgboxes.extend(e)
        pagenum.extend(f)
        titlepagenum.extend(g)
        remain_txt_head.append(h)
        remain_txt_tail.append(j)
        remain_form_head.append(k)
        remain_form_tail.append(l)
        page_h=m
        page_w=n
    # print(remain_form_head)
    # print(remain_form_tail)
    #--------------------------------------------------------------
    #找不同頁的title in No title case
    for idx,title in enumerate(lst_caption):
        if (title==''):
            match_caption_num=0
            match_caption_box=[]
            match_caption_size=page_h
            match_caption_idx=0
            if (imgboxes[idx][1]<(page_h*0.25)):
                if (pagenum[idx]-1)>0:
                    for idxs,txtbox in enumerate(titlebox):
                        if (pagenum[idx]-1)==titlepagenum[idxs]:
                            if ((imgboxes[idx][2])>(txtbox[0]*page_w))and((imgboxes[idx][0])<(txtbox[2]*page_w)):
                                if (txtbox[1]*page_h)>(page_h*0.5):
                                    new_size=abs((txtbox[1]*page_h)-page_h)
                                    if (new_size < match_caption_size):
                                        match_caption_size=new_size
                                        match_caption_num=idxs
                                        match_caption_box=titlebox[idxs]
            if (match_caption_box!=[]):
                lst_caption[idx]=titlecontent[match_caption_num]
    #--------------------------------------------------------------
    #區分重複多餘title問題
    for idx,title in enumerate(lst_caption):
        titlematch=re.findall(r'(图+表*\s*\d+)',lst_caption[idx])
        # for match in titlematch:
        if (len(titlematch)>1):
            for idxs,box in enumerate (imgboxes):
                if pagenum[idxs]==pagenum[idx] and idx!=idxs:
                    if (abs(imgboxes[idx][1]-box[1]))<50:
                        if (imgboxes[idx][0]+imgboxes[idx][2])<(box[0]+box[2]):
                            secondmatch=re.search(titlematch[1],lst_caption[idx])
                            lst_caption[idx]=lst_caption[idx][:secondmatch.start(0)]
                            secondmatch2=re.search(titlematch[1],lst_caption[idxs])
                            lst_caption[idxs]=lst_caption[idxs][secondmatch2.start(0):]
                        else:
                            secondmatch=re.search(titlematch[1],lst_caption[idx])
                            lst_caption[idx]=lst_caption[idx][secondmatch.start(0):]
                            secondmatch2=re.search(titlematch[1],lst_caption[idxs])
                            lst_caption[idxs]=lst_caption[idxs][:secondmatch2.start(0)]
    #---------------------------------------------------------------
    for idx,title in enumerate(lst_caption):
        if (title==''):
            for idxs,box in enumerate(imgboxes):
                if (pagenum[idx]-pagenum[idxs])==1:
                    if ((box[2]-box[0])*(box[3]-box[1]))>(page_h*page_w/2):
                        lst_caption[idx]=lst_caption[idxs]
    #---------------------------------------------------------------
    with open("{}/{}.txt".format(all_dir[5],name), "w",encoding="utf-8") as text_file1:
        for itm in range(len(lst_caption)):
            titles=lst_caption[itm] if len(lst_caption[itm])>0 else 'No Title'
            text_file1.write("{}   {}\n".format(lst_name[itm],titles))
        text_file1.close()
    return remain_txt_head,remain_txt_tail,remain_form_head,remain_form_tail
#--------------
def merge_txt(dir2,dir5,name,page_num):#merge all page  context into one files
    with open("{}/{}.txt".format(dir5,name), "w",encoding="utf-8") as all_text_file:
        for num in range(page_num):
            tmp = "%03d" % (num+1)
            with open("{}/{}/{}_{}.txt".format(dir2,name,name,tmp), "r",encoding="utf-8") as text_file:
                all_text_file.write(text_file.read())
                text_file.close()
        all_text_file.close()

#--------------------------------
def merge_json(name,page_num,remain_txt_head,remain_txt_tail,remain_form_head,remain_form_tail):
    # print(remain_txt_head)
    
    for num in range(page_num):
        if (num < page_num-1):
            if (remain_txt_tail[num]) & (remain_txt_head[num+1]):
                print('merge txt:',num+1)
                # fp = open("test.txt", "a",encoding="utf-8")
                # fp.write('change page:{}\n'.format(num+1))
                # fp.close()
                # print('change page:',num+1)
                next_txt=[]
                add=str()
                with open("output_json/{}/{}.json".format(name,num+2), "r",encoding="utf-8") as next_file:
                    for next_object in next_file.readlines():
                        next_txt.append(next_object)
                    next_file.close()

                for idx, next_object in enumerate(next_txt):
                    if re.search(r'"text"',next_object):
                        add=next_txt[idx-4].replace('"content":','').strip()[:-1]
                        del next_txt[idx+1]
                        del next_txt[idx]
                        del next_txt[idx-1]
                        del next_txt[idx-2]
                        del next_txt[idx-3]
                        del next_txt[idx-4]
                        del next_txt[idx-5]
                        break
                # print(add)
                with open("output_json/{}/{}.json".format(name,num+2), "w",encoding="utf-8") as next_file:
                    for next_object in next_txt:
                        next_file.write(next_object)
                    next_file.close()

                this_txt=[]
                with open("output_json/{}/{}.json".format(name,num+1), "r",encoding="utf-8") as this_file:
                    for this_object in this_file.readlines():
                        # print(this_object)
                        this_txt.append(this_object)
                    this_file.close()

                this_txt.reverse()
                for idx, this_object in enumerate(this_txt):
                    if re.search(r'"text"',this_object):
                        this_txt[idx+4]=this_txt[idx+4][:-3]+add[1:-1]+'",\n'
                        break

                this_txt.reverse()
                with open("output_json/{}/{}.json".format(name,num+1), "w",encoding="utf-8") as this_file:
                    for this_object in this_txt:
                        # all_text_file.write(this_object)
                        this_file.write(this_object)
                    this_file.close()

            elif (remain_form_tail[num]) & (remain_form_head[num+1]):
                print('merge form:',num+1)
                this_txt=[]
                add=str()
                with open("output_json/{}/{}.json".format(name,num+1), "r",encoding="utf-8") as this_file:
                    for this_object in this_file.readlines():
                        this_txt.append(this_object)
                    this_file.close()

                this_txt.reverse()
                up_idx=0
                down_idx=0
                for idx,this_object in enumerate(this_txt):
                    # print(this_object)
                    if re.search(r'"table"',this_object):
                        # print('yes1')
                        down_idx=idx-1
                    if re.search(r'{',this_object):
                        # print('yes2')
                        up_idx=idx
                        break

                for idx in range(up_idx,down_idx-1,-1):
                    if down_idx+6<= idx <=up_idx-2:
                        add+=(this_txt[idx])
                    del this_txt[idx]
                this_txt.reverse()
                add=add[:-1]+',\n'
                # print('add:',add)

                # print(this_txt)

                with open("output_json/{}/{}.json".format(name,num+1), "w",encoding="utf-8") as this_file:
                    for this_object in this_txt:
                        this_file.write(this_object)
                        # all_text_file.write(this_object)
                    this_file.close()

                next_txt=[]
                with open("output_json/{}/{}.json".format(name,num+2), "r",encoding="utf-8") as next_file:
                    for next_object in next_file.readlines():
                        next_txt.append(next_object)
                    next_file.close()
                
                # print(type(next_txt))
                next_txt.insert(2,add)
                with open("output_json/{}/{}.json".format(name,num+2), "w",encoding="utf-8") as next_file:
                    for next_object in next_txt:
                        next_file.write(next_object)
                        # all_text_file.write(next_object)
                    next_file.close()
    
    with open("output_json/{}.json".format(name), "w",encoding="utf-8") as all_text_file:
        for num in range(page_num):
            with open("output_json/{}/{}.json".format(name,num+1), "r",encoding="utf-8") as text_file:
                all_text_file.write(text_file.read())
                text_file.close()
        all_text_file.close()

#---------------------------------
def find_horizontal_group(box_list):
    list_tree=[]
    flag=[True for _ in box_list]
    for idx,box in enumerate(box_list):
        for idx1,nodes in enumerate(list_tree):
            for node in nodes:
                if (box[3]-node[1])*(box[1]-node[3]) <= 0:
                    list_tree[idx1].append(box)
                    flag[idx]=False
                    break
            if not flag[idx]:
                break
        if flag[idx]:
            list_tree.append([box])
            flag[idx]=False
    sort_list_tree=[]
    for nodes in list_tree:
        # if len(nodes)>1:
            sort_list_tree.append(sorted(nodes, key=lambda x: x[0]))
    return sort_list_tree
#----
def mergev1(box_list):#水平聚類 
    list_tree=find_horizontal_group(box_list)
    list_tree1=[]
    for nodes in list_tree:
        if len(nodes)>1:
            now=nodes[0]
            filt_hor=[]
            for node in nodes[1:]:
                if min(abs(node[2]-now[0]),abs(now[2]-node[0])) < max((node[3]-node[1]),(now[3]-now[1]))*0.7:
                    now=[min(now[0],node[0]),min(now[1],node[1]),max(now[2],node[2]),max(now[3],node[3])]
                else:
                    filt_hor.append(now)
                    now=node
            filt_hor.append(now)
            if len(filt_hor)>1:
                list_tree1.append(filt_hor)

    list_tree2=real_mgovlap(list_tree1,True)
    list_tree_org=real_mgovlap(list_tree,False)
    return list_tree2,list_tree_org
#----
def real_mgovlap(list_tree1,istree):#真正合併重疊
    list_tree2=[]
    for nodes in list_tree1:
        while 1:
            if len(nodes)>1:
                all_not_change=True
                for idx,node in enumerate(nodes):
                    for node1 in nodes[idx+1:]:
                        if min(node1[2],node[2])<max(node1[0],node[0]) or min(node1[3],node[3])<max(node1[1],node[1]):
                            continue
                        else:
                            nodes[idx]=[min(node1[0],node[0]),min(node1[1],node[1]),max(node1[2],node[2]),max(node1[3],node[3])]
                            nodes.remove(node1)
                            all_not_change=False
                            break
                    if not all_not_change:
                        break
                if all_not_change:
                    break
            else:
                break
        if istree:
            if len(nodes)>1:
                list_tree2.append(nodes)
        else:
            list_tree2.extend(nodes) 
    return list_tree2
#---
def find_similarity(h,w,more_box_list,less_box_list):#無格線表格聚類
    more1=[]
    for box in more_box_list:#filt out overlap part 
        notovlap=True
        for boxin in less_box_list:
            if min(box[2],boxin[2])<max(box[0],boxin[0]) or min(box[3],boxin[3])<max(box[1],boxin[1]):
                continue
            else:
                notovlap=False
                break
        if notovlap:
            more1.append(box)
    for box in more1:
        for boxin in less_box_list:
            if min(abs(box[1]-boxin[3]),abs(box[3]-boxin[1]))< 0.025*h :
                if boxin[0]> box[2] or boxin[2]<box[0]:
                    continue
                elif (box[2]-box[0])<=(boxin[2]-boxin[0])*1.2:
                    less_box_list.append(box)
                    break

    for box in more1:
        if box not in less_box_list:
            for boxin in less_box_list:
                if box[1]<=boxin[3] and box[3]>=boxin[1]:
                    less_box_list.append(box)
                    break
    return less_box_list
#########################

#--------------
def download_pdf(file_path,name): #use in API 
    # print('--------------')
    print(name)
    print(file_path)
    r=requests.get(file_path,stream=True)
    print(r)
    # print(r.status_code,'-----------------------')
    if r.status_code == requests.codes.ok:
        with open('./pdf_data/{}'.format(name),'wb') as pdf:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    pdf.write(chunk)
            pdf.close()
        print("{}.pdf download success".format(name))
 
        path_file_number=glob.glob('./pdf_data/*.PDF')#或者指定文件下个数
        # path_file_number=glob.glob(pathname='*.py') #获取当前文件夹下个数
        print(path_file_number)
        print(len(path_file_number)) 

        return True
    else:
        with open("output_json/{}.json".format(name), "w",encoding="utf-8") as f:
            f.truncate()
            f.close()
        print('download pdf failed')
        return False

#--------------
def main(file_path):
    #print(file_path)
    file_name=os.path.split(file_path)[-1]
    #print(file_name)
    name=file_name.split('.')[0]
   # print(name)
    indir='./pdf_data'
    origin_img_dir='./test'
    all_dir=['./FORM','./TXT','./IMAGE','./whole','./whole_txt','./caption','./nonline','./outfile']
    show_all=True
    show_whole_txt=True
    show_caption=True
    json_path='output_json/{}.json'.format(name)
    if download_pdf(file_path,file_name):
        
        path_file_number=glob.glob('./pdf_data/*.PDF')#或者指定文件下个数
        # path_file_number=glob.glob(pathname='*.py') #获取当前文件夹下个数
        print(path_file_number)
        print(len(path_file_number))

        tStart = time.time()
        print(444444444444) 
        t = Form_DCT(indir,file_name,origin_img_dir)
        # t_time=time.time()
        textract=time.time()
        t1= htplusimage(indir,file_name)
        # t1_time=time.time()
        print(333333333333333)
        t2=htplusimage_se(indir,file_name)
        print(555555555555555)
        # t2_time=time.time()

        val1,val2=t
        val3,val4,val_txt,thresh_h,thresh_t,head_txt,tail_txt,head_txt_page,tail_txt_page,page_all=t1
        val5,val6,val_txt_se=t2
        tStarts=time.time()
        # print(val1,'-----------------------------')
        if len(val1)!=page_all:
            print('error: pdf cannot detect')
            with open("output_json/{}.json".format(name), "w") as f:
                f.truncate()
                f.close()

        else:
            drawbox_v1(val1,val2,val3,val4,val_txt,origin_img_dir,name,all_dir,show_all,show_caption,thresh_h,thresh_t,val5,val_txt_se,head_txt,tail_txt,head_txt_page,tail_txt_page)

            if show_whole_txt:
                merge_txt(all_dir[1],all_dir[4],name,len(val_txt))
                merge_txt('clean_TXT','clean_whole_txt',name,len(val_txt))
        pdf_path='./pdf_data/{}'.format(file_name)
        if os.path.isfile(pdf_path):
            os.remove(pdf_path)
    else:
        if os.path.isfile(json_path):
            os.remove(json_path)
        os.mknod(json_path)

    print('Finish {}'.format(file_name))
    tEnd = time.time()
    print ("3func cost %f sec" % (tStarts - tStart))
    print ("extract text cost %f sec" % (tStarts - textract))
    print ("drawbox cost %f sec" % (tEnd - tStarts))
    print ("It cost %f sec" % (tEnd - tStart))#會自動做近位
# ------------------------------------------------
if __name__ == '__main__':
    ttStart = time.time()
    origin_img_dir='./test'
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

    main('http://static.cninfo.com.cn/finalpage/2019-09-25/1206946261.PDF')
    # for i in range(13,2,-1):
    #     main('{}.pdf'.format(i))
    # main('file:///media/rong/WD/46.pdf')
    # main('r6-1.pdf')
    print('total time')
    print(time.time()-ttStart)
    print('End')



