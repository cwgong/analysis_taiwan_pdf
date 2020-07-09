import datetime
import os

file_path = "http://static.cninfo.com.cn/finalpage/2020-02-18/1207309925.PDF"
def test_1():
    timeStamp = 1381419600
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    otherStyleTime = dateArray.strftime("%Y-%m-%d")
    print(otherStyleTime)
    print(type(otherStyleTime))

def test_2():
    file_name = os.path.split(file_path)[-1]
    print(file_name)
if __name__ == "__main__":
    # test_1()
    test_2()