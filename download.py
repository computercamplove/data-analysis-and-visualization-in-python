import requests
import os, sys, os.path
from bs4 import BeautifulSoup
from urllib.request import urlopen
from zipfile import ZipFile
import re
import csv
from io import TextIOWrapper
import numpy as np

class Region():
    region_data = {
        "PHA": None,
        "STC": None,
        "JHC": None,
        "PLK": None,
        "KVK": None,
        "ULK": None,
        "LBK": None,
        "HKK": None,
        "PAK": None,
        "OLK": None,
        "MSK": None,
        "JHM": None,
        "ZLK": None,
        "VYS": None
    }

    region = {
        "PHA": "00.csv",
        "STC": "01.csv",
        "JHC": "02.csv",
        "PLK": "03.csv",
        "KVK": "19.csv",
        "ULK": "04.csv",
        "LBK": "18.csv",
        "HKK": "05.csv",
        "PAK": "17.csv",
        "OLK": "14.csv",
        "MSK": "07.csv",
        "JHM": "06.csv",
        "ZLK": "15.csv",
        "VYS": "16.csv"
    }


    headers = ['p1', 'p36', 'p37', 'p2a', 'weekday(p2a)', 'p2b', 
                'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13a', 
                'p13b', 'p13c', 'p14', 'p15', 'p16', 'p17', 'p18', 
                'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p27', 
                'p28', 'p34', 'p35', 'p39', 'p44', 'p45a', 'p47', 
                'p48a', 'p49', 'p50a', 'p50b', 'p51', 'p52', 'p53', 
                'p55a', 'p57', 'p58', 'a', 'b', 'd', 'e', 'f', 'g', 
                'h', 'i', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 'r', 's', 't', 'p5a', "REGION"]
    

class DataDownloader():
    

    def __init__(self, url="https://ehw.fit.vutbr.cz/izv/", folder="data", cache_filename="data_{}.pkl.gz"):
        self.url = url
        self.folder = folder
        self.cache_filename = cache_filename

    def download_data(self):
        filename = sys.argv[0]
        path_project = os.path.abspath(filename+"/..")
        data = os.path.join(path_project, self.folder)
        if not os.path.isdir(data):
            os.makedirs(data)
        resp = requests.get(self.url, headers={'User-Agent': 'XYZ'})
        resp.headers.get('content-type')
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.content, "html.parser")
        for tr in soup.findAll('tr'):
            tr_mesic = list(tr.children)[0]
            td_file = list(tr.children)[1]
            if "ZIP" in td_file.get_text():
                if 'Prosinec' in tr_mesic.get_text():
                    link_zip = self.url + list(td_file.children)[3].get('href')
                    respzip = urlopen(link_zip)
                    tempzip = open(data + "/" + re.findall(r"\d+", link_zip)[0] + ".zip", "wb")
                    tempzip.write(respzip.read())
                    tempzip.close()
                elif "2020" in tr_mesic.get_text() and "Září" in tr_mesic.get_text():
                    link_zip = self.url + list(td_file.children)[3].get('href')
                    respzip = urlopen(link_zip)
                    tempzip = open(data + "/2020.zip", "wb")
                    tempzip.write(respzip.read())
                    tempzip.close()

    def parse_region_data(self, region):
        filename = sys.argv[0]
        path_p = os.path.abspath(filename+"/..")
        reg = region
        path_data = path_p + "/" + self.folder
        if not os.path.isdir(path_data):
           self.download_data()
        files = os.listdir(path_data) #zip files
        list_region = Region()
        list_65 =[[] for i in range(64)]
        for f in files:
            with ZipFile(path_data + "/" + f, 'r') as zipObj:
                csv_files = zipObj.namelist()
                if list_region.region[reg] in csv_files:
                    csvname = list_region.region[reg] #o00.csv
                    with zipObj.open(csvname, 'r') as infile:
                        csvreader = csv.reader(TextIOWrapper(infile,'windows-1250'), delimiter=';')
                        for r in csvreader:
                            for i, elem in enumerate(r):
                                if elem == '' or elem == 'XX':
                                    elem = -1
                                if "," in str(elem):
                                    elem = elem.replace(",",".")
                                if ";" in str(elem):
                                    elem = elem.replace(";","")
                                if "A:" in str(elem) or "B:" in str(elem) or "D" in str(elem) or "E:" in str(elem) or "F:" in str(elem) or "G:" in str(elem):
                                    elem = -1
                                list_65[i].append(elem)
                                #print(f)

        zkratka = [region for i in range(len(list_65[0][0]))]
        list_65.append(zkratka)
        list_int = [1,2,4,5,6,7,34,35,32,15]
        list_float = [45,46,47,48,49,50]
        list_string = [0,8,9,10,11,13,12,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,41,36,37,38,39,40,42,43,44,51,52,53,54,55,56,57,58,59,60,61,62,63,64]
        
        #ndaaray
        data =[]
        for i , elem in enumerate(list_65):
            if i in list_string:
                data.append(np.array(elem).astype("str"))
            elif i == 3:
                data.append(np.array(elem).astype('datetime64'))
            elif i in list_int:
                data.append(np.array(elem).astype('i1'))
            elif i in list_float:
                data.append(np.array(elem).astype('float'))

        data_tuple = (list_region.headers, data)
        self.region_data[region] = data_tuple

        #cache
        '''
        TO-DO
        '''
        return data_tuple

    def get_list(self,regions=None):
        if regions != None:
            print(self.region_data[regions])
            if self.region_data[regions] != None:
                print(self.region_data[regions])
                return self.region_data[regions]
            '''
            elif self.region_data[regions] == None:
                result = self.parse_region_data(regions)
                print(result)
                return result
            '''


def main():
    data = DataDownloader()
    data.get_list("PHA")



if __name__ == '__main__':
    main()
