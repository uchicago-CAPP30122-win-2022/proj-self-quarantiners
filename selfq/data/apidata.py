import requests
import json
import csv
import os

BASE_DIR = os.path.join(os.path.dirname( __file__ ), '..' )
DATA_DIR = os.path.join(BASE_DIR, "data")

def treasury():
    '''
    Call api for treasury bond yield and save as a csv file.
    '''
    response_API = requests.get(
        "https://data.nasdaq.com/api/v3/datasets/USTREASURY/YIELD.csv?start_date=2020-01-01&end_date=2022-02-28&order=asc&api_key=mMRm5eQ7W25vmcrzEsHd")
    data = response_API.text
    with open(os.path.join(DATA_DIR, 'treasury.csv'), 'w') as myfile:
        myfile.write(data)
        
def wti():
    '''
    Call api for WTI price and save as a json file.
    '''
    response_API = requests.get(
        "https://api.eia.gov/series/?api_key=9y1puIfQBuHIvtLvyLvoaKcAalLbgjCzsI8FLQb2&series_id=PET.RWTC.D")
    data = response_API.text
    with open(os.path.join(DATA_DIR, 'wti.json'), 'w') as myfile:
        myfile.write(data)

def brent():
    '''
    Call api for Brent crude price and save as a json file.
    '''
    response_API = requests.get(
        "https://api.eia.gov/series/?api_key=9y1puIfQBuHIvtLvyLvoaKcAalLbgjCzsI8FLQb2&series_id=PET.RBRTE.D")
    data = response_API.text
    with open(os.path.join(DATA_DIR, 'brent.json'), 'w') as myfile:
        myfile.write(data)
        
def gas():
    '''
    Call api for natural gas price and save as a json file.
    '''
    response_API = requests.get("https://api.eia.gov/series/?api_key=9y1puIfQBuHIvtLvyLvoaKcAalLbgjCzsI8FLQb2&series_id=NG.RNGWHHD.D")
    data = response_API.text
    with open(os.path.join(DATA_DIR, 'gas.json'), 'w') as myfile:
        myfile.write(data)