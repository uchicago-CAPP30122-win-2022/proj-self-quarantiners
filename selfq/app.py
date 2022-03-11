from selfq.data import apidata
from selfq.model import analysis
from selfq.gui import gui

def run():
    apidata.treasury()
    apidata.wti()
    apidata.brent()
    apidata.gas()
    gui.launch()