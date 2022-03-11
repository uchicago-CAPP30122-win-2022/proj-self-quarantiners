from selfq.model import analysis as an
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
    NavigationToolbar2Tk)
from tkinter import *
from tkinter.ttk import *

def launch():
    '''
    Launch tkinter and build interface
    '''
    root = Tk()
    root.title('Hi, welcome to Self-Quarantiners Data Analytics!')
    width= root.winfo_screenwidth() 
    height= root.winfo_screenheight()
    root.geometry("%dx%d" % (width, height))

    def close():
        root.quit()
        root.destroy()
    
    def open_covid():
        fig = an.plot_covid()
        covid_window = Toplevel(root)
        covid_window.title("Worldwide COVID-19 data")
        covid_window.geometry("%dx%d" % (width, height))
        covid_canvas = FigureCanvasTkAgg(fig, master = covid_window)  
        covid_canvas.draw()
        toolbar = NavigationToolbar2Tk(covid_canvas,
                                    covid_window)
        toolbar.update()
        covid_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=False)
        toolbar.pack(side=BOTTOM, fill=X)

    def open_safe():
        fig = an.plot_safe()
        window = Toplevel(root)
        window.title("Safe asset price data")
        window.geometry("%dx%d" % (width, height))
        canvas = FigureCanvasTkAgg(fig, master = window)  
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=False)
        toolbar.pack(side=BOTTOM, fill=X)

    def open_equity():
        fig = an.plot_equity()
        window = Toplevel(root)
        window.title("Equity indices data")
        window.geometry("%dx%d" % (width, height))
        canvas = FigureCanvasTkAgg(fig, master = window)  
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=False)
        toolbar.pack(side=BOTTOM, fill=X)

    def open_energy():
        fig = an.plot_energy()
        window = Toplevel(root)
        window.title("Energy price data")
        window.geometry("%dx%d" % (width, height))
        canvas = FigureCanvasTkAgg(fig, master = window)  
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=False)
        toolbar.pack(side=BOTTOM, fill=X)

    def open_comm():
        fig = an.plot_commodity()
        window = Toplevel(root)
        window.title("Commodities price data")
        window.geometry("%dx%d" % (width, height))
        canvas = FigureCanvasTkAgg(fig, master = window)  
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=False)
        toolbar.pack(side=BOTTOM, fill=X)

    def open_regr():
        window = Toplevel(root)
        window.title("Regression Analysis on COVID-19 data")
        window.geometry("%dx%d" % (width, height))
        lbl_text ='Please select the asset type you would like to regress on:'
        regr_lbl = Label(window, text = lbl_text, background='lightblue', font=('Helvetica', 15))
        lbl_safe_text ='Safe assets'
        regr_lbl_safe = Label(window, text = lbl_safe_text, background='lightblue', font=('Helvetica', 12))
        lbl_equity_text ='Equity indices'
        regr_lbl_equity = Label(window, text = lbl_equity_text, background='lightblue', font=('Helvetica', 12))
        lbl_energy_text ='Energy prices'
        regr_lbl_energy = Label(window, text = lbl_energy_text, background='lightblue', font=('Helvetica', 12))
        lbl_comm_text ='Commodity prices'
        regr_lbl_comm = Label(window, text = lbl_comm_text, background='lightblue', font=('Helvetica', 12))
        regr_trs_button = Button(window, width = 30,
                            text = "Treasury Bond (10yr)", command = lambda: open_regr_key('TRS 10YR'))
        regr_gold_button = Button(window, width = 30,
                            text = "Gold spot price", command = lambda: open_regr_key('Gold spot'))
        regr_eur_button = Button(window, width = 30,
                            text = "EUR/USD F/X", command = lambda: open_regr_key('EUR/USD'))
        regr_jpy_button = Button(window, width = 30,
                            text = "JPY/USD F/X", command = lambda: open_regr_key('JPY/USD'))
        regr_gbp_button = Button(window, width = 30,
                            text = "GBP/USD F/X", command = lambda: open_regr_key('GBP/USD'))
        regr_nasdaq_button = Button(window, width = 30,
                            text = "NASDAQ Composite", command = lambda: open_regr_key('NASDAQ Composite'))
        regr_n100_button = Button(window, width = 30,
                            text = "NASDAQ100", command = lambda: open_regr_key('NASDAQ100'))
        regr_dow_button = Button(window, width = 30,
                            text = "Dow Jones Industrial Average", command = lambda: open_regr_key('DJIA'))
        regr_sp500_button = Button(window, width = 30,
                            text = "S&P 500", command = lambda: open_regr_key('S&P500'))
        regr_wti_button = Button(window, width = 30,
                            text = "WTI crude spot", command = lambda: open_regr_key('WTI'))
        regr_brent_button = Button(window, width = 30,
                            text = "Brent crude spot", command = lambda: open_regr_key('Brent'))
        regr_gas_button = Button(window, width = 30,
                            text = "Natural gas spot", command = lambda: open_regr_key('Natural gas'))
        regr_wheat_button = Button(window, width = 30,
                            text = "Wheat spot", command = lambda: open_regr_key('Wheat'))
        regr_corn_button = Button(window, width = 30,
                            text = "Corn spot", command = lambda: open_regr_key('Corn'))
        regr_soybean_button = Button(window, width = 30,
                            text = "Soybean spot", command = lambda: open_regr_key('Soybean'))
        regr_coffee_button = Button(window, width = 30,
                            text = "Coffee spot", command = lambda: open_regr_key('Coffee'))
        regr_sugar_button = Button(window, width = 30,
                            text = "Sugar spot", command = lambda: open_regr_key('Sugar'))

        regr_lbl.pack(side=TOP, padx=100, pady=100)
        regr_lbl_safe.place(x = 80, y = 250)
        regr_trs_button.place(x = 80, y = 300)
        regr_gold_button.place(x = 80, y = 330)
        regr_eur_button.place(x = 80, y = 360)
        regr_jpy_button.place(x = 80, y = 390)
        regr_gbp_button.place(x = 80, y = 420)
        regr_lbl_equity.place(x = 380, y = 250)
        regr_nasdaq_button.place(x = 380, y = 300)
        regr_n100_button.place(x = 380, y = 330)
        regr_dow_button.place(x = 380, y = 360)
        regr_sp500_button.place(x = 380, y = 390)
        regr_lbl_energy.place(x = 680, y = 250)
        regr_wti_button.place(x = 680, y = 300)
        regr_brent_button.place(x = 680, y = 330)
        regr_gas_button.place(x = 680, y = 360)
        regr_lbl_comm.place(x = 980, y = 250)
        regr_wheat_button.place(x = 980, y = 300)
        regr_corn_button.place(x = 980, y = 330)
        regr_soybean_button.place(x = 980, y = 360)
        regr_coffee_button.place(x = 980, y = 390)
        regr_sugar_button.place(x = 980, y = 420)

    def open_regr_key(key):
        fig = an.regr_key(key)
        window2 = Toplevel(root)
        window2.title('Regression Analysis on ' + key)
        window2.geometry("%dx%d" % (width, height))
        canvas = FigureCanvasTkAgg(fig, master = window2)  
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, window2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=False)
        toolbar.pack(side=BOTTOM, fill=X)

    covid_button = Button(root, width = 50,
                        text = "COVID data", command = open_covid)
    safe_button = Button(root, width = 50,
                        text = "Safe asset data", command = open_safe)
    equity_button = Button(root, width = 50,
                        text = "Equity indices data", command = open_equity)
    energy_button = Button(root, width = 50,
                        text = "Energy price data", command = open_energy)
    comm_button = Button(root, width = 50,
                        text = "Commodities price data", command = open_comm)
    reg_button = Button(root, width = 50,
                        text = "Regression Analysis", command = open_regr)
    quit_button = Button(root, text="Quit", command=close)

    lbl_text ='Hi, welcome! \nPlease select from the menu below :'
    root_lbl = Label(root, text = lbl_text, background='lightblue', font=('Helvetica', 18, 'bold'))

    root_lbl.pack(side=TOP, padx=100, pady=100)
    covid_button.pack(side=TOP, padx=10, pady=10)
    safe_button.pack(side=TOP, padx=10, pady=10)
    equity_button.pack(side=TOP, padx=10, pady=10)
    energy_button.pack(side=TOP, padx=10, pady=10)
    comm_button.pack(side=TOP, padx=10, pady=10)
    reg_button.pack(side=TOP, padx=10, pady=10)
    quit_button.pack(side=TOP, padx=50, pady=50)

    root.mainloop()