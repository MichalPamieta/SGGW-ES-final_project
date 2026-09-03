from tkinter import *
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)

class PeopleInRoom(object):
    def __init__(self):
        self.window = Tk()
        self.window.geometry("1024x500")
        self.window.title("Ile ludzi w pokoju")
        self.BuildWindow()
        self.window.mainloop()

    def BuildWindow(self):
        self.order = Label(self.window)
        self.order["text"] = "Podaj kolejne odczyty z pokoju:"
        self.order.grid(row=0, column=0, sticky=W)

        self.temp_label = Label(self.window)
        self.temp_label["text"] = "Temperatura:"
        self.temp_label.grid(row=1, column=0, sticky=W)

        self.temp_scale = Scale(self.window, from_=0, to=1, resolution=0.05, orient=HORIZONTAL)
        self.temp_scale.grid(row=1, column=1, sticky=W)

        self.light_label = Label(self.window)
        self.light_label["text"] = "Natężenie światła:"
        self.light_label.grid(row=2, column=0, sticky=W)

        self.light_scale = Scale(self.window, from_=0, to=1, resolution=0.05, orient=HORIZONTAL)
        self.light_scale.grid(row=2, column=1, sticky=W)

        self.sound_label = Label(self.window)
        self.sound_label["text"] = "Natężenie dźwięku:"
        self.sound_label.grid(row=3, column=0, sticky=W)

        self.sound_scale = Scale(self.window, from_=0, to=1, resolution=0.05, orient=HORIZONTAL)
        self.sound_scale.grid(row=3, column=1, sticky=W)

        self.motion_label = Label(self.window)
        self.motion_label["text"] = "Odczyt pomiaru ruchu:"
        self.motion_label.grid(row=4, column=0, sticky=W)

        self.motion_scale = Scale(self.window, from_=0, to=1, resolution=0.05, orient=HORIZONTAL)
        self.motion_scale.grid(row=4, column=1, sticky=W)

        self.var = IntVar()

        self.checkbox = Checkbutton(self.window, text="Czy pokazać wykresy funkcji przynależności zmiennych?", variable=self.var)
        self.checkbox.grid(row=5, column=0, sticky=W)

        self.find_button = Button(self.window, width=15)
        self.find_button["text"] = "Sprawdź liczbę osób"
        self.find_button["command"] = self.Calculate
        self.find_button.grid(row=6, column=0, sticky=W)

        self.space3 = Label(self.window)
        self.space3["text"] = ""
        self.space3.grid(row=7, column=0, sticky=W)

    def Calculate(self):
        # Tworzenie zmiennych lingwistycznych
        temp = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'temp')
        light = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'light')
        sound = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'sound')
        motion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'motion')

        eps = 0.001

        # Zmienna wyjściowa: liczba osób w pokoju
        occupancy = ctrl.Consequent(np.arange(0, 3.1, eps), 'occupancy')
        occupancy.defuzzify_method = 'centroid' # metoda wyostrzania

        # Definicja funkcji przynależności dla zmiennych lingwistycznych
        temp['low'] = fuzz.trapmf(temp.universe, [0, 0, 0.2, 0.4])
        temp['medium'] = fuzz.trapmf(temp.universe, [0.3, 0.4, 0.6, 0.8])
        temp['high'] = fuzz.trapmf(temp.universe, [0.7, 0.8, 1, 1])

        light['low'] = fuzz.trapmf(light.universe, [0, 0, 0.2, 0.4])
        light['medium'] = fuzz.trapmf(light.universe, [0.2, 0.3, 0.6, 0.85])
        light['high'] = fuzz.trapmf(light.universe, [0.6, 0.7, 1, 1])

        sound['low'] = fuzz.trapmf(sound.universe, [0, 0, 0.2, 0.5])
        sound['medium'] = fuzz.trapmf(sound.universe, [0.3, 0.4, 0.7, 0.8])
        sound['high'] = fuzz.trapmf(sound.universe, [0.6, 0.8, 1, 1])

        motion['low'] = fuzz.trapmf(motion.universe, [0, 0, 0.1, 0.3])
        motion['medium'] = fuzz.trapmf(motion.universe, [0.2, 0.25, 0.55, 0.8])
        motion['high'] = fuzz.trapmf(motion.universe, [0.5, 0.7, 1, 1])

        occupancy['none'] = fuzz.trapmf(occupancy.universe, [0, 0, 0.3, 0.7])
        occupancy['few'] = fuzz.trapmf(occupancy.universe, [0.3, 0.7, 1.4, 1.8])
        occupancy['many'] = fuzz.trapmf(occupancy.universe, [1.4, 1.7, 2.4, 2.7])
        occupancy['full'] = fuzz.trapmf(occupancy.universe, [2.4, 2.7, 3, 3])

        # Definicja reguł wnioskowania
        rule1 = ctrl.Rule(temp['low'] & (motion['low'] | sound['low'] | light['high']), occupancy['none'])
        rule2 = ctrl.Rule(temp['low'] & light['low'] & (motion['low'] | sound['low']), occupancy['none'])
        rule3 = ctrl.Rule(sound['low'] & motion['low'] & temp['low'], occupancy['none'])
        rule4 = ctrl.Rule(temp['low'] & (motion['medium'] | sound['medium']), occupancy['few'])
        rule5 = ctrl.Rule(temp['medium'] & (sound['low'] | sound['medium']) & light['high'], occupancy['few'])
        rule6 = ctrl.Rule(temp['medium'] & (motion['low'] | motion['medium']), occupancy['few'])
        rule7 = ctrl.Rule(light['medium'] & sound['medium'] & (motion['medium'] | motion['low']), occupancy['few'])
        rule8 = ctrl.Rule(temp['medium'] & (motion['high'] | light['low'] | sound['high']), occupancy['many'])
        rule9 = ctrl.Rule(temp['medium'] & motion['high'] & light['low'] & sound['high'], occupancy['many'])
        rule10 = ctrl.Rule(temp['medium'] & light['low'] & sound['high'] & motion['medium'], occupancy['many'])
        rule11 = ctrl.Rule(temp['medium'] & (sound['medium'] | sound['high']) & (light['low'] | light['medium']), occupancy['many'])
        rule12 = ctrl.Rule(temp['high'] & (light['low'] | light['medium']) & (sound['low'] | sound['medium']) & (motion['medium'] | motion['high']), occupancy['many'])
        rule13 = ctrl.Rule(temp['high'] & (motion['medium'] | light['medium']), occupancy['many'])
        rule14 = ctrl.Rule(temp['high'] & (motion['low'] | motion['high']), occupancy['full'])
        rule15 = ctrl.Rule(temp['high'] & (light['low'] | sound['high']), occupancy['full'])
        rule16 = ctrl.Rule(temp['high'] & light['low'] & sound['high'] & (motion['high'] | motion['low']), occupancy['full'])
        rule17 = ctrl.Rule(light['low'] & sound['high'] & (motion['high'] | motion['low']), occupancy['full'])
        
        # Tworzenie systemu sterowania
        occupancy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17])
        occupancy_sim = ctrl.ControlSystemSimulation(occupancy_ctrl)

        # Przykładowe wejścia do systemu
        occupancy_sim.input['temp'] = float(self.temp_scale.get())
        occupancy_sim.input['light'] = float(self.light_scale.get())
        occupancy_sim.input['sound'] = float(self.sound_scale.get())
        occupancy_sim.input['motion'] = float(self.motion_scale.get())

        # Obliczenie wyniku
        occupancy_sim.compute()
      
        # Wyświetlenie wyniku
        self.result = Label(self.window)
        self.result["text"] = "Wyniki:"
        self.result.grid(row=8, column=0, sticky=W)

        self.result1 = Label(self.window)
        self.result1["text"] = f"Obliczona wartość: {occupancy_sim.output['occupancy']:.2f}"
        self.result1.grid(row=9, column=0, sticky=W)

        self.result2 = Label(self.window)
        self.result2["text"] = f"Przewidywana liczba osób: {round(occupancy_sim.output['occupancy'])}"
        self.result2.grid(row=10, column=0, sticky=W)

        self.space4 = Label(self.window)
        self.space4["text"] = ""
        self.space4.grid(row=11, column=0, sticky=W)

        # Rysujemy wykres na obiekcie 'ax'
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, 3.1, eps), occupancy['none'].mf, label="none", linewidth=1)
        ax.plot(np.arange(0, 3.1, eps), occupancy['few'].mf, label="few", linewidth=1)
        ax.plot(np.arange(0, 3.1, eps), occupancy['many'].mf, label="many", linewidth=1)
        ax.plot(np.arange(0, 3.1, eps), occupancy['full'].mf, label="full", linewidth=1)
        oc = occupancy_sim.output['occupancy']
        max_y = max(occupancy['none'].mf[int(oc/eps)],occupancy['few'].mf[int(oc/eps)],occupancy['many'].mf[int(oc/eps)],occupancy['full'].mf[int(oc/eps)])
        ax.plot([occupancy_sim.output['occupancy'], occupancy_sim.output['occupancy']],[0,max_y], color='black', linewidth=3)
        ax.set_title("Wykres wypełnienia pokoju")
        ax.set_xlabel("occupancy")
        ax.set_ylabel("Membership")
        ax.legend()
        plot_window1 = FigureCanvasTkAgg(fig, self.window)
        plot_window1.draw()
        plot_window1.get_tk_widget().grid(row=0, column=2, rowspan=1000)


        fig1, ax1 = plt.subplots()
        ax1.plot(np.arange(0, 1.01, 0.01), temp['low'].mf, label="low", linewidth=1)
        ax1.plot(np.arange(0, 1.01, 0.01), temp['medium'].mf, label="medium", linewidth=1)
        ax1.plot(np.arange(0, 1.01, 0.01), temp['high'].mf, label="high", linewidth=1)
        ax1.set_title("Funkcja przynależności dla temperatury")
        ax1.set_xlabel("temp")
        ax1.set_ylabel("Membership")
        ax1.legend()

        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(0, 1.01, 0.01), light['low'].mf, label="low", linewidth=1)
        ax2.plot(np.arange(0, 1.01, 0.01), light['medium'].mf, label="medium", linewidth=1)
        ax2.plot(np.arange(0, 1.01, 0.01), light['high'].mf, label="high", linewidth=1)
        ax2.set_title("Funkcja przynależności dla światła")
        ax2.set_xlabel("light")
        ax2.set_ylabel("Membership")
        ax2.legend()

        fig3, ax3 = plt.subplots()
        ax3.plot(np.arange(0, 1.01, 0.01), sound['low'].mf, label="low", linewidth=1)
        ax3.plot(np.arange(0, 1.01, 0.01), sound['medium'].mf, label="medium", linewidth=1)
        ax3.plot(np.arange(0, 1.01, 0.01), sound['high'].mf, label="high", linewidth=1)
        ax3.set_title("Funkcja przynależności dla dźwięku")
        ax3.set_xlabel("sound")
        ax3.set_ylabel("Membership")
        ax3.legend()

        fig4, ax4 = plt.subplots()
        ax4.plot(np.arange(0, 1.01, 0.01), motion['low'].mf, label="low", linewidth=1)
        ax4.plot(np.arange(0, 1.01, 0.01), motion['medium'].mf, label="medium", linewidth=1)
        ax4.plot(np.arange(0, 1.01, 0.01), motion['high'].mf, label="high", linewidth=1)
        ax4.set_title("Funkcja przynależności dla ruchu")
        ax4.set_xlabel("motion")
        ax4.set_ylabel("Membership")
        ax4.legend()

        try:
            self.window2.destroy()
        except AttributeError:
            pass
        except TclError:
            pass


        if self.var.get()==1:
            self.window2 = Tk()
            self.window2.geometry("1280x960")
            self.window2.title("Wykresy")
            self.BuildWindow2(fig1, fig2, fig3, fig4)
            self.window.mainloop()
    
    def BuildWindow2(self,fig1,fig2,fig3,fig4):
        plot_window1 = FigureCanvasTkAgg(fig1, self.window2)
        plot_window1.draw()
        plot_window1.get_tk_widget().grid(row=0, column=0, sticky=W)

        plot_window2 = FigureCanvasTkAgg(fig2, self.window2)
        plot_window2.draw()
        plot_window2.get_tk_widget().grid(row=0, column=1, sticky=W)

        plot_window3 = FigureCanvasTkAgg(fig3, self.window2)
        plot_window3.draw()
        plot_window3.get_tk_widget().grid(row=1, column=0, sticky=W)

        plot_window4 = FigureCanvasTkAgg(fig4, self.window2)
        plot_window4.draw()
        plot_window4.get_tk_widget().grid(row=1, column=1, sticky=W)


k = PeopleInRoom()