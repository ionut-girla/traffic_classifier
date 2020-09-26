# -*- coding: utf-8 -*-
"""
@author: ionut.girla
"""
# Importing the libraries
import csv
from functools import partial
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from os import path

list_of_traffic_types = ('ping', 'DNS', 'VoIP','Telnet')    #tipurile de trafic utilizate in deschiderea fisierelor de antrenare
list_of_models = {1:'LogisticRegression', 2:'KNN', 3:'Naive Bayes',4:'K-means'}  #tipurile de modele utilizate
list_of_components = ('1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th')    #lista utilizata in prelucrarea componentelor complexe

class Sdn_classifier:
    def __init__ (self):        #constructorul clasei
        self.model = "fara_model"   #initializarea modelului utilizat. acesta este selectat in timpul executiei
        #print ('apel_constructor')
        self.X = pd.DataFrame()     #initializarea matricei de caracteristici
        self.y = pd.DataFrame()     #initializarea vectorului de variabile dependente 
        self.X_train = pd.DataFrame()   #initializarea partii de antrenare a matricei de caracteristici
        self.X_test = pd.DataFrame()    #initializarea partii de test a matricei de caracteristici
        self.y_train = pd.DataFrame()   #initializarea partii de antrenare a vectorului de variabile dependente 
        self.y_test = pd.DataFrame()     #initializarea partii de antrenare a vectorului de variabile dependente 
        self.y_pred = pd.DataFrame()     #initializarea a vectorului de variabile dependente prezis
        self.y_pred_train = pd.DataFrame()  #initializare vectorului de predictii pentru datele de test
        self.dataset = pd.DataFrame()   #initializarea setului de date utilizat
        self.test_data = pd.DataFrame()     #initializarea setului de date pentru verificare locala
        self.sc = StandardScaler()      #initializarea uneltei de scalare a datelor

    def import_data(self,traffic_type):     #metoda responsabila cu incarcarea datelor in dataset
        new_data = pd.read_csv(traffic_type + '_training_data.csv',delimiter='\t')  #variabila locala new_data este incarcata cu valorile citite din csv-uri
        self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)   #datele deja prezente in dataset sunt concatenate cu datele incarcate la pasul anterior
        self.dataset.dropna(inplace=True)   #sunt eliminate inregistrarile incomplete
        self.dataset['Traffic Type'] = self.dataset['Traffic Type'].astype('category')  #convertim ultima coloana a setului de date in categorie

    def drop_complex_features (self,current_component,test_flag):   #metoda responsabila cu eliminarea componentelor complexe 
         if (test_flag == 0):   #pentru setul de date utilizat in antrenarea si testarea algoritmului
            self.dataset = self.dataset.drop(current_component + ' Component of Fourier transform of IAT',1)
         elif (test_flag == 1):     #pentru setul de date utilizat in testarea locala
            self.test_data = self.test_data.drop(current_component + ' Component of Fourier transform of IAT',1)

    def use_complex_features(self, current_component, test_flag):   #metoda responsabila cu aducerea componentelor complexe la o forma acceptata de algoritm
        if (test_flag == 0):    #pentru setul de date utilizat in antrenarea si testarea algoritmului
            self.dataset[current_component + '_fourier_comp_re'] = self.dataset[current_component + ' Component of Fourier transform of IAT'].map(lambda x: complex(x).real)
            self.dataset[current_component + '_fourier_comp_imag'] = self.dataset[current_component + ' Component of Fourier transform of IAT'].map(lambda x:complex(x).imag)   
            self.dataset = self.dataset.drop(current_component + ' Component of Fourier transform of IAT',1)
        elif (test_flag == 1):  #pentru setul de date utilizat in testarea locala
            self.test_data[current_component + '_fourier_comp_re'] = self.test_data[current_component + ' Component of Fourier transform of IAT'].map(lambda x: complex(x).real)
            self.test_data[current_component + '_fourier_comp_imag'] = self.test_data[current_component + ' Component of Fourier transform of IAT'].map(lambda x:complex(x).imag)   
            self.test_data = self.test_data.drop(current_component + ' Component of Fourier transform of IAT',1)

    def dataset_split(self):    #metoda responsabila cu impartirea datelor in date de test si de antrenare
        self.X = self.dataset.drop('Traffic Type',axis=1)   #matricea de caracteristici este initializata cu toate coloanele setului de date incarcat, mai putin ultima
        self.y = self.dataset['Traffic Type']   #vectorul de variabile dependente este initializat cu ultima coloana a setului de date
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.10)   #datele sunt impartite in date de test si date de antrenare cu o proportie de 10% date de test
        self.X_train = self.sc.fit_transform(self.X_train)  #instrument de scalare a datelor este potrivita pe datele de antrenare si acestea sunt standardizate prin eliminarea mediei si scalarea la varianta unitara
        self.X_test = self.sc.transform(self.X_test)    #standardizarea datelor de test
        sc_for_unsupervised = StandardScaler()  #initializarea unei unelte noi de scalare ce urmeaza sa fie aplicata intregii matrici de caracteristici
        self.X = sc_for_unsupervised.fit_transform(self.X)  #scalare matricei de caracteristici pentru algoritmul nesupervizat
        
    def model_select (self, model_name):    #metoda responsabila cu alegerea modelului
        #print('model select ->')
        if (model_name == 'LogisticRegression'):        
            #print('model select -> Logist')
            if path.exists("LogisticRegression"): #daca exista deja un fisier denumit LogisticRegression in locatia in care se gaseste codul sursa 
                infile = open('LogisticRegression','rb')    #deschidem respectivul fisier
                self.model = pickle.load(infile) #si incarcam modelul 
                infile.close()      #inchidem fisierul deschis
                #print('model select ->logistic -   > load file')
            else:                    
                self.model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs' )    #altfel este creat un nou model de acest tip
                #print('model select ->logistic -> new')
        elif (model_name == 'KNN'): #daca exista deja un fisier denumit KNN in locatia in care se gaseste codul sursa 
            if path.exists("KNN"):
                infile = open('KNN','rb')    #deschidem respectivul fisier
                self.model = pickle.load(infile) #si incarcam modelul 
                infile.close()
            else:
                self.model = KNeighborsClassifier(n_neighbors = 100, metric = 'minkowski', p = 2) #altfel este creat un nou model de acest tip
        elif (model_name == 'Naive Bayes'): #daca exista deja un fisier denumit Naive Bayes in locatia in care se gaseste codul sursa 
            if path.exists("Naive Bayes"):
                infile = open('Naive Bayes','rb') #deschidem respectivul fisier
                self.model = pickle.load(infile)  #si incarcam modelul 
                infile.close()
            else:
                self.model = GaussianNB()  #altfel este creat un nou model de acest tip 
        elif (model_name == 'K-means'):
            if path.exists("K-means"):   #daca exista deja un fisier denumit K-means in locatia in care se gaseste codul sursa
                infile = open('K-means','rb') #deschidem respectivul fisier
                self.model = pickle.load(infile) #si incarcam modelul
                infile.close()
            else:
                clusters = []   #altfel este creat un nou model
                clusters.append(self.X[367])               
                clusters.append(self.X[10153])    #element al lui X pe care algoritmul LogisticRegression il prezice corect drept DNS daca nu sunt folosite caracteristicile complexe
#                clusters.append(self.X[9771])   #element al lui X pe care algoritmul LogisticRegression il prezice corect drept DNS daca sunt folosite caracteristicile complexe
                clusters.append(self.X[15377])
                clusters.append(self.X[22364])
                centers = np.asarray(clusters)
                self.model = KMeans(n_clusters = 4, init = centers )    #modelul este creat cu centroide predefinite pentru imbunatatirea performantelor
#                self.model = KMeans(n_clusters = 4)
        title_text_var.set('Your model is: '+ str(self.model))  #pregatim un mesaj pentru afisarea modelului in interfata

    def model_fit_predict (self,model_name):    #metoda responsabila cu potrivirea si predictia
        if (model_name == 'LogisticRegression'):    
            self.model.fit(self.X_train, self.y_train)  #modelul este potrivit pe datele de antrenare
            self.y_pred = self.model.predict(self.X_test)   #sunt prezise etichetele pentru datele de test
            self.y_pred_train = self.model.predict(self.X_train)    #sunt prezise etichetele pentru datele de antrenare
        elif (model_name == 'KNN'):  
            self.model.fit(self.X_train, self.y_train)  #modelul este potrivit pe datele de antrenare
            self.y_pred = self.model.predict(self.X_test)   #sunt prezise etichetele pentru datele de test
            self.y_pred_train = self.model.predict(self.X_train)    #sunt prezise etichetele pentru datele de antrenare
        elif (model_name == 'Naive Bayes'):  
           self.model.fit(self.X_train, self.y_train)   #modelul este potrivit pe datele de antrenare
           self.y_pred = self.model.predict(self.X_test)#sunt prezise etichetele pentru datele de test
           self.y_pred_train = self.model.predict(self.X_train)     #sunt prezise etichetele pentru datele de antrenare
        elif (model_name == 'K-means'):
            y_kmeans = self.model.fit_predict(self.X)   #este creat un vector local ce contine clusterul corespunzator fiecarui punct de date
            infile = open('LogisticRegression','rb')    #este initializat modelul secundar folosit in etichetarea centroidelor
#            infile = open('KNN','rb')
            model_secundar = pickle.load(infile) #este initializat modelul secundar folosit in etichetarea centroidelor
            infile.close()
            label_1 = model_secundar.predict(self.model.cluster_centers_[0].reshape(1, -1)) #sunt etichetate centroidele
            label_2 = model_secundar.predict(self.model.cluster_centers_[1].reshape(1, -1))
            label_3 = model_secundar.predict(self.model.cluster_centers_[2].reshape(1, -1))
            label_4 = model_secundar.predict(self.model.cluster_centers_[3].reshape(1, -1))
            print (label_1 + ' ' +label_2 + ' '+label_3+ ' '+label_4 + ' ' )
            L  = [] #lista in care sunt salvate etichetele rezultate
            for i in range(len(y_kmeans)):
                if y_kmeans[i]==0: L.append((str(label_1)).split('\'')[1])  #facand presupunerea ca fiecare punct dintr-o centroida apartine aceluiasi tip de trafic 
                elif y_kmeans[i]==1: L.append((str(label_2)).split('\'')[1])    #sunt etichetate punctele de date
                elif y_kmeans[i]==2: L.append((str(label_3)).split('\'')[1])
                elif y_kmeans[i]==3: L.append((str(label_4)).split('\'')[1])
            self.y_pred = pd.Series(L)  #etichetele rezultate sunt atribuite vectorului de predictii
            self.y_pred.name = 'Traffic Type'
            
    def print_accuracy (self,model_name,flag):  #metoda responsabila de calcularea acuratetii algoritmului
        if (model_name == 'LogisticRegression') or (model_name == 'KNN') or (model_name == 'Naive Bayes') :           
            if (flag == 2):
                print ('Accuracy of  ' + model_name +' is: '+ str(accuracy_score(self.y_test, self.y_pred)*100.0))
                acc_text_var.set('Accuracy of  ' + model_name +' is: '+ str(accuracy_score(self.y_test, self.y_pred)*100.0))
            else:
                print ('Accuracy of  ' + model_name +' is: '+ str(accuracy_score(self.y_train, self.y_pred_train)*100.0))
                acc_text_var.set('Accuracy of  ' + model_name +' is: '+ str(accuracy_score(self.y_train, self.y_pred_train)*100.0))
        else:
            print ('Accuracy of ' + model_name +' is: '+ str(accuracy_score(self.y, self.y_pred)*100.0))
            acc_text_var.set('Accuracy of ' + model_name +' is: '+ str(accuracy_score(self.y, self.y_pred)*100.0))
     
    def print_precision (self,model_name,flag): #metoda responsabila de calcularea preciziei algoritmului
        if (model_name == 'LogisticRegression') or (model_name == 'KNN') or (model_name == 'Naive Bayes') :           
            if (flag == 2):
                print ('Precision of  ' + model_name +' is: '+ str(precision_score(self.y_test, self.y_pred, average = 'macro')*100.0))
                prec_text_var.set('Precision of  ' + model_name +' is: '+ str(precision_score(self.y_test, self.y_pred, average = 'macro')*100.0))
            else:
                print ('Precision of  ' + model_name +' is: '+ str(precision_score(self.y_train, self.y_pred_train, average = 'macro')*100.0))
                prec_text_var.set('Precision of  ' + model_name +' is: '+ str(precision_score(self.y_train, self.y_pred_train, average = 'macro')*100.0))
        else:
            print ('Precision of ' + model_name +' is: '+ str(precision_score(self.y, self.y_pred, average = 'macro')*100.0))
            prec_text_var.set('Precision of ' + model_name +' is: '+ str(precision_score(self.y, self.y_pred, average = 'macro')*100.0))
     
    def print_recall (self,model_name,flag):    #metoda responsabila de calcularea recall-ului algoritmului
        if (model_name == 'LogisticRegression') or (model_name == 'KNN') or (model_name == 'Naive Bayes') :           
            if (flag == 2):
                print ('Recall of  ' + model_name +' is: '+ str(recall_score(self.y_test, self.y_pred, average = 'macro')*100.0))
                recall_text_var.set('Recall of  ' + model_name +' is: '+ str(recall_score(self.y_test, self.y_pred, average = 'macro')*100.0))
            else:
                print ('Recall of  ' + model_name +' is: '+ str(recall_score(self.y_train, self.y_pred_train, average = 'macro')*100.0))
                recall_text_var.set('Recall of  ' + model_name +' is: '+ str(recall_score(self.y_train, self.y_pred_train, average = 'macro')*100.0))
        else:
            print ('Recall of ' + model_name +' is: '+ str(recall_score(self.y, self.y_pred, average = 'macro')*100.0))
            recall_text_var.set('Recall of ' + model_name +' is: '+ str(recall_score(self.y, self.y_pred, average = 'macro')*100.0))
                
    def save_model (self,model_name):   #metoda responsabila de salvarea modelului
        pickle.dump(self.model,open(model_name,'wb'))
    
    def print_conf_matrix(self,model_name, drw_frame,flag): #metoda responsabila de calcularea si afisarea matricei de confuzie
        for widget in drw_frame.winfo_children():
            widget.destroy()    #este stearsa matricea de confuzie anterioara 
        if (model_name == 'LogisticRegression') or (model_name == 'KNN') or (model_name == 'Naive Bayes') :           
            if (flag == 2):
                cm = confusion_matrix(y_true = self.y_test, y_pred = self.y_pred,labels = self.y.cat.categories)
            else:
                cm = confusion_matrix(y_true = self.y_train, y_pred = self.y_pred_train,labels = self.y.cat.categories)
        else:
            cm = confusion_matrix(y_true = self.y,y_pred = self.y_pred,labels =  self.y.cat.categories)
        print(cm)
        fig = plt.figure(figsize=(8,8))
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(self.y.unique()))
        plt.xticks(tick_marks, self.y.cat.categories, fontsize=12)
        plt.yticks(tick_marks, self.y.cat.categories, fontsize=12)
        plt.xlabel('Predicted Label', fontsize=15)
        plt.ylabel('True Label', fontsize=15)
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                color = 'black'   
                if cm[i][j] > 600:
                    color = 'white'
                plt.text(j, i, format(cm[i][j]), 
                        horizontalalignment='center',
                        color=color, fontsize=15)
        plt.gcf().canvas.draw()
        canvas = FigureCanvasTkAgg(fig, drw_frame)  
        canvas.draw()   #noua matrice este desenata in interfata
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

def train_the_model(drw_frame): #functie responsabila executarea pasilor algoritmului
    global last_model 
    if ((selected_model.get() != 0) and (last_model != selected_model.get())) :
            last_model = selected_model.get()
            obj.dataset_split()
            obj.model_select(list_of_models[selected_model.get()])
            obj.model_fit_predict(list_of_models[selected_model.get()])
            obj.print_accuracy(list_of_models[selected_model.get()],selected_data.get())
            obj.print_precision(list_of_models[selected_model.get()],selected_data.get())
            obj.print_recall(list_of_models[selected_model.get()],selected_data.get())
            obj.save_model(list_of_models[selected_model.get()])
            obj.print_conf_matrix(list_of_models[selected_model.get()],drw_frame, selected_data.get() )
    else:
            obj.print_conf_matrix(list_of_models[selected_model.get()],drw_frame, selected_data.get() )
            obj.print_accuracy(list_of_models[selected_model.get()],selected_data.get())
            obj.print_precision(list_of_models[selected_model.get()],selected_data.get())
            obj.print_recall(list_of_models[selected_model.get()],selected_data.get())

        
            

def createNewWindow():  #functie responsabila de definirea ferestrei de afisare a performantelor modelului
    if (selected.get() == 1):
        newWindow = tkinter.Toplevel(window)
        selected_model.set(0)  # initializing the choice
        selected_data.set(2)    #init to test test data
        text_frame = tkinter.Frame(newWindow, relief=tkinter.RAISED, bd=2)       
        text_fereastra_train = tkinter.Label(text_frame, textvariable=title_text_var, anchor=tkinter.NW, justify=tkinter.LEFT, wraplength=398)
        text_fereastra_train.grid(row=0, column=0, sticky="ew", padx=25, pady=15)
        accuracy_text = tkinter.Label(text_frame, textvariable=acc_text_var, anchor=tkinter.NW, justify=tkinter.LEFT, wraplength=398)
        accuracy_text.grid(row=1, column=0, sticky="ew", padx=25, pady=5)
        precision_text = tkinter.Label(text_frame, textvariable=prec_text_var, anchor=tkinter.NW, justify=tkinter.LEFT, wraplength=398)
        precision_text.grid(row=2, column=0, sticky="ew", padx=25, pady=5)
        recall_text = tkinter.Label(text_frame, textvariable=recall_text_var, anchor=tkinter.NW, justify=tkinter.LEFT, wraplength=398)
        recall_text.grid(row=3, column=0, sticky="ew", padx=25, pady=5)
        text_frame.grid(row = 0,sticky="ew", padx=25, pady=5)
        radio_frame = tkinter.Frame(newWindow, relief=tkinter.RAISED, bd=2)
        rad_1 = tkinter.Radiobutton(radio_frame,text='Logistic', value=1,variable = selected_model)
        rad_1.grid(row=0, column=0, sticky="ew", padx=25, pady=5)
        rad_2 = tkinter.Radiobutton(radio_frame,text='KNN', value=2,variable = selected_model)
        rad_2.grid(row=0, column=1, sticky="ew", padx=25, pady=5)
        rad_3 = tkinter.Radiobutton(radio_frame,text='Naive', value=3,variable = selected_model)
        rad_3.grid(row=0, column=2, sticky="ew", padx=25, pady=5)
        rad_4 = tkinter.Radiobutton(radio_frame,text='K-means', value=4,variable = selected_model)
        rad_4.grid(row=0, column=3, sticky="ew", padx=25, pady=5)
        radio_frame.grid(row = 1, sticky="ew", padx=25, pady=5)
        drw_frame=tkinter.Frame(newWindow,relief=tkinter.RAISED)
        drw_frame.grid(row = 0, column = 1 )        
        radio_frame_for_train_or_test_data = tkinter.Frame(newWindow, relief=tkinter.RAISED, bd=2)
        rad_1 = tkinter.Radiobutton(radio_frame_for_train_or_test_data,text='Train data results', value=1,variable = selected_data)
        rad_1.grid(row=0, column=0, sticky="ew", padx=25, pady=5)
        rad_2 = tkinter.Radiobutton(radio_frame_for_train_or_test_data,text='Test data results', value=2,variable = selected_data)
        rad_2.grid(row=0, column=1, sticky="ew", padx=25, pady=5)        
        radio_frame_for_train_or_test_data.grid(row = 2, sticky="ew", padx=25, pady=5)
        drw_frame=tkinter.Frame(newWindow,relief=tkinter.RAISED)
        drw_frame.grid(row = 0, column = 1 )
        f = plt.figure(figsize=(8,8))
        a = f.add_subplot(111)  #definire imagine initiala
        a.plot([0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0])
        canvas = FigureCanvasTkAgg(f, drw_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        buttonExample = tkinter.Button(newWindow, text = "Select", command = partial(train_the_model, drw_frame))
        buttonExample.grid(row = 2, column = 1,pady=15)
        newWindow.update_idletasks()
    else: 
        newWindow = tkinter.Toplevel(window)
        text_fereastra_train = tkinter.Label(newWindow, text = "Verify your data")
        text_fereastra_train.grid(row=0, column=0, sticky="ew", padx=25, pady=15)
        btn_open = tkinter.Button(newWindow, text="Open", command=open_file)
        btn_open.grid(row=1, column=0, sticky="ew", padx=5, pady=5)        
        text_simple_predict = tkinter.Label(newWindow, textvariable=text_simple_predict_var, anchor=tkinter.NW, justify=tkinter.LEFT, wraplength=398)
        text_simple_predict.grid(row=2, column=0, sticky="ew", padx=25, pady=15)
        radio_frame = tkinter.Frame(newWindow, relief=tkinter.RAISED, bd=2)
        rad_1 = tkinter.Radiobutton(radio_frame,text='Logistic', value=1,variable = selected_model)
        rad_1.grid(row=0, column=0, sticky="ew", padx=25, pady=5)
        rad_2 = tkinter.Radiobutton(radio_frame,text='KNN', value=2,variable = selected_model)
        rad_2.grid(row=0, column=1, sticky="ew", padx=25, pady=5)
        rad_3 = tkinter.Radiobutton(radio_frame,text='Naive', value=3,variable = selected_model)
        rad_3.grid(row=0, column=2, sticky="ew", padx=25, pady=5)
        radio_frame.grid(row = 3, sticky="ew", padx=25, pady=5)  
        newWindow.update_idletasks()       
        
def open_file():    #functie responsabila de deschiderea si incarcarea unui fisier in setul de date pentru predictia locala
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    with open(filepath, "r") as input_file:
        obj.test_data = pd.read_csv(input_file,delimiter='\t')
        simple_predict_results()
        
def simple_predict_results():   #functie responsabila cu efectuarea unei predictii locale
    if (selected_model.get() != 0):
        obj.model_select(list_of_models[selected_model.get()])
#    for current_component in list_of_components:
#        obj.use_complex_features(current_component,1)
    for current_component in list_of_components:
        obj.drop_complex_features(current_component,1) #no complex features
    x_test = obj.test_data.drop('Traffic Type',axis=1)
    x_test = obj.sc.transform(x_test)
    y_test = obj.test_data['Traffic Type']
    y_test_pred = obj.model.predict(x_test)
    resultsDF = pd.DataFrame({
            '   True Label  ':y_test,
            '  Predicted Label':y_test_pred
    })
    text_simple_predict_var.set('Results with '+ str(list_of_models[selected_model.get()]) + '\n' + str(resultsDF))
                       
    
if __name__ == '__main__':
    last_model = 0
    obj = Sdn_classifier()
    for traffic_type in list_of_traffic_types:  
        obj.import_data(traffic_type)
#    for current_component in list_of_components:
#        obj.use_complex_features(current_component,0)
    for current_component in list_of_components:
        obj.drop_complex_features(current_component,0) #no complex features
#    obj.dataset_split()
    window = tkinter.Tk()
    acc_text_var = tkinter.StringVar()
    prec_text_var = tkinter.StringVar()
    recall_text_var = tkinter.StringVar()
    title_text_var = tkinter.StringVar()
    text_simple_predict_var = tkinter.StringVar()
    acc_text_var.set(' ')
    prec_text_var.set(' ')
    recall_text_var.set(' ')
    text_simple_predict_var.set(' ') 
    title_text_var.set('Chose your model')
    selected = tkinter.IntVar()
    selected_data = tkinter.IntVar()    
    selected_model = tkinter.IntVar()
    selected.set(1)  # initializing the choice
    window.title(" Traffic Classifier ")
    window.configure(background='#555555')
    opt_1 = tkinter.Label(window,text=" Train and test a model ", font=("Helvetica",12),background='#555555',fg="#ffffff")
    opt_1.grid(row=0, column=0, sticky="ew", padx=25, pady=15)
    rad1 = tkinter.Radiobutton(window,text='Train', value=1,variable = selected)
    rad1.grid(row=0, column=1, sticky="ew", padx=25, pady=15)
    opt_2 = tkinter.Label(window,text=" Verify your own CSV ", font=("Helvetica",12),background='#555555',fg="#ffffff")
    opt_2.grid(row=1, column=0, sticky="ew", padx=25, pady=15)
    rad2 = tkinter.Radiobutton(window,text='Verify', value=2, variable = selected)
    rad2.grid(row=1, column=1, sticky="ew", padx=25, pady=15)
    btn = tkinter.Button(window, text="Select" ,command=createNewWindow,fg="#ffffff",background='#555555') 
    btn.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
    window.mainloop()
    

