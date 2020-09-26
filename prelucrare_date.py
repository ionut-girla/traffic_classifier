import subprocess, sys 
import signal 
import os 
import numpy as np
import pickle 
from statistics import median

cmd = "sudo ryu run /usr/local/lib/python3.8/dist-packages/ryu/app/version_6_of_simple_monitor.py" #comanda pentru pornirea procesului de monitorizare
TIMEOUT = 600 #durata rularii scriptului de prelucrare
flows = {} #dictionar de flow-uri

class Flow:
    def __init__(self, time_start, datapath, inport, ethsrc, ethdst, outport, packets, bytes):
        #print('apel constructor \n')
        #initializare variabile utilizate in prelucrarea datelor 
        self.time_start = time_start / 1000 #variabila utilizata pentru aflarea duratei comununicatiei in secunde
        self.datapath = datapath    #variabila specifica OVS-ului utilizata in calcularea unui ID unic pentru fiecare comunicatie
        self.inport = inport    #numarul portului sursa utilizat in calcularea unui ID unic pentru fiecare comunicatie
        self.ethsrc = ethsrc    #adresa MAC sursa utilizat in calcularea unui ID unic pentru fiecare comunicatie
        self.ethdst = ethdst    #adresa MAC destinatie utilizata in calcularea unui ID unic pentru fiecare comunicatie
        self.outport = outport  #numarul portului destinatie utilizat in calcularea unui ID unic pentru fiecare comunicatie
        self.last_time = time_start / 1000   #variabila utilizata pentru evitarea scrierilor multiple 
        self.packet_vector = [] #vector al lungimii pachetelor utilizat pentru determinarea cuartilelor, percentilelor si variantei
        self.IAT_vector = []    #vector al timpului intre sosiri utilizat pentru calcularea cuartilelor, percentilelor si variantei
        self.packet_per_second_vector = [] #vector al numarului de pachete pe secunde pentru calcularea variantei pachetelor pe secunda
        self.bytes_per_second_vector = []   #vector al numarului de octeti pe secunde pentru calcularea variantei pachetelor pe secunda
        self.status = 'ACTIVE'  #starea comunicatiei
        
        #Initializarea caracteristicilor statistice
        self.bytes_per_second_mean = 0.0    
        self.bytes_per_second_variance = 0.0
        self.packets_per_second_mean = 0.0
        self.packets_per_second_variance = 0.0
        self.packets_length_mean = 0.0
        self.packets_length_variance = 0.0
        self.packets_length_first_quartile = 0.0
        self.packets_length_third_quartile = 0.0
        self.packet_IAT_mean = 0.0
        self.packet_IAT_variance = 0.0 
        self.packet_IAT_first_quartile = 0.0 
        self.packet_IAT_third_quartile = 0.0 
        self.packets_length_ten_percentile = 0.0
        self.packets_length_ninety_percentile = 0.0
        self.packet_IAT_ten_percentile = 0.0
        self.packet_IAT_ninety_percentile = 0.0


        #Initializarea caracteristicilor scalare
        self.bytes_per_second_min = 11111.0 #valoare mult mai mare decat o valoare minima uzuala. A fost aleasa arbitrar
        self.bytes_per_second_max = 0.0
        self.packets_per_second_min = 11111.0
        self.packets_per_second_max = 0.0  
        self.packets_length_min = 11111.0
        self.packets_length_max = 0.0
        self.packet_IAT_min = 11111.0 
        self.packet_IAT_max = 0.0 
        self.flow_duration = 0.0
        self.flow_size_in_packets = packets 
        self.flow_size_in_bytes = bytes
        
        #Initializarea caracteristicilor complexe
        self.vector_of_fft_components = [0,0,0,0,0,0,0,0,0,0,0] #vector utilizat in stocarea componentelor tr Fourier a timpilor dintre sosiri
         
    def update_flow(self, packets, bytes, curr_time):
        #print('apel update ',curr_time,' \n')
        curr_time_in_seconds = curr_time / 1000 #timpul curent in secunde
        #print('apel update ',curr_time_in_seconds,' \n')
        number_of_packets = packets - self.flow_size_in_packets #numarul de pachete receptionate de la ultimul update
        number_of_bytes = bytes - self.flow_size_in_bytes   #numarul de octeti receptionati de la ultimul update
        if curr_time_in_seconds > self.last_time:   #daca am primit informatii noi de la OVS putem sa actualizam caracteristicile obiectului curent
            #print('apel if din update \n', packets,'  ' ,bytes)
            #print('apel update cu timp curent\n', curr_time_in_seconds, ' last time', self.last_time, ' si number of packets  ', packets, ' number of packets ', number_of_packets, ' \n')
            if (number_of_bytes > 0 or number_of_packets > 0):  #daca au fost observate pachete sau octeti calculam noile caracteristici
                packet_length = (bytes - self.flow_size_in_bytes) / (packets - self.flow_size_in_packets)   #lungimea pachetelor primite este calculata ca numarul de octeti de la ultima actualizare supra numarul de pachete de la ultima actualizare
                IAT_actual = curr_time_in_seconds - self.last_time  #timpul dintre sosiri a fost calculat drept timpul dintre doua actualizari consecutive
                self.packet_per_second_vector.append(number_of_packets / IAT_actual) #adaugam numarul de pachete primite pe secunda in vector
                self.bytes_per_second_vector.append(number_of_bytes / IAT_actual) #adaugam numarul de octeti primite pe secunda in vector
                self.packet_vector.append(packet_length)    #adaugam lungimea ultimelor pachete in vector
                self.IAT_vector.append(IAT_actual)  #adaugam timpul dintre sosiri in vector
                self.packet_vector.sort()   #sortam cei doi vectori
                self.IAT_vector.sort()  #sortam cei doi vectori
                
                #Actualizarea caracteristicilor scalare
                self.flow_size_in_packets = packets     #Dimensiunea comunicației in pachete
                self.flow_size_in_bytes = bytes     #Dimensiunea comunicației in octeți
                self.flow_duration = curr_time_in_seconds - self.time_start     #Durata comunicației        
                self.packet_IAT_max = max(self.packet_IAT_max, IAT_actual)      #Valoarea maxima a timpului dintre sosirile pachetelor
                self.packet_IAT_min = min(self.packet_IAT_max, IAT_actual)      #Valoarea minima a timpului dintre sosirile pachetelor              
                self.packets_length_max = max(self.packets_length_max, packet_length)       #Lungimea maxima a pachetelor
                self.packets_length_min = min(self.packets_length_min, packet_length)       #Lungimea minima a pachetelor            
                self.packets_per_second_max = max( self.packets_per_second_max , number_of_packets / IAT_actual )       #Valoarea maxima a pachetelor pe secunda
                self.packets_per_second_min = min( self.packets_per_second_min , number_of_packets / IAT_actual )       #Valoarea minima a pachetelor pe secunda          
                self.bytes_per_second_max = max( self.bytes_per_second_max , number_of_bytes / IAT_actual )     #Valoarea maxima a octeților pe secunda
                self.bytes_per_second_min = min( self.bytes_per_second_min , number_of_bytes / IAT_actual )     #Valoarea minima a octeților pe secunda
                
                #Actualizarea caracteristicilor statistice                
                self.bytes_per_second_mean = bytes /(curr_time_in_seconds - self.time_start)    #Numărul mediu de octeți pe secunda 
                self.packets_per_second_mean = packets /(curr_time_in_seconds - self.time_start)    #Numărul mediu de pachete pe secunda
                self.packets_length_mean = bytes/packets    #Lungimea medie a pachetelor
                self.packets_length_first_quartile = np.percentile(self.packet_vector, 25)      #Prima cuartilă a lungimii pachetelor
                self.packets_length_third_quartile = np.percentile(self.packet_vector, 75)      #A treia cuartilă a lungimii pachetelor
                self.packet_IAT_first_quartile = np.percentile(self.IAT_vector, 25)     #Prima cuartilă a timpului dintre soririle pachetelor
                self.packet_IAT_third_quartile = np.percentile(self.IAT_vector, 75)     #A treia cuartilă a timpului dintre sosirile pachetelor
                self.packets_length_ten_percentile = np.percentile(self.packet_vector, 10)      #Percentilă de 10 a lungimii pachetelor
                self.packets_length_ninety_percentile = np.percentile(self.packet_vector, 90)   #Percentilă de 90 a lungimii pachetelor
                self.packet_IAT_ten_percentile = np.percentile(self.IAT_vector, 10)     #Percentilă de 10 a timpului dintre sosirile pachetelor
                self.packet_IAT_ninety_percentile = np.percentile(self.IAT_vector, 90)      #Percentilă de 90 a timpului dintre sosirile pachetelor
                self.packet_IAT_mean = np.percentile(self.IAT_vector, 50)       #Timpul mediu intre sosirile pachetelor
                self.packets_per_second_variance = np.var(self.packet_per_second_vector)    #Varianta pachetelor pe secunda
                self.bytes_per_second_variance = np.var(self.bytes_per_second_vector)       #Varianta octeților pe secunda
                self.packets_length_variance = np.var(self.packet_vector)       #Varianta lungimii pachetelor
                self.packet_IAT_variance = np.var(self.IAT_vector)      #Varianta timpului dintre sosirile pachetelor
                self.last_time = curr_time_in_seconds       #Actualizarea ultimului timp
                
                self.vector_of_fft_components = np.fft.fft(self.IAT_vector, n = 11) #actualizarea caracteristicilor complexe
                
        if (number_of_bytes==0 or number_of_packets==0): #Daca nu se primisera octeti si pachete de la ultima actualizare aceasta comunicatie este considerata inactiva 
            self.status = 'INACTIVE'
        else:
            self.status = 'ACTIVE'

def printflows(traffic_type,f):
    #print('apel print \n')
    for key,flow in flows.items():
        #print ('key for flow is ', key,'and the flow status is: ', flow.status );
        if (flow.status == 'ACTIVE'): #daca respectiva comunicatie este activa o inseamna ca au fost actualizate caracteristicile deci o scriem in fisierul csv
            flow.status = 'INACTIVE'    #odata scrise in fisier informatiile consideram ca aceasca comunicatie nu mai este activa (evitam scrieri multiple)
            outstring = '\t'.join([
            str(flow.bytes_per_second_mean),
            str(flow.bytes_per_second_variance),
            str(flow.packets_per_second_mean),
            str(flow.packets_per_second_variance), 
            str(flow.packets_length_mean), 
            str(flow.packets_length_variance),
	        str(flow.packets_length_ten_percentile),
            str(flow.packets_length_first_quartile),
            str(flow.packets_length_ninety_percentile), 
            str(flow.packets_length_third_quartile), 
            str(flow.packet_IAT_mean),
            str(flow.packet_IAT_variance),
            str(flow.packet_IAT_ten_percentile),
            str(flow.packet_IAT_first_quartile),
            str(flow.packet_IAT_ninety_percentile),
            str(flow.packet_IAT_third_quartile),
            str(flow.bytes_per_second_min),
            str(flow.bytes_per_second_max),
            str(flow.packets_per_second_min),
            str(flow.packets_per_second_max),
            str(flow.packets_length_min),
            str(flow.packets_length_max),
            str(flow.packet_IAT_min),
            str(flow.packet_IAT_max),        
            str(flow.flow_duration),
            str(flow.flow_size_in_packets),   
            str(flow.flow_size_in_bytes),            
            str(flow.vector_of_fft_components[1]),            
            str(flow.vector_of_fft_components[2]),            
            str(flow.vector_of_fft_components[3]),            
            str(flow.vector_of_fft_components[4]),            
            str(flow.vector_of_fft_components[5]),            
            str(flow.vector_of_fft_components[6]),            
            str(flow.vector_of_fft_components[7]),            
            str(flow.vector_of_fft_components[8]),            
            str(flow.vector_of_fft_components[9]),            
            str(flow.vector_of_fft_components[10]),
            str(traffic_type)])
            f.write(outstring+'\n') #instructiunea de scriere 


def csv_filler(p,traffic_type=None,f=None):
    #print('apel csv_filler \n')
    #print('avem pidul procesului %s',os.getpgid(p.pid))
    while True:
        #print('avem pidul procesului %s',os.getpgid(p.pid))
        out = p.stdout.readline()   #citim iesirea scriptului de monitorizare
        #print('apel csv_filler while\n')
        if out == '' and p.poll() != None:      #daca nu avem iesire si procesul este oprit iesim din bucla
            break
        if out != '' and out.startswith(b'Flow'):   #daca iesirea scriptului de monitorizare incepe cu 'Flow' avem info din din scriptul modificat de monitorizare si pot fi folosite in actualizare
            #print('apel csv_filler while if \n', out)
            fields = out.split(b'\t')   #impartim in campuri iesirea scriptului de monitorizare 
            fields = [f.decode(encoding='utf-8', errors='strict') for f in fields]      #decodam informatia primita        
            unique_id = hash(''.join([fields[2],fields[4],fields[5],fields[3],fields[6]])) #folosim functia hash pentru generarea unui identificator unic 
            if unique_id in flows.keys():   #daca a mai fost vazut indicatorul actualizam caracteristicile comunicatiei
                #print('\nunique_id %s', unique_id)
                flows[unique_id].update_flow(int(fields[7]),int(fields[8]),int(fields[1]))
            else:   #altfel initializam un nou obiect de acest tip
                #print('\n unique_id ce nu era in flows %s', unique_id) 
                flows[unique_id] = Flow(int(fields[1]), fields[2], fields[3], fields[4], fields[5], fields[6], int(fields[7]), int(fields[8]))           
            printflows(traffic_type,f) #scriem in fisier informatiile actualizate

        

def alarm_handler(signum, frame):
    print("Finished collecting data.")
    raise Exception()        #daca a trecut timpul este ridicata o exceptie cu mesajul de finalizare a colectarii datelor

if __name__ == '__main__':
    if len(sys.argv) == 1: 
        print("Usage: sudo python prelucrare_date.py [traffic_type]") #afisam mesaj de ajutor daca scriptul nu este pornit cu un argument ce reprezinta numele traficului
    elif len(sys.argv) == 2:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  #deschidem un proces pentru monitor
            traffic_type = sys.argv[1]  # tipul de trafic este preluat ca argument al comenzii de pornire a scriptului
            f = open(traffic_type+'_training_data.csv', '+a')  #deschidem un fisier csv in care sunt scrise informatiile
            signal.signal(signal.SIGALRM, alarm_handler)        #initializam alarma
            signal.alarm(TIMEOUT)   #Pornim numaratoarea inversa
            #print('avem pidul procesului %s',os.getpgid(p.pid))
            try:
                #print('apel chestie initiala in try \n')
                headers = 'Bytes per second mean\tBytes per second variance\tPackets per second mean\tPackets per second variance\tPackets Length Mean\tPackets Length Variance\tPackets Length ten percentile\tPackets Length First Quartile\tPackets Length ninety percentile\tPackets Length Third Quartile\tPackets IAT mean\tPackets IAT variance\tPackets IAT ten percentile\tPackets IAT first quartile\tPackets IAT ninety percentile\tPackets IAT Third Quartile\tBytes per second min\tBytes per second max\tPackets per second min\tPackets per second max\tPackets Length min\tPackets Length max\tPackets IAT min\tPackets IAT max\tFlow Duration\tFlow Size in packets\tFlow Size in bytes\t1st Component of Fourier transform of IAT\t2nd Component of Fourier transform of IAT\t3rd Component of Fourier transform of IAT\t4th Component of Fourier transform of IAT\t5th Component of Fourier transform of IAT\t6th Component of Fourier transform of IAT\t7th Component of Fourier transform of IAT\t8th Component of Fourier transform of IAT\t9th Component of Fourier transform of IAT\t10th Component of Fourier transform of IAT\tTraffic Type\n'
                f.write(headers)    #scriem header-ele in fisierul 
                csv_filler(p,traffic_type=traffic_type,f=f) #apelam functia de populare a fisierului
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                print('Exiting', Exception)
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)    #orpim procesul monitorului
                f.close()   #inchidem fisierul csv
    else:
            print("ERROR: specify traffic type.\n") #afisam un mesaj de eroare daca scriptul este apelat cu prea multe argumente

        
    sys.exit();
