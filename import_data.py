import serial
import serial.tools.list_ports
import io
import os,sys,stat
import time
import csv
serialport=str(serial.tools.list_ports.comports()[0])[:-10]

os.chmod(serialport, stat.S_IRWXO  )


ser = serial.Serial(
    port=serialport,\
    baudrate=9600)

print("connected to: " + ser.portstr)
for no_of_readings in range(10):
    with open('file.csv','a')as csvfile:
        data=csv.writer(csvfile)
        for no_of_lines_per_reading in range(8):
            getData=str(ser.readline())
            data=getData[2:][:-5]
            data=data.split(',')
            datafile.writerow(data)
        
    ser.close()
    time.sleep(7)