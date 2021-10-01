import io
import socket
import struct
from tkinter import *
from PIL import Image, ImageTk
import datetime
import os
import tensorflow as tf
import numpy as np

modle = tf.keras.models.load_model('/home/louay/Desktop/ML/3_fing_v3.h5')
print('Model loaded successfully! ')

server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
print('server accepting connection')
connection = server_socket.accept()[0].makefile('rb')
print('connected to : ',connection)	


path = "/home/louay/Desktop/Raspberry/dump/"
img_dims=(640, 480)

root = Tk()
root.title("Slide Show")
root.resizable(0, 0)

class show:
    def __init__(self):
        self.files=[]
        self.record=False
        self.keep_running=True
        self.current_save_location=''
    def main(self):
        if self.keep_running:
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            image_stream = io.BytesIO()
            image_stream.write(connection.read(image_len))
            image_stream.seek(0)
            image = Image.open(image_stream) 
            if self.record:self.files.append(image)
            next_image=ImageTk.PhotoImage( image )
            slide_image.configure( image=next_image ) 
            slide_image.image=next_image
            pred=np.argmax(modle.predict(np.expand_dims(  np.asarray( image.resize((150,150)) )/255   , axis=0))  ,  axis=1)
            prediction.set('PREDICTION: {}'.format(pred[0]))
            slide_image.after(20,self.main) 
            
            
        elif self.record: #to make sure we dont shutoff while saving
            server_socket.close()
            connection.close()
                
    def start_recording(self):
        temp_foler=datetime.datetime.now().strftime("%H:%M:%S")
        print('recording started! Frames will be saved in {}'.format(temp_foler))
        os.mkdir(path+temp_foler)
        self.current_save_location=path+temp_foler+'/'
        self.record=True
    
    def Stop_recording(self):
        self.keep_running=False
        print('recording stoped ... Number of recorded frames: {} taking a small break from straming...'.format(len(self.files)))
        for img_index in range(len(self.files)):self.files[img_index].save(self.current_save_location+str(img_index), format='PNG')
        self.files=[]
        print('Done!')
        self.keep_running=True
        self.record=False
        
       
    def switch_recording_state(self):
        if self.record:self.Stop_recording()
        else:self.start_recording()
    
    def stop_show(self):
        self.keep_running=False

        
    
PLACE_HOLDER = ImageTk.PhotoImage( Image.open(path+'PLACE_HOLDER'))
slide_image = Label(root , image=PLACE_HOLDER)


slide_show=show()


btn1 = Button(root, text="Start", bg='black', fg='gold', command=slide_show.main,font=("Tahoma Regular",11))
btn1.pack(side=LEFT)

btn2 = Button(root, text="start/stop recording", bg='black', fg='gold', command= slide_show.switch_recording_state ,font=("Tahoma Regular",11))
btn2.pack(side=LEFT)

btn3 = Button(root, text="Shut Down", bg='black', fg='gold', command=slide_show.stop_show,font=("Tahoma Regular",11))
btn3.pack(side=LEFT)

prediction = StringVar()
text_filed=Label(root, textvariable=prediction, font=("Tahoma Regular", 15) )
text_filed.pack(side=LEFT)

slide_image.pack()
root.mainloop()


