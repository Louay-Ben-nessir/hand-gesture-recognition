import socket
import struct
from PIL import Image
import tensorflow as tf
import numpy as np
from pynput.mouse import Button, Controller
import io

mouse = Controller()

modle = tf.keras.models.load_model('/home/louay/Desktop/ML/3_fing_v3.h5')
print('Model loaded successfully! ')

server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

print('server accepting connection')
connection = server_socket.accept()[0].makefile('rb')
print('connected to : ',connection)	

    
controle=8
clicked= False
try:
    while True:
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream) 
        pred = np.argmax(np.round(modle.predict(np.expand_dims(  np.asarray( image.resize((150,150)) )/255   , axis=0))),axis=1) 
        controle = pred if pred!=0 else  controle# if 0 continue 
        
        if controle==8: pass
        elif controle==6:
            mouse.press(Button.right)
            mouse.release(Button.right)
        elif controle==1:
            mouse.press(Button.left)
            mouse.release(Button.left)
        elif controle==2:mouse.move(0, 10)
        elif controle==3:mouse.move(-10, 0)
        elif controle==4:mouse.move(0, -10)
        elif controle==5:mouse.move(10, 0)
        elif controle==7:
            mouse.release(Button.right) if clicked else mouse.press(Button.right)
            clicked= not clicked
            
            
        
        # i could have done this in many diffrent better ways.... im pretty sure this is one of the worst way tp dp ot
finally:
    connection.close()
    server_socket.close()
'''
8 halt
6click left
1 click right
2 go up
3 go left 
4 go down
5 go right 
7 hold and 7 again release 
0 contiunue 
'''
