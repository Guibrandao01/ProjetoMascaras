from IPython.display import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd





#iniciando rede neural convolucional CNN
 

# Criando os objetos train_datagen e validation_datagen 
#com as regras de pré-processamento das imagens

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator( rescale = 1. / 255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               width_shift_range=0.2,
                               height_shift_range=0.2, 
                               fill_mode='nearest'                      
                                 )

    
validation_datagen = ImageDataGenerator( rescale = 1. / 255 ,                                        
                                      )
                                        


#pre processamento de dados e validaçao
batch_size = 64

training_set = train_datagen.flow_from_directory('treino', target_size = (49,49),
                                                 batch_size = batch_size,
                                                 class_mode='binary')

Nclasses = 1






validation_set = validation_datagen.flow_from_directory('validacao', target_size = (49,49),
                                                        batch_size = batch_size , 
                                                        class_mode = 'binary' )




def criandomodelo():
    model=Sequential()
       
        #primeira camada de convoluçao
        #convertendo imagens para 64 pixel em um array 3D pois as imagens sao coloridas
        #com 3 camadas de cores

    model.add(Conv2D(32,(3,3),input_shape=(49,49,3) , activation='relu'))
    #aplicando agrupamento pooling para reduzir o tamanho do mapa
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3),input_shape=(49,49,3) , activation='relu'))
    #aplicando agrupamento pooling para reduzir o tamanho do mapa
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    
    



    #aplicando achatamento = flatten para converter estruturas de dados 2D resultado
    #da camada anterior em uma estrutura 1D

    model.add(Flatten())
    
        #conectando as camadas usando a funçao de ativaçao retificadore "relu"
        #e depois uma funçao de sigmóide para obter a probabilidade de cada imagem conter
        # mascara ou sem mascara
        
        #full connection
    model.add(Dense(units=256,activation='relu'))
  
    model.add(Dropout(0.25))
    
  
    
    model.add(Dense(units = Nclasses,activation ='sigmoid'))  
    model.add(Dropout(0.45))
   
   
    

        #compilando rede usando o algoritmo rmsprop , e a funçao log loss com a binary_crossentropy
    
    
    model.compile(optimizer='rmsprop' , loss='binary_crossentropy',metrics=['accuracy'])
            #rede neural construida
    model.summary()
    
    

    
    
    
    
    return model  





modelo = criandomodelo()



#executando o treinamento (processo pode levar tempo , dependendo do seu pc)
history = modelo.fit_generator(training_set,
                         steps_per_epoch = 50,      
                         epochs = 20,
                         validation_data=validation_set,
                          validation_steps= 25
                         )





modelo.save_weights('modelo.h5')


#fazendo as previsoes

from keras.preprocessing import image
test_image = image.load_img('teste/with_mask652.jpeg', target_size=
                          (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = modelo.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'sem mascara'
else:
    prediction = 'com mascara'
    
Image(filename='teste/with_mask652.jpeg')
        

prediction 


def graficoresultados(history):
    
    #gera grafico de resultados do treino
    mpl.rc("font",**{"size":17})
    fig,axes = plt.subplots(1,2, figsize = (17,14))
    
    #loss
    
    axes[0].plot(range(1, len(history.history["loss"])+1), history.history["loss"], 
          label="Train Loss", color="royalblue", lw=3)
    
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")
   # axes[0].set_xticks(range(1,len(history.history["accuracy"])+1))
    #axes[0].legend()
    
    #accuracy
    
    axes[1].plot(range(1,len(history.history["accuracy"])+1),
                 history.history["accuracy"],
                 label ="Train Accuracy",
                 color ="royalblue",
                 lw=3)
    
     
  
    
    axes[1].plot(range(1,len(history.history["val_accuracy"])+1),
                 history.history["val_accuracy"],
                 label="Test Accuracy",
                 color="forestgreen",
                 lw=3)
    
    
    
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
   # axes[1].set_xticks(range(1,len(history.history["accuracy"])+1))
    axes[1].legend()
    plt.show()
    graficoresultados(history)
    
    
    
    
    
    

    

