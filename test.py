#RAKEZ MAYA NTY HETHA RUNNIH MARA BARK BAD YETSAJELEK MODELE F MODELCNN.h5 w temchy LTESTINGIMAGE.PY HEKA FYH CAMERA YE5O MODELE W Y5DOM ALIH
from tensorflow import keras
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from matplotlib import pyplot as plt
# LA PARTIE APPRENTISSAGE ET ENTRENNEMENT DU MODELE LENNNA MODELE T3ALLEM W AWKA NBRE D EPOCHS 100 YTAWL AMA YATY MODELE TA7FOUN W TEJEM TSA8RO HASSYLO
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
train_data_dir="bd/American"

model= keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu",strides=(1,1),padding='valid'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(32,(3,3),activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10,activation='softmax'))

mcustom_optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=mcustom_optimizer, metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,      # Normalisation des valeurs de pixel (mise à l'échelle)
    rotation_range=30,      # Rotation aléatoire de l'image de -40 à 40 degrés
    width_shift_range=0.3,  # Déplacement horizontal aléatoire de l'image
    height_shift_range=0.3, # Déplacement vertical aléatoire de l'image
    zoom_range=0.3,         # Zoom aléatoire de l'image
    horizontal_flip=True,   # Retournement horizontal aléatoire de l'image
    fill_mode='nearest'     # Mode de remplissage des pixels après les transformations
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),   # Redimensionnement des images à la taille spécifiée
    batch_size=128,           # Taille du batch d'entraînement
    class_mode='categorical' # Mode de classification catégorielle
)
# HETHY TE5OULK AHSEN EPOCHS W TESTA7FA4 ALIH BCH KEN LPROFA SA2LETEK
model_checkpoint = ModelCheckpoint("modelCNN.h5", verbose=1, save_best_only=True, save_weights_only=False,save_freq=1,mode='auto',monitor="accuracy")



history = model.fit_generator(
    train_generator,  # Générateur de données d'entraînement
    steps_per_epoch=len(train_generator),  # Nombre d'étapes par époque (nombre total de lots)
    epochs=100,  # Nombre d'époques
    callbacks=[model_checkpoint]# Nombre d'étapes de validation (nombre total de lots de validation)
)

# PARTIE CAMERA AWKI F TESTINGIMAGE W ENA MANRFHECH T5DOM WALA AMA NRMLMEME?NT CV 5TR MANCDYCH CAMERA
# HEKI MAIN 5ODMA 9DIMA 7ATYTHA W BARRAA