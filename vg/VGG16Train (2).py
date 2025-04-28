import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import optimizers, models, layers
import sys
from matplotlib import pyplot as plt

def make_model(): 
    model = models.Sequential()

    # Load cnv-only part of VGG16
    cnv_layers = VGG16(weights='imagenet', include_top=False,
     input_shape=(150,150,3))

    # Lock it down so we don't mess it up while training
    cnv_layers.trainable = False
    print(cnv_layers.summary())

    # Start with convolutional portion of VGG16
    model.add(cnv_layers)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # Change to 3 outputs for 3 classes with softmax activation
    model.add(layers.Dense(3, activation='softmax'))
    
    # Change to categorical_crossentropy for multi-class
    model.compile(loss='categorical_crossentropy',
     optimizer=optimizers.RMSprop(learning_rate=2e-5),
     metrics=['accuracy'])

    return model

def dump_generator(gen, num_batches = 1):
    for bnum, batch in zip(range(num_batches), gen):
       print("Batch {}".format(bnum), flush=True)
       for image, label in zip(batch[0], batch[1]):
           plt.imshow(image)
           plt.title(f"Label: {label}")
           plt.show()
    
def make_generators():
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Simple scaling for validation data
    noaug_datagen = ImageDataGenerator(rescale=1./255)  

    # Use data_simplified directories by default if no arguments provided
    train_dir = sys.argv[1] if len(sys.argv) > 1 else 'data_simplified2/train'
    val_dir = sys.argv[2] if len(sys.argv) > 2 else 'data_simplified2/validation'
    
    # Change to categorical for multi-class
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150), 
        batch_size=20,
        class_mode='categorical')       
    
    vld_generator = noaug_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

    # Show some sample images
    print("Validation data samples:")
    dump_generator(vld_generator)
    print("Training data samples:")
    dump_generator(train_generator)

    return (train_generator, vld_generator)
    
def main():
    model = make_model()
    train_generator, vld_generator = make_generators()
    
    hst = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=vld_generator,
        validation_steps=50).history
    
    # Save model to default path if no argument provided
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'material_model.h5'
    model.save(model_path)
    
    # Print training history
    for acc, loss, val_acc, val_loss in zip(
        hst['accuracy'], hst['loss'], hst['val_accuracy'], hst['val_loss']): 
        print("%.5f / %.5f  %.5f / %.5f" % (acc, loss, val_acc, val_loss))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python VGG16Train.py [train_dir] [validation_dir] [model_save_path]")
        print("Using default paths: data_simplified/train, data_simplified/validation, material_model.h5")
    
    main()