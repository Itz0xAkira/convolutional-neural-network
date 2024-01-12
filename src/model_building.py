# model_building.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold


EPOCHS = 1


def create_resnet152V2_model():
    name = "ResNet152V2"
    base_model = ResNet152V2(include_top=False, input_shape=(256, 256, 3), weights='imagenet')
    base_model.trainable = False

    return Sequential([
        base_model,
        GAP(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ], name=name)

def compile_model(model):
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

def setup_callbacks(model_name):
    return [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(model_name + ".h5", save_best_only=True)
    ]

def get_image_data_generators(train_fold, val_fold):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_images = train_gen.flow_from_dataframe(
        train_fold,
        x_col='Filepath',
        y_col='Label',
        target_size=(256, 256),
        batch_size=32,
        class_mode='sparse'
    )

    val_images = train_gen.flow_from_dataframe(
        val_fold,
        x_col='Filepath',
        y_col='Label',
        target_size=(256, 256),
        batch_size=32,
        class_mode='sparse'
    )

    return train_images, val_images

def train_model(model, train_images, val_images, callbacks):
    return model.fit(
        train_images,
        validation_data=val_images,
        epochs=EPOCHS,
        callbacks=callbacks
    )

def evaluate_model(model, test_images):
    return model.evaluate(test_images, verbose=0)
