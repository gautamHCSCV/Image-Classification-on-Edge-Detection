import tensorflow as tf
import warnings
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense,Conv2D,Flatten, Input,MaxPool2D,GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

path = 'img/'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        path + 'TRAIN',
        target_size=(96, 96),
        batch_size=32)
validation_generator = test_datagen.flow_from_directory(
        path + 'TEST',
        target_size=(96, 96),
        batch_size=32)

class Binarize(tf.keras.constraints.Constraint):
    def __init__(self, ref_value = 1):
            self.ref_value = ref_value

    def __call__(self, w):
            return tf.math.sign(w)*self.ref_value

    def get_config(self):
            return {'ref_value': self.ref_value}


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME, kernel_constraint=Binarize(), bias_constraint=Binarize())
        self.bn_1 = BatchNormalization(beta_constraint=Binarize(), gamma_constraint=Binarize())
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME, kernel_constraint=Binarize(), bias_constraint=Binarize())
        self.bn_2 = BatchNormalization(beta_constraint=Binarize(), gamma_constraint=Binarize())
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same", kernel_constraint=Binarize(), bias_constraint=Binarize())
            self.res_bn = BatchNormalization(beta_constraint=Binarize(), gamma_constraint=Binarize())

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out

class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal", kernel_constraint=Binarize(), bias_constraint=Binarize())
        self.init_bn = BatchNormalization(beta_constraint=Binarize(), gamma_constraint=Binarize())
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.res_5_1 = ResnetBlock(512, down_sample=True)
        self.res_5_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2, self.res_5_1, self.res_5_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

model = ResNet18(10)
model.build(input_shape = (None,96,96,1))
model.compile(optimizer = "adam",loss='categorical_crossentropy', metrics=["accuracy"])

inp = Input(shape = (96,96,3))
gray = tf.image.rgb_to_grayscale(inp)
sobel = tf.image.sobel_edges(gray)
out = model(tf.sqrt(tf.square(sobel[:,:,:,:,1])+tf.square(sobel[:,:,:,:,0])))

model = Model(inputs = inp, outputs = out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(3*1e-2, 30)
callback = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)
early_stop = EarlyStopping(monitor='val_accuracy', patience = 25)

his = model.fit(train_generator, epochs=250, validation_data=validation_generator, callbacks = [callback, early_stop], verbose = 1)

epochs = range(1,len(his.history['accuracy'])+1)
plt.plot(epochs,his.history['accuracy'], label = 'Train')
plt.plot(epochs, his.history['val_accuracy'], label = 'Validation')
plt.title('Accuracy variations')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc_stl_sobel.png')
plt.show()


import matplotlib.pyplot as plt
plt.plot(epochs,his.history['loss'], label = 'Train')
plt.plot(epochs, his.history['val_loss'], label = 'Validation')
plt.title('Loss Variations')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_stl_sobel.png')
plt.show()