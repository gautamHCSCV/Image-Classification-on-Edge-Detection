
**Mobilenet V2**

1. Trained on 45 epochs
2. Sobel filter along y-direction
3. No gray-scaling of the images

</br>
Train accuracy: 0.8602
</br>
Test accuracy: 0.5820

</br></br>
**Custom Model**

inp = Input(shape = (150,150,3))</br>
gray = tf.image.rgb_to_grayscale(inp)</br>
sobel = tf.image.sobel_edges(gray)</br>
x = Conv2D(16,(3,3), activation = 'relu')(sobel[:, :, :, :, 0])</br>
x = MaxPool2D(2)(x)</br>
x = Conv2D(32,(3,3), activation = 'relu')(x)</br>
x = MaxPool2D(3)(x)</br>
x = Flatten()(x)</br>
out = Dense(100, activation = 'softmax')(x)
</br></br>
model = Model(inputs = inp, outputs = out)</br></br>

Train accuracy: 0.538</br>
Test accuracy: 0.3435
