**Model:** </br>
1. Early Stopping: EarlyStopping(monitor='val_accuracy', patience = 30)
2. Learning rate deday: CosineDecay(3*1e-2, 30)
3. Input in grayscale: tf.sqrt(tf.square(sobel[:,:,:,:,1])+tf.square(sobel[:,:,:,:,0]))

 </br> </br>
**Augmentation** </br>
train_datagen = ImageDataGenerator( </br>
        rescale=1./255, rotation_range=60, width_shift_range=2.0, height_shift_range=2.0, </br>
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

 </br> </br>
**RESULTS** </br>
Train accuracy = 30.30% </br>
Test accuracy = 29.24%  </br>


![image](https://user-images.githubusercontent.com/65457437/144982054-f0c869a7-a246-4b3d-902d-920f5e1e5517.png)

