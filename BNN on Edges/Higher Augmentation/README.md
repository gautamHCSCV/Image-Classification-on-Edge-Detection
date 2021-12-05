Used ResNet-18 with more residual blocks</br>

![image](https://user-images.githubusercontent.com/65457437/144732223-b2b6d6c7-d0a6-46c9-82eb-a1e23db52f31.png)
</br>

Training Images Augmentation:</br?
train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=60, width_shift_range=2.0, height_shift_range=2.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


</br></br>
1. Used Early Stopping : EarlyStopping(monitor='val_accuracy', patience = 30)
2. Used Learning rate scheduler : CosineDecay(3*1e-2, 30)
3. Model trained for 80 epochs (provided 250)

**Results**</br>
Train Accuracy = 26.60% </br>
Test Accuracy = 26.54% </br>

![image](https://user-images.githubusercontent.com/65457437/144738060-3a805d61-128c-4898-a624-8dafccdd0290.png)

![image](https://user-images.githubusercontent.com/65457437/144738061-08219e24-21e6-4bd1-8ad5-30feb23008c7.png)
