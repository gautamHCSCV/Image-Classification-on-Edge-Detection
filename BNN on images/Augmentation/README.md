**Model:** </br>
![image](https://user-images.githubusercontent.com/65457437/144980895-e7c44c97-b1ca-4586-8739-e2d9328465da.png)

 </br> </br>
**Augmentation** </br>
train_datagen = ImageDataGenerator( </br>
        rescale=1./255, rotation_range=60, width_shift_range=2.0, height_shift_range=2.0, </br>
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

 </br> </br>
**RESULTS** </br>
Train accuracy = 36.14% </br>
Test accuracy = 35.68%  </br>

![image](https://user-images.githubusercontent.com/65457437/144980821-7c1ecac8-9668-4fc5-89a6-129fa5e46113.png)

![image](https://user-images.githubusercontent.com/65457437/144980822-672be03c-646c-479e-ae7a-875b2646d9ee.png)
