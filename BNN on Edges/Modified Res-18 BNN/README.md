Used ResNet-18 with more residual blocks</br>

![image](https://user-images.githubusercontent.com/65457437/144732223-b2b6d6c7-d0a6-46c9-82eb-a1e23db52f31.png)
</br></br>
1. Used Early Stopping : EarlyStopping(monitor='val_accuracy', patience = 25)
2. Used Learning rate scheduler : CosineDecay(3*1e-2, 30)
3. Model trained for 74 epochs (provided 250)

**Results**</br>
Train Accuracy = 30.04% </br>
Test Accuracy = 28.61% </br>

![image](https://user-images.githubusercontent.com/65457437/144732292-6867ea1f-ae2b-4d11-b42d-5acf141171e7.png)

![image](https://user-images.githubusercontent.com/65457437/144732293-8ac0882a-f9cc-4ea7-bf5b-34d5d485bd95.png)
