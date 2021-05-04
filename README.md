# Distracted Driver Detection
Here I tried to take on the [State Farm Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview) challenge on Kaggle from 2015. The training dataset had a total of 22424 images divided into the following 10 classes: safe driving, texting - right, texting - left, talking on the phone - right, talking on the phone - left, operating the radio, drinking, reaching behind, hair and makeup and talking to passenger. I used the Module API in Pytorch to create a CNN inspired by the architecture of VGG16, with the only differenece being the number of channels output by each convolutional layer (which were reduced to speed up computation).

I first trained the model for a single epoch on different learning rates and chose the one that made the loss go down significantly over the first couple hundred iterations. I then did the same thing with various different values for weight decay. I trained the model for just 10 epochs becuase I saw the loss go down and accuracy on the validation set go up very quickly. 

However, the model didn't end up scoring as well on the test set in Kaggle as it did on the validation set during training. I believe a reason for this could be the size of the training set that was provided. The model was trained on 17939 images and used a validation set of 4485 images. The model did not seem to be overfitting based on the loss and accuracy plots, however, it is possible that the validation data was very similar to the training data. This would also explain why the loss dropped so quickly and why the accuracy on the validations set reached 97% in just one epoch. 

## Future Considerations
- Use data augmnetation to create more training examples and make the model to more robust to minor transformations
- Use more regularization methods to combat overfitting on the training data (ex. Cutout or Mixup work well for small datasets)
- Try using a smaller and simpler model, maybe the depth of the model and # of parameters were limiting generalization
- Experiement with some preprocessing on the images to help the CNN identify features more easily
