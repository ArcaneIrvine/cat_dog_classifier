# Cat or Dog Classifier
Deep Neural Network Classifier that recognizes cat or dog images. This model is easy to use or even tweak it so it classifies different objects of your choice instead of cats or dogs. All you have to do is simply change the data from cats and dogs to objects of your desire!

## How it works
- Grabbed image data off the web (used a tool to massively download images from a single page of cats/dogs)
- Cleared the data by removing too small or faulty images 
- Loaded the data
- Scaled the data
- Split the data
- Built the NN model
- Trained model on the data
- Plot performance
- Test the model
- Saved the model

## What you need to run it
A requirements.txt file has been made that includes all the libraries needed but here is a more general list
- Python 3.9
- Tensorflow
- OpenCV
- Matplotlib
- Numpy

## How to tweak the model work for your desired data
- Get some image data between some objects you would want to differentiate
- If you want to change the project's folder setup you will also need to change the directories in the code accordingly
- Otherwise throw your images on the 'Data' folder as it is and remove the 'cat' and 'dog' ones
- (reccomended to manually remove images smaller than 10kb)
- Run the script and wait for the training to finish and you now have the model of your desire saved in the 'models' directory

### Enjoy!
* for the cam_identify.py script you need to hit 'p' after you have an image in front of your webcam ready to get identified and 'q' to close the webcam. 
