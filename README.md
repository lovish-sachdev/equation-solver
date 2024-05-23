
# Equation Solver 

This project detects and recognize a simple hand written algebraic equation from image and then solves it .

## Demo

![demo](https://github.com/lovish-sachdev/equation-solver/blob/main/media/Screenshot%202024-05-23%20174542.png?raw=true)

![demo](https://github.com/lovish-sachdev/equation-solver/blob/main/media/Screenshot%202024-05-23%20174419.png?raw=true)

## Working 

The whole program works in 3 steps

1. Firstly a yolo model was trained on a custom synthetic data made by combining images from another custom image generator program ,the image generator generate images of shape 96,384 containing 1 single equation . then some of these images were vertically tacked with randomness and was treated as a dataset for line/equation detection

2. Then these equations were cropped out from original image and passed into an another model which was c-rnn that is convolutional-recurrent neaural network. for recognizing/extracting text from equation

3. Then a small function calculates the value of equation if equation is valid 

## Data Generation

* The whole data was generated synthetically 

* firstly a dataset like mnist was chosen ,then a number of images were transformed randomly and stacked vertically in random backgroung and brightness to generate images for resembling equations

* then a number of these equation-image were stacked vertically to get multiple equation in single page
