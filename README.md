# NutriScanner
For our SwampHacks IX submission, we wanted to create a product that promoted healthy eating habits. What better way to do that than know what nutrients are in your food, no matter what you're eatiing? NutriScanner allows you to upload pictures of your food and get a close estimate of the macros(calories, carbohydrates, protein, etc.) in your meal. It makes macro counting that much easier!

## Tools we used
This website was made built with raw JavaScript, HTML, and CSS for the frontend and a Python Flask backend to manage file uploads and interfacing with the convolutional neural network. We preprocessed over 20,000 images of food from across the internet with NumPy and built our deep learning model with TensorFlow and Keras.

## Challenges
Where do we begin? We all had very little web development experience and no experience with Flask, so getting a website up and running in the first place was already difficult. Put that on top of integrating a neural network into the backend. It taught us a lot about implimenting machine learning algorithms into websites, and how to connect Flask with a frontend. All in all, we learned a lot about adapting to new technologies quickly(all within 36 hours to be exact!).

## Improvements for the Future
In the future, we will be coming back to this project to deploy it and set up a permanent database to process more user images. In addition, we hope to optimize the CNN through careful examination of each layer, a larger dataset, and experimentation with different optimization functions. Lastly, we'll be adding more features to help the user track their eating, such as a personalized daily macro tracker and food recommendations considering known healthy patterns and your preferences.

## Try it yourself!
After cloning the repo, make sure to have all of the libraries and modules in requirements.txt using:
```
pip install -r requirements.txt
```
Afterwards, navigate to the WebServer directory and type:
```
run flask
```

## Meet the Team!
Anthony Yao - https://www.linkedin.com/in/anthonyjyao/ <br>
Evan Hadam - https://www.linkedin.com/in/evan-hadam/ <br>
Andrew Tang - https://www.linkedin.com/in/andrtang/ <br>
James Hu - https://www.linkedin.com/in/hu-james/
