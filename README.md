
# AI Gym Tracker
...


## Introduction


The AI Gym Tracker project represents an innovative web application that harnesses the power of artificial intelligence to track and monitor users’ physical activities during their workout sessions. Its main goal is to provide an intuitive and user-friendly user interface that allows users to perform their physical workouts easily, obtaining detailed information about their performance in real time.
The primary goal of the application is to provide users with an interactive and highly motivating training environment. Using cutting-edge technologies such as artificial intelligence, computer vision for video analysis, and pose recognition, the application is able to count
exercise repetitions in real time, while also providing precise and accurate feedback on the speed of execution of poses.
To perform action prediction and process the stream of frames from the webcam, we developed and trained an accurate artificial intelligence
model based on recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) neural networks. We created this model through a
data collection phase during which we recorded a considerable number of videos of correctly performed exercises, thus creating an accurate custom training dataset. In this way through the use of the powerful Mediapipe library, we extracted the poses performed by the user in real
time, determining the joints as a list of skeletal key points of the body

![Alt text](static/parkTraining.png)


## WebApp 
The webApp of AIGymTracker is minimal and moder at the same time.
It guides users to start the training and provides powerful tool to execute exercises.


### Homepage 
The homepage is the platform's landing page, featuring a concise HTML layout. It provides a brief overview of how the portal works and essential information, such as the need for ample space for movements.
![Alt text](static/home.png)


### Form 

A key feature of our page is an interactive form designed for users to effortlessly initiate their personalized sessions. By inputting essential details such as email address, age, height, weight, and gender, users kickstart a journey tailored just for them. It's worth emphasizing that the email address is a crucial piece of information, ensuring seamless association with each training session.
Equally vital are the inputs for weight and gender. These specifics play a pivotal role in calculating the calories burned during the workout, offering users valuable insights into their fitness progress. To guarantee accuracy, each field in the form undergoes rigorous validation, minimizing errors and ensuring the consistency of user data.

![Alt text](static/form.png)

### Exercise page
Upon correct form submission, users are directed to an exercise page where a 10-second countdown, visible on screen with audio cues, precedes the start of the training session. The webcam displays the user's body landmarks. A top bar provides key exercise information, including the last exercise, repetitions, and a timer. Exercise details in cards on the sides dynamically update via a Python backend. A warning alerts users if they are too close to the webcam. Feedback on exercise execution speed (perfect, good, too fast) is provided, and users can independently conclude the workout by clicking a button.

![Alt text](static/Allenamento.png)

### Statistics

After the session, users are redirected to a summary page displaying workout details, including exercise repetitions and burned calories. We provide transparency by showing the calorie calculation formula. Users can opt to receive a workout report via email and choose to repeat the session or return to the homepage.

![Alt text](static/statistics.png)


## Project Structure
This Flask project has the following structure:

- `static` folder: this folder contains various files such as audio and static that are used in the web application.

- `templates` folder: this folder contains the HTML files for all the pages of the web application. These files are used to render the web pages on the client side.

- `action.h5` file: this binary file contains the weights of the neural network, network configurations, training information and other necessary information to reconstruct the pre-trained deep learning model. This file is loaded into app.py file to use the pre-trained model.

- `README` file: this file contains a description of the project, its features, and how to use it.

- `requirements.txt` file: this file contains all the packages that need to be installed in the virtual environment for the Flask application to run.

- `app.py` file: contains all the Python code for the Flask web application.


## Setup and run the App
To setup and run AI Gym Tracker application, follow these steps:

1) Clone or download the source code

2) Create a new virtual environment for the application using the following command in your terminal:
	- On Windows: `python -m venv venv`
	- On macOS and Linux: `python3 -m venv venv`

3) Activate the virtual environment using the following command:
	- On Windows: `venv\Scripts\activate`
	- On macOS and Linux: `source venv/bin/activate`

4) Install the required packages of the application using the following command: `pip install -r requirements.txt`

5) Once all the packages have been installed successfully, you can start AI Gym Tracker application by running the following command: `python app.py`

6) After running the above command, the application will be running in the localhost at port 5000. Open your web browser and navigate to http://localhost:5000/ to see the application running.

7) Remember to deactivate the virtual environment once you're done by running the following command: `deactivate`. This will exit the virtual environment and return you to your system's default Python environment. Have fun 😄.


## Authors
 
- [Alberto Cotumaccio](https://github.com/albertoCotumaccio)
- [Alessandro Di Patria](https://github.com/AlessandroDiPatria)

