from flask import Flask, render_template, request, Response, jsonify,redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import imutils
import time

app = Flask(__name__)



""""
NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32
"""
diz = {"curl":0, "squat":0, "flessioni":0, "plank":0}
action = None
reps = 0
msg = ""
time_diff = None

METS_MAN = 4.5
METS_WOMAN = 3.5


@app.template_filter()
def firstHalfExercises(diz):
    half_length = (len(diz) // 2)
    return dict(list(diz.items())[0:half_length])


@app.template_filter()
def secondHalfExercises(diz):
    half_length = (len(diz) // 2)
    return dict(list(diz.items())[half_length:])


colors = [(245,117,16), (117,245,16), (16,117,245),(45,110,160)]
actions = np.array(['curl', 'squat','flessioni','null'])

@app.route('/get_data')
def get_data():
    return jsonify(diz,action,reps,msg,time_diff)


@app.route('/', methods=['GET', 'POST'])
def home():
    form_type = request.form.get('form_type')
    if form_type == 'modal':
        #form della modale
        email = request.form.get('email')
        age = request.form.get('age')
        height = request.form.get('height')
        weight = request.form.get('weight')
        gender = request.form.get('gender')

        return redirect(url_for('gym', email=email, age=age, height=height, weight=weight,gender=gender))
    elif form_type == 'footer': 
        #form del footer
        email = request.form.get('email')
        age = request.form.get('age')
        height = request.form.get('height')
        weight = request.form.get('weight')
        gender = request.form.get('gender')
        
        return redirect(url_for('gym', email=email, age=age, height=height, weight=weight,gender=gender))

    return render_template('home.html')
    


@app.route('/gym/',methods=['GET', 'POST'])
def gym():
    gender = request.args.get('gender')
    email = request.args.get('email')
    age = request.args.get('age')
    height = request.args.get('height')
    weight = request.args.get('weight')

    if request.method == 'POST':
        #return redirect("/")
        return redirect(url_for('stats', email=email, age=age, height=height, weight=weight,gender=gender))
    return render_template('gym.html', email=email, age=age, height=height, weight=weight, diz=diz,gender=gender)


@app.route('/gym/stats/',methods=['GET', 'POST'])
def stats():
    email = request.args.get('email')
    age = request.args.get('age')
    height = request.args.get('height')
    weight = request.args.get('weight')
    gender = request.args.get('gender')
    kcal = 0.000
    for x in diz.keys():
        kcal += compute_kcal(gender,weight,x)

    return render_template('stats.html', email=email,age=age, height=height, weight=weight, gender=gender,kcal=kcal,diz=diz)


"""Return = risposta HTTP contenente un flusso di dati video in formato multipart/x-mixed-replace --> flusso di dati che verrà
aggiornato costantemente sostituendo ogni immagine precedente.
"""
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_pose = mp.solutions.pose
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 

     



def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    try:
        #print(len(results.pose_landmarks.landmark))
        if float(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].visibility) <= 0.001:
            #print("piede non visibile")
            pass
        else:
            #print("piede visibile ",results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]) 
            #print("corretto ",results.pose_landmarks.landmark)
            pass
    except:
        #print("errore ", results.pose_landmarks)
        pass
    return np.concatenate([pose])

def check_visibility(results):
    try:
        for res in results.pose_landmarks.landmark:
            if float(res.visibility) <= 0.3:
                return False
    except:
        return False
    return True


def classify_speed(time,action=None):
    if action == "curl":
        if time >= 0.8 and time < 1.4: 
            return "Perfect"
        elif time > 1.5:
            return "Good"
        elif time < 0.8:
            return "Too fast.."
    if action == "squat":
        if time >= 0.8 and time < 1.4: 
            return "Perfect"
        elif time > 1.5:
            return "Good"
        elif time < 0.8:
            return "Too fast.."
    if action == "flessioni":
        if time >= 0.8 and time < 1.4: 
            return "Perfect"
        elif time > 1.5:
            return "Good"
        elif time < 0.8:
            return "Too fast.."
    if action == "abs":
        pass
    



model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#LOAD WEIGHTS
model.load_weights('action.h5')

def calculate_angle(a,b,c): # teh tree points to create angles
    a = np.array(a) # We convert all to numpy arrays
    b = np.array(b)
    c = np.array(c)
    radiants = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle= np.abs(radiants*180.0/np.pi)
    if angle > 180:# convert angle between 0 and 180    
        angle = 360 - angle
    return angle 

""" METS X 3.5 X BW (kg) / 200 = KCAL/MIN. """
def compute_kcal(sex,kg,exercize):
    if sex == "male":
        MET = METS_MAN
    elif sex == "other":
        MET = METS_MAN
    else:
        MET = METS_WOMAN
    kcal_1m = (MET * 3.5 * float(kg)) / 200 # equivalent to 25 exercise 
     ##25 : kcal_1m = 36 : y
    return round((kcal_1m * diz[exercize])/25,1)
   

    
def compute_curl(landmarks) : 
    mp_pose = mp.solutions.pose
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y ]
    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y ]
    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y ]
    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y ]
    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y ]
    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y ]
    left_angle = calculate_angle(l_shoulder,l_elbow,l_wrist)
    right_angle  = calculate_angle(r_shoulder,r_elbow,r_wrist)
    return left_angle, right_angle


def compute_squat(landmarks):
    mp_pose = mp.solutions.pose
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y ]
    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y ]
    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y ]
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y ]
    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y ]
    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y ]
    right_angle = calculate_angle(r_hip,r_knee,r_ankle)
    left_angle = calculate_angle(l_hip,l_knee,l_ankle)
    return right_angle,left_angle



def gen_frames():
    global reps
    global action
    global msg
    global time_diff
    last_down = {'curl':None, 'squat':None, 'flessioni':None, 'plank':None }
    
    sequence = []
    actions_state = ["null"]
    counter = 0
    stage = None
    threshold = 0.95
 
    show_webcam_only = True
    start_time = time.time()
    countdown = 10
 
    cap = cv2.VideoCapture(0)

    # Set mediapipe model. 'with' allows to access our holistic model
    #min_detection_condifence = our initial detection. So how mediapipe holistic works? --> makes an initial detection and then fromt here it just track the keypoints
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            #frame = imutils.resize(frame, width=1280) #capire se in pc diversi è uguale la risoluzione.
            
            # during countdown: only webcam shown (no detection)
            if show_webcam_only and time.time() - start_time < countdown:
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
            else:
                show_webcam_only = False
            
            image, results = mediapipe_detection(frame, holistic)

            keypoints = extract_keypoints(results)
            
            #print("keypoint", keypoints)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if check_visibility(results):
                #msg = "Ok"
                if len(sequence) == 30  :
                
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        action = actions[np.argmax(res)]
                        if action != 'null':
                            #diz[action] += 1
                            actions_state.append(action)
                            if action != actions_state[-2]:
                                counter = 0
                                stage = None
                                
                        try:
                            if results.pose_landmarks:
                                landmarks = results.pose_landmarks.landmark
                                l_curlangle, r_curlangle = compute_curl(landmarks)
                                r_squatangle, l_squatangle = compute_squat(landmarks)
                                draw_styled_landmarks(image, results)
                                if action == 'curl':
                                    if l_curlangle > 160 or r_curlangle > 160:
                                        stage = "down"
                                        if not last_down['curl']:
                                            last_down['curl'] = time.time()
                                            
                                    if (l_curlangle < 40 or r_curlangle < 40) and stage == "down":
                                        stage = "up"
                                        if last_down['curl'] is not None:
                                            time_diff = round(time.time() - last_down['curl'],4)
                                            msg = classify_speed(time_diff, 'curl')
                                            last_down['curl'] = None
                                        counter += 1
                                        diz[action] += 1
                                if action == 'squat':
                                    if (r_squatangle <= 110 or l_squatangle <= 110):
                                        stage = "down"
                                        if not last_down['squat']:
                                            last_down['squat'] = time.time()
                                    if r_squatangle > 155 and l_squatangle > 155 and stage == "down":
                                        stage = "up"
                                        if last_down['squat'] is not None:
                                            time_diff = round(time.time() - last_down['squat'],4)
                                            msg = classify_speed(time_diff, 'squat')
                                            last_down['squat'] = None
                                        counter += 1
                                        diz[action] += 1
                                    
                                if action == 'flessioni':
                                    if (l_curlangle < 90 or r_curlangle < 90) :
                                        stage = "down"
                                        if not last_down['flessioni']:
                                            last_down['flessioni'] = time.time()
                                    if l_curlangle > 150 or r_curlangle > 150 and stage == "down":
                                        stage = "up"
                                        if last_down['flessioni'] is not None:
                                            time_diff = round(time.time() - last_down['flessioni'],4)
                                            msg = classify_speed(time_diff, 'flessioni')
                                            last_down['flessioni'] = None
                                        counter += 1
                                        diz[action] += 1
                                    
                                    listOfStrings = actions_state[-10:]
                                    if len(listOfStrings) > 0:
                                        if all(elem == 'flessioni' for elem in listOfStrings) and counter == 0:
                                            if reps == 0:
                                                clock = time.time()
                                            action = "plank"
                                            reps += 0.2
                                            diz[action] += int(reps)
                                    
                        except Exception as e:
                            pass

                    if action != 'plank':
                        reps = counter

            else:
                msg = "⚠️Please move away from the camera a bit"


            #conversione: trasmissione in formato compatibile con il protocollo di libreria Flask-SocketIO
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            #generatore di ogni singolo frame del video in formato jpeg : visualizzare lo stream video
            yield (b'--frame\r\n' 
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    

        


if __name__ == '__main__':
    app.run(debug=True)
