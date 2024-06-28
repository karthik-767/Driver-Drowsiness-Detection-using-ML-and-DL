from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import json
import http.client
import math
import warnings
import winsound
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8)
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
iris_right = [473, 474, 475, 476, 477]
iris_left = [468, 469, 470, 471, 472]
states = ['alert', 'drowsy']
model_path = 'D:\\DDD SYS\\FLASK-RANDOM_FOREST_ME_FINAL_1\\SOURCE_FILES\\training\\models\\created_dataset_1.pkl'

model = None
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Global variables to track calibration status
calibration_in_progress = False
inference_in_progress = False
stop_flag=0
@app.route('/')
def index():
    return render_template('index.html')
@socketio.on('stop_application')
def handle_stop_application():
    global stop_flag
    stop_flag=1
def send_sms():
    conn = http.client.HTTPSConnection("j3m1gn.api.infobip.com")
    payload = json.dumps({
        "messages": [
            {
                "from": "447860099299",
                "to": "918374176841",
                "messageId": "a5c76428-d6e2-480c-8efa-9051b6421d38",
                "content": {
                    "templateName": "message_test",
                    "templateData": {
                        "body": {
                            "placeholders": ["Your driver is drowsy 🔴 🔴 🔴 !!! "]
                        }
                    },
                    "language": "en"
                }
            }
        ]
    })
    headers = {
        'Authorization': 'App ce070e9b55a26a0a82953771d757fb09-1ebf5f2d-1817-489f-af3c-91a6a172e56d',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    conn.request("POST", "/whatsapp/1/message/template", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))

def distance(p1, p2):
    p1 = p1[:2]
    p2 = p2[:2]
    return (((p1 - p2)**2).sum())**0.5

def eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2

def run_face_mp(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z])
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]
        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image

def calibrate(calib_frame_count=25):
    global calibration_in_progress
    calibration_in_progress = True
    ears = []
    mars = []
    pucs = []
    moes = []

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        ear, mar, puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)
            
            cv2.putText(image, "EAR: %.2f" %(ears[-1]), (int(0.02*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "MAR: %.2f" %(mars[-1]), (int(0.27*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "PUC: %.2f" %(pucs[-1]), (int(0.52*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "MOE: %.2f" %(moes[-1]), (int(0.77*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(image, "State: Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # cv2.imshow('MediaPipe FaceMesh', image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        emit('video_frame', {'image': base64.b64encode(frame).decode('utf-8')})
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if len(ears) >= calib_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)
    calibration_in_progress = False
    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
        [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]

def get_classification(input_data, model):    
    model_input = np.array(input_data)
    preds = model.predict(model_input)
    return int(preds.sum() >= 13)

def infer(ears_norm, mars_norm, pucs_norm, moes_norm, model):
    global inference_in_progress
    global stop_flag
    inference_in_progress = True
    ear_main = 0
    mar_main = 0
    puc_main = 0
    moe_main = 0
    decay = 0.9

    label = None

    input_data = []
    frame_before_run = 0

    cap = cv2.VideoCapture(0)
    c = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        ear, mar, puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ear = (ear - ears_norm[0])/ears_norm[1]
            mar = (mar - mars_norm[0])/mars_norm[1]
            puc = (puc - pucs_norm[0])/pucs_norm[1]
            moe = (moe - moes_norm[0])/moes_norm[1]
            if ear_main == -1000:
                ear_main = ear
                mar_main = mar
                puc_main = puc
                moe_main = moe
            else:
                ear_main = ear_main*decay + (1-decay)*ear
                mar_main = mar_main*decay + (1-decay)*mar
                puc_main = puc_main*decay + (1-decay)*puc
                moe_main = moe_main*decay + (1-decay)*moe
        else:
            ear_main = -1000
            mar_main = -1000
            puc_main = -1000
            moe_main = -1000
        
        if len(input_data) == 20:
            input_data.pop(0)
        input_data.append([ear_main, mar_main, puc_main, moe_main])

        frame_before_run += 1
        if frame_before_run >= 15 and len(input_data) == 20:
            frame_before_run = 0
            label = get_classification(input_data, model)
            print('got label ', label)
        
        cv2.putText(image, "EAR: %.2f" %(ear_main), (int(0.02*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "MAR: %.2f" %(mar_main), (int(0.27*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "PUC: %.2f" %(puc_main), (int(0.52*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "MOE: %.2f" %(moe_main), (int(0.77*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if label is not None:
            cv2.putText(image, "State %s" %(states[label]), (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if label == 0:
            c = 0
        if label == 1:
            c += 1
            print(c)
            if c == 400:
                send_sms()
                frequency = 1000  
                duration = 5000
                winsound.Beep(frequency, duration)
                c = 0
                break

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        emit('video_frame', {'image': base64.b64encode(frame).decode('utf-8'), 'label': label})

        if stop_flag==1:
            stop_flag=0
            break
    
    cv2.destroyAllWindows()
    cap.release()
    inference_in_progress = False

@socketio.on('start_calibration')
def handle_start_calibration():
    global calibration_in_progress, inference_in_progress
    if not inference_in_progress:
        ears_norm, mars_norm, pucs_norm, moes_norm = calibrate()
        emit('calibration_complete', {
            'ears_norm': ears_norm,
            'mars_norm': mars_norm,
            'pucs_norm': pucs_norm,
            'moes_norm': moes_norm
        })

@socketio.on('start_inference')
def handle_start_inference(data):
    global calibration_in_progress, inference_in_progress
    if not calibration_in_progress:
        ears_norm = data['ears_norm']
        mars_norm = data['mars_norm']
        pucs_norm = data['pucs_norm']
        moes_norm = data['moes_norm']
        infer(ears_norm, mars_norm, pucs_norm, moes_norm, model)        
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000)
