from flask import Flask, render_template, Response
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame
from twilio.rest import Client

app = Flask(__name__)

# Your Twilio Account SID and Auth Token
account_sid = "ACb682831688bf559dafe96b8b6fae8eb3"
auth_token = "a802fdf41d59b0aa778795310f902e9b"

# Create a Twilio client
client = Client(account_sid, auth_token)

# Replace 'your_phone_number' with the phone number you want to send SMS alerts to
to_phone_number = "+918010410993"

# Initialize pygame mixer for sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('sound.wav')  # Replace 'sound.wav' with your desired sound file

def play_alert_sound():
    alert_sound.play()

    # Send an SMS alert
    message = client.messages.create(
        body="Alert: Drowsiness detected!",
        from_="+15172331315",  # Replace with your Twilio phone number
        to=to_phone_number
    )
    print("SMS sent:", message.sid)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0  # Initialize flag as a global variable
alert_active = False
alert_start_time = None  # Initialize alert_start_time to None

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global flag, alert_active, alert_start_time  # Reference the global variables

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            if ear < thresh:
                flag += 1
                if flag >= frame_check and not alert_active:
                    play_alert_sound()
                    alert_active = True

                # Draw blue outlines around eyes
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            else:
                flag = 0
                if alert_active:
                    alert_start_time = cv2.getTickCount() / cv2.getTickFrequency()
                    alert_active = False

            if alert_start_time is not None:
                elapsed_time = (cv2.getTickCount() / cv2.getTickFrequency()) - alert_start_time
                if elapsed_time > 10:  # Play sound for 10 seconds
                    alert_sound.stop()
                    alert_start_time = None

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)