# ==================================================================
# IMPORTING NECESSARY LIBRARIES
# ===================================================================

import cv2
import os
import pickle
import face_recognition
from collections import Counter

# ========================================================================
# Opencv library is used to capture and detect the faces 
# face_recognition library is used to recognize and match the faces
# then labeled the knows faces with the name the unknown faces as unknown 
# ========================================================================


DATASET_DIR = "dataset"
ENCODING_FILE = "encodings.pkl"

os.makedirs(DATASET_DIR, exist_ok=True)


# ============================================================
# REGISTER NEW PERSON
# ============================================================

def register_person():

    name = input("Enter person's name: ").strip()

    person_path = os.path.join(DATASET_DIR, name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    count = 0

    print("\nPress 's' to save image")
    print("Press 'q' to quit\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Register Person", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):

            file_path = os.path.join(
                person_path,
                f"{count}.jpg"
            )

            cv2.imwrite(file_path, frame)

            count += 1

            print(f"Saved Image {count}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Registration Completed")


# ========================================================
# TRAIN FACES
# ========================================================

def train_faces():

    known_encodings = []
    known_names = []

    print("Training Started...\n")

    for person_name in os.listdir(DATASET_DIR):

        person_folder = os.path.join(
            DATASET_DIR,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):

            image_path = os.path.join(
                person_folder,
                image_name
            )

            image = face_recognition.load_image_file(
                image_path
            )

            encodings = face_recognition.face_encodings(
                image
            )

            if len(encodings) > 0:

                known_encodings.append(
                    encodings[0]
                )

                known_names.append(
                    person_name
                )

                print(
                    f"Processed: {person_name}/{image_name}"
                )

    data = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open(ENCODING_FILE, "wb") as file:
        pickle.dump(data, file)

    print("\nTraining Completed Successfully")


# ========================================================
# FACE RECOGNITION
# ========================================================

def recognize_faces():

    if not os.path.exists(ENCODING_FILE):
        print("Please train faces first.")
        return

    with open(ENCODING_FILE, "rb") as file:
        data = pickle.load(file)

    known_encodings = data["encodings"]
    known_names = data["names"]

    face_cascade = cv2.CascadeClassifier("C:/Users/KUSUM/OneDrive/Desktop/Syntecxhub_Projects/AirtifiacialIntelligence_project/haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    print("Press Q to Quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:

            top = y
            right = x + w
            bottom = y + h
            left = x

            encodings = face_recognition.face_encodings(
                rgb,
                [(top, right, bottom, left)]
            )

            name = "Unknown"

            if len(encodings) > 0:

                encoding = encodings[0]

                matches = face_recognition.compare_faces(
                    known_encodings,
                    encoding,
                    tolerance=0.5
                )

                if True in matches:

                    matched_names = [
                        known_names[i]
                        for i, match in enumerate(matches)
                        if match
                    ]

                    name = Counter(
                        matched_names
                    ).most_common(1)[0][0]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Face Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ========================================================
# MAIN MENU
# ===========================================================

while True:

    print("\n===== FACE RECOGNITION =====")
    print("1. Register New Person")
    print("2. Train Faces")
    print("3. Start Face Recognition")
    print("4. Exit")

    choice = input("Enter Choice: ")

    if choice == "1":
        register_person()

    elif choice == "2":
        train_faces()

    elif choice == "3":
        recognize_faces()

    elif choice == "4":
        break

    else:
        print("Invalid Choice")











