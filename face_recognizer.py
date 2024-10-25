import cv2
import face_recognition_models
import face_recognition
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font

class FaceRecognitionApp:
    def __init__(self, master):
        # Initialize the main window
        self.master = master
        self.master.title("Face Recognition")

        self.custom_font = font.Font(family="Helvetica", size=9)

        # Initialize lists for image paths, encodings, and names
        self.image_paths = []  # List to hold paths of multiple images
        self.known_face_encodings = []
        self.known_face_names = []

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image, font=self.custom_font)
        self.load_button.pack(pady=10)

        self.start_button = tk.Button(master, text="Start Recognition", command=self.start_recognition, state=tk.DISABLED, font=self.custom_font)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Stop Recognition", command=self.stop_recognition, state=tk.DISABLED, font=self.custom_font)
        self.stop_button.pack(pady=10)

        self.video_capture = None  # Var to hold video capture object
        self.running = False  # Flag to control the recognition loop

    def load_image(self):
        # Load an image and store its encoding
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if image_path:
            example_image = cv2.imread(image_path)
            if example_image is None:
                messagebox.showerror("Error", "Could not load the image. Please check the file exists and is in a valid format.")
                return

            example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
            example_encoding = face_recognition.face_encodings(example_image)[0]

            # Add the encoding and the corresponding name to the lists
            self.known_face_encodings.append(example_encoding)
            self.known_face_names.append(image_path.split('/')[-1])  # Use the file name as the person's name

            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Image loaded successfully.")

    def start_recognition(self):
        # Start the video capture and recognition process
        if self.known_face_encodings:  # Ensure there are images loaded
            self.running = True
            self.video_capture = cv2.VideoCapture(0)  # Open the webcam
            self.run_recognition()

    def run_recognition(self):
        # Run the face recognition process in a loop
        if not self.running:
            return

        ret, frame = self.video_capture.read()  # Capture a frame from the camera
        if not ret:
            print("Could not capture the frame from the camera.")
            self.stop_recognition()  # Stop recognition if frame capture fails
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations in the camera frame
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare detected face encodings with known encodings
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "No Match"  # Default name if no match found

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]  # Use the corresponding name

                # Draw a rectangle around the face and display the name
                cv2.rectangle(frame, (left, top), (right, bottom), (131, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (87, 46, 175), 2)

        cv2.imshow('Video', frame)  # Show the video feed

        # Schedule the next execution
        self.master.after(10, self.run_recognition)

    def stop_recognition(self):
        # Stop the recognition process
        self.running = False
        if self.video_capture:
            self.video_capture.release()  # Release the video capture
            cv2.destroyAllWindows()  # Close all OpenCV windows
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)

# Create the Tkinter application
root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()