---
title: Chapter 12 - Human-Robot Interaction
sidebar_position: 3
---

# Chapter 12: Human-Robot Interaction

## Learning Goals

- Design intuitive human-robot interfaces
- Understand social robotics principles
- Learn collaborative robotics concepts
- Implement natural language interfaces
- Create gesture recognition systems
- Design collaborative robot behaviors
- Integrate speech recognition APIs

## Introduction to Human-Robot Interaction

Human-Robot Interaction (HRI) is a multidisciplinary field that focuses on the design, development, and evaluation of robots that can interact with humans in a natural, safe, and effective manner. As robots become more prevalent in our daily lives, from industrial settings to homes and hospitals, the ability to interact seamlessly with humans becomes increasingly important.

### Key Principles of HRI

1. **Safety**: Ensuring human safety during interaction
2. **Intuitiveness**: Making robot behavior predictable and understandable
3. **Social Acceptance**: Designing robots that are socially acceptable
4. **Collaboration**: Enabling effective human-robot teamwork
5. **Adaptability**: Adjusting to different users and contexts

### HRI Challenges

- **Communication Barriers**: Different modalities and languages
- **Trust Issues**: Building trust between humans and robots
- **Social Norms**: Understanding and respecting social conventions
- **Cultural Differences**: Adapting to cultural contexts
- **Ethical Considerations**: Privacy, autonomy, and fairness

## Natural Language Processing for HRI

### Speech Recognition

Speech recognition enables robots to understand spoken commands from humans:

```python
import speech_recognition as sr
import pyttsx3
import threading
import time
import queue


class SpeechRecognitionSystem:
    def __init__(self, language='en-US'):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()

        # Configuration
        self.language = language
        self.energy_threshold = 300  # Minimum audio energy to consider for recording
        self.dynamic_energy_threshold = True

        # Callback functions
        self.command_callbacks = []

        # Thread-safe command queue
        self.command_queue = queue.Queue()

    def listen_continuously(self):
        """Continuously listen for speech commands"""
        def callback(recognizer, audio):
            try:
                # Recognize speech using Google's speech recognition
                text = recognizer.recognize_google(audio, language=self.language)

                # Add recognized text to queue
                self.command_queue.put(text)

                # Process command
                self.process_command(text)

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

        # Start listening in background
        stop_listening = self.recognizer.listen_in_background(self.microphone, callback)

        return stop_listening

    def process_command(self, text):
        """Process recognized command"""
        print(f"Recognized command: {text}")

        # Trigger callbacks
        for callback in self.command_callbacks:
            callback(text)

    def add_command_callback(self, callback):
        """Add callback function for command processing"""
        self.command_callbacks.append(callback)

    def speak(self, text):
        """Speak text using text-to-speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def recognize_from_audio_file(self, audio_file_path):
        """Recognize speech from an audio file"""
        with sr.AudioFile(audio_file_path) as source:
            audio = self.recognizer.record(source)

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"


# Example usage
def main():
    # Initialize speech recognition system
    speech_system = SpeechRecognitionSystem()

    # Define command callback
    def command_handler(text):
        """Handle recognized commands"""
        text_lower = text.lower()

        if "hello" in text_lower or "hi" in text_lower:
            speech_system.speak("Hello! How can I help you?")
        elif "move forward" in text_lower:
            speech_system.speak("Moving forward")
            # In a real system, this would trigger robot movement
        elif "stop" in text_lower:
            speech_system.speak("Stopping")
            # In a real system, this would stop robot
        elif "what is your name" in text_lower:
            speech_system.speak("I am a social robot designed to assist you")
        else:
            speech_system.speak(f"I heard: {text}")

    # Add command handler
    speech_system.add_command_callback(command_handler)

    print("Starting speech recognition system...")
    print("Say 'hello', 'move forward', 'stop', or 'what is your name'")

    # Start continuous listening
    stop_listening = speech_system.listen_continuously()

    try:
        # Keep the program running
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_listening(wait_for_stop=False)


if __name__ == '__main__':
    main()
```

### Natural Language Understanding

Beyond speech recognition, we need to understand the meaning and intent behind human language:

```python
import re
import spacy
import numpy as np
from typing import Dict, List, Tuple, Optional


class NaturalLanguageUnderstanding:
    def __init__(self):
        """Initialize NLU system"""
        try:
            # Load spaCy English model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            # Fallback to simple keyword matching
            self.nlp = None

        # Define command patterns
        self.command_patterns = {
            'navigation': [
                r'go to (.+)',
                r'move to (.+)',
                r'navigate to (.+)',
                r'go (.+)',
                r'move (.+)'
            ],
            'action': [
                r'pick up (.+)',
                r'grab (.+)',
                r'lift (.+)',
                r'put (.+) (?:down|there)',
                r'place (.+) (?:down|there)'
            ],
            'information': [
                r'what is (.+)',
                r'tell me about (.+)',
                r'explain (.+)',
                r'how do i (.+)'
            ],
            'social': [
                r'hello',
                r'hi',
                r'good morning',
                r'good afternoon',
                r'good evening'
            ]
        }

        # Define location keywords
        self.locations = {
            'kitchen': ['kitchen', 'cooking area', 'food area'],
            'living_room': ['living room', 'lounge', 'sitting area'],
            'bedroom': ['bedroom', 'sleeping area', 'bed area'],
            'office': ['office', 'study', 'work area'],
            'bathroom': ['bathroom', 'restroom', 'washroom'],
            'dining_room': ['dining room', 'dining area', 'eating area']
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            'persons': [],
            'places': [],
            'objects': [],
            'actions': []
        }

        if self.nlp:
            doc = self.nlp(text)

            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities['persons'].append(ent.text)
                elif ent.label_ in ["GPE", "LOC", "FAC"]:
                    entities['places'].append(ent.text)
                elif ent.label_ in ["OBJECT", "PRODUCT"]:
                    entities['objects'].append(ent.text)

        # Fallback to keyword matching if spaCy is not available
        else:
            # Simple keyword matching for places
            text_lower = text.lower()
            for loc_type, keywords in self.locations.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        entities['places'].append(keyword)

        return entities

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the given text"""
        text_lower = text.lower()
        best_match = ('unknown', 0.0)

        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                # Simple keyword matching
                if re.search(pattern.replace('(.+)', ''), text_lower):
                    confidence = 0.8  # High confidence for exact matches
                    if confidence > best_match[1]:
                        best_match = (intent, confidence)

                # Pattern matching with capture groups
                match = re.search(pattern, text_lower)
                if match:
                    confidence = 0.9  # High confidence for pattern matches
                    if confidence > best_match[1]:
                        best_match = (intent, confidence)

        # If no pattern matches, use simple keyword classification
        if best_match[0] == 'unknown':
            if any(word in text_lower for word in ['hello', 'hi', 'hey']):
                best_match = ('social', 0.7)
            elif any(word in text_lower for word in ['go', 'move', 'navigate', 'walk']):
                best_match = ('navigation', 0.6)
            elif any(word in text_lower for word in ['pick', 'grab', 'lift', 'put', 'place']):
                best_match = ('action', 0.6)
            elif any(word in text_lower for word in ['what', 'tell', 'explain', 'how']):
                best_match = ('information', 0.6)

        return best_match

    def parse_command(self, text: str) -> Dict[str, any]:
        """Parse a command into structured format"""
        intent, confidence = self.classify_intent(text)
        entities = self.extract_entities(text)

        # Extract target from text based on intent
        target = None
        location = None

        if intent == 'navigation':
            # Extract location using regex
            for pattern in self.command_patterns['navigation']:
                match = re.search(pattern, text.lower())
                if match:
                    target = match.group(1)
                    break
        elif intent in ['action', 'information']:
            # Extract object using regex
            for pattern in self.command_patterns.get(intent, []):
                match = re.search(pattern, text.lower())
                if match:
                    target = match.group(1)
                    break

        # Check for location keywords in entities
        for place in entities.get('places', []):
            location = place
            break  # Take first location found

        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'target': target,
            'location': location,
            'original_text': text
        }

    def generate_response(self, parsed_command: Dict[str, any]) -> str:
        """Generate appropriate response based on parsed command"""
        intent = parsed_command['intent']

        if intent == 'social':
            return "Hello! How can I assist you today?"
        elif intent == 'navigation':
            if parsed_command['target']:
                return f"Okay, I will navigate to the {parsed_command['target']}."
            else:
                return "Where would you like me to go?"
        elif intent == 'action':
            if parsed_command['target']:
                return f"Okay, I will {parsed_command['original_text'].split()[0]} the {parsed_command['target']}."
            else:
                return "What would you like me to do?"
        elif intent == 'information':
            if parsed_command['target']:
                return f"I can provide information about {parsed_command['target']}. What specifically would you like to know?"
            else:
                return "What would you like to know?"
        else:
            return "I'm not sure I understood. Could you please rephrase that?"


# Example usage
def main():
    nlu = NaturalLanguageUnderstanding()

    # Test commands
    test_commands = [
        "Hello robot",
        "Go to the kitchen",
        "Move to the living room",
        "Pick up the red cup",
        "What is the weather like?",
        "Tell me about the meeting",
        "Navigate to the office"
    ]

    print("Testing Natural Language Understanding system:")
    print("=" * 50)

    for command in test_commands:
        parsed = nlu.parse_command(command)
        response = nlu.generate_response(parsed)

        print(f"Input: {command}")
        print(f"Parsed: Intent='{parsed['intent']}', Confidence={parsed['confidence']:.2f}, Target='{parsed['target']}', Location='{parsed['location']}'")
        print(f"Response: {response}")
        print("-" * 30)


if __name__ == '__main__':
    main()
```

## Gesture Recognition

### Computer Vision-Based Gesture Recognition

Gesture recognition allows robots to interpret human gestures as commands:

```python
import cv2
import numpy as np
import mediapipe as mp
from enum import Enum


class GestureType(Enum):
    UNKNOWN = 0
    THUMB_UP = 1
    THUMB_DOWN = 2
    PEACE = 3
    ROCK_ON = 4
    STOP = 5
    GO = 6
    WAVE = 7


class GestureRecognizer:
    def __init__(self):
        """Initialize gesture recognition system"""
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Define gesture templates
        self.gesture_templates = {
            GestureType.THUMB_UP: self._is_thumb_up,
            GestureType.THUMB_DOWN: self._is_thumb_down,
            GestureType.PEACE: self._is_peace,
            GestureType.ROCK_ON: self._is_rock_on,
            GestureType.STOP: self._is_stop,
            GestureType.GO: self._is_go,
            GestureType.WAVE: self._is_wave
        }

    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        # Convert to numpy arrays
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _is_thumb_up(self, hand_landmarks):
        """Check if thumb is up gesture"""
        # Thumb tip should be above thumb joint
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_INTERMEDIATE]

        # Index finger should be bent (tip below PIP)
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]

        # Thumb should be extended up while other fingers are curled
        return (thumb_tip.y < thumb_ip.y and  # Thumb up
                index_tip.y > index_pip.y and  # Index finger down
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y >
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y >
                hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y and
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y >
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y)

    def _is_peace(self, hand_landmarks):
        """Check if peace sign gesture"""
        # Index and middle fingers extended, others curled
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        return (index_tip.y < index_pip.y and  # Index finger up
                middle_tip.y < middle_pip.y and  # Middle finger up
                ring_tip.y > ring_pip.y and  # Ring finger down
                pinky_tip.y > pinky_pip.y)  # Pinky down

    def _is_stop(self, hand_landmarks):
        """Check if stop gesture (palm facing forward)"""
        # All fingers extended and palm facing camera
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]

        # Check if palm is facing forward (wrist to pinky mcp should be more horizontal than vertical)
        return abs(wrist.x - pinky_mcp.x) > abs(wrist.y - pinky_mcp.y)

    def _is_go(self, hand_landmarks):
        """Check if go gesture (pointing)"""
        # Index finger extended, others curled
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

        return (index_tip.y < index_pip.y and  # Index finger up
                middle_tip.y > middle_pip.y)  # Middle finger down

    def _is_thumb_down(self, hand_landmarks):
        """Check if thumb down gesture"""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_INTERMEDIATE]

        # Index finger should be bent (tip below PIP)
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]

        # Thumb should be extended down while other fingers are curled
        return (thumb_tip.y > thumb_ip.y and  # Thumb down
                index_tip.y > index_pip.y and  # Index finger down
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y >
                hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y >
                hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y and
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y >
                hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y)

    def _is_rock_on(self, hand_landmarks):
        """Check if rock-on gesture (index and pinky extended)"""
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        return (index_tip.y < index_pip.y and  # Index finger up
                middle_tip.y > middle_pip.y and  # Middle finger down
                ring_tip.y > ring_pip.y and  # Ring finger down
                pinky_tip.y < pinky_pip.y)  # Pinky up

    def _is_wave(self, hand_landmarks):
        """Check if waving gesture"""
        # This is a simplified check - in practice, you'd track movement over time
        # For now, we'll consider an open hand as potentially waving
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # If hand is relatively open and moving (would need temporal tracking for true wave)
        return True  # Simplified - real implementation would track movement

    def recognize_gesture(self, image):
        """Recognize gesture from image"""
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.hands.process(image_rgb)

        recognized_gestures = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Try to match each gesture type
                for gesture_type, gesture_func in self.gesture_templates.items():
                    if gesture_func(hand_landmarks):
                        recognized_gestures.append({
                            'gesture': gesture_type,
                            'landmarks': hand_landmarks,
                            'confidence': 0.9  # For now, assume high confidence
                        })

                        # Draw landmarks on image
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

        return recognized_gestures, image

    def process_video_stream(self, camera_index=0):
        """Process video stream for gesture recognition"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("Gesture recognition started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Recognize gestures
            gestures, annotated_frame = self.recognize_gesture(frame)

            # Display recognized gestures
            for gesture in gestures:
                cv2.putText(annotated_frame,
                           f"Gesture: {gesture['gesture'].name}",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           (0, 255, 0),
                           2)

            # Show frame
            cv2.imshow('Gesture Recognition', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage
def main():
    gesture_recognizer = GestureRecognizer()

    # For demonstration, we'll show how to use it with an image
    # In practice, you'd use the process_video_stream method

    print("Gesture recognition system initialized.")
    print("Available gestures:", [g.name for g in GestureType if g != GestureType.UNKNOWN])

    # To test with video stream, uncomment the following:
    # gesture_recognizer.process_video_stream()


if __name__ == '__main__':
    main()
```

## Facial Expression Recognition

Facial expression recognition enables robots to understand human emotions:

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os


class FacialExpressionRecognizer:
    def __init__(self, model_path=None):
        """Initialize facial expression recognition system"""
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Define emotion labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Load emotion classification model if provided
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            # For this example, we'll create a simple classifier
            # In practice, you'd load a pre-trained model
            self.model = None
            print("Warning: No pre-trained model provided. Using rule-based classification.")

    def preprocess_face(self, face_image):
        """Preprocess face image for emotion classification"""
        # Resize to model input size (assuming 48x48 for example)
        face_resized = cv2.resize(face_image, (48, 48))

        # Convert to grayscale if needed
        if len(face_resized.shape) == 3:
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_resized

        # Normalize pixel values
        face_normalized = face_gray / 255.0

        # Reshape for model input (add batch and channel dimensions)
        face_reshaped = face_normalized.reshape(1, 48, 48, 1)

        return face_reshaped

    def classify_expression_rule_based(self, face_image):
        """Rule-based expression classification (simplified)"""
        # This is a very simplified approach
        # In practice, use a deep learning model

        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # Simple heuristics for emotion detection
        # These are very basic and not accurate - for demonstration only

        # Calculate some basic features
        height, width = gray.shape
        eye_region = gray[int(height*0.3):int(height*0.5), :]
        mouth_region = gray[int(height*0.6):int(height*0.8), :]

        # Simple metrics
        eye_brightness = np.mean(eye_region)
        mouth_brightness = np.mean(mouth_region)

        # Very basic classification based on simple features
        if mouth_brightness > 150:  # Bright mouth area (possibly smile)
            return 'happy', 0.6
        elif eye_brightness < 100:  # Dark eyes (possibly sad/angry)
            return 'sad', 0.5
        else:
            return 'neutral', 0.4

    def recognize_expressions(self, image):
        """Recognize facial expressions in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Classify expression
            if self.model:
                # Use deep learning model
                processed_face = self.preprocess_face(face_roi)
                predictions = self.model.predict(processed_face)
                emotion_idx = np.argmax(predictions[0])
                emotion = self.emotions[emotion_idx]
                confidence = float(predictions[0][emotion_idx])
            else:
                # Use rule-based classification
                emotion, confidence = self.classify_expression_rule_based(face_roi)

            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Label with emotion
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'bbox': (x, y, w, h)
            })

        return results, image

    def process_video_stream(self, camera_index=0):
        """Process video stream for facial expression recognition"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("Facial expression recognition started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Recognize expressions
            results, annotated_frame = self.recognize_expressions(frame)

            # Show frame
            cv2.imshow('Facial Expression Recognition', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage
def main():
    expression_recognizer = FacialExpressionRecognizer()

    print("Facial expression recognition system initialized.")
    print("Available emotions:", expression_recognizer.emotions)

    # To test with video stream, uncomment the following:
    # expression_recognizer.process_video_stream()


if __name__ == '__main__':
    main()
```

## Social Robotics Principles

### Theory of Mind for Robots

Theory of Mind is the ability to attribute mental states to oneself and others:

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import json


@dataclass
class MentalState:
    beliefs: Dict[str, any]
    desires: Dict[str, any]
    intentions: List[str]
    emotions: Dict[str, float]  # Emotion name to intensity (0.0 to 1.0)


class TheoryOfMindSystem:
    def __init__(self):
        """Initialize Theory of Mind system for the robot"""
        self.robot_mental_state = MentalState(
            beliefs={},
            desires={},
            intentions=[],
            emotions={'calm': 1.0}  # Start calm
        )

        self.human_mental_models = {}  # Track mental models of different humans
        self.interaction_history = []  # Track past interactions

    def update_robot_beliefs(self, belief_key: str, belief_value: any):
        """Update robot's beliefs about the world"""
        self.robot_mental_state.beliefs[belief_key] = belief_value

    def update_robot_emotions(self, emotion: str, intensity: float):
        """Update robot's emotional state"""
        self.robot_mental_state.emotions[emotion] = max(0.0, min(1.0, intensity))

        # Normalize emotions so they sum to 1.0 (simple approach)
        total = sum(self.robot_mental_state.emotions.values())
        if total > 0:
            for emo in self.robot_mental_state.emotions:
                self.robot_mental_state.emotions[emo] /= total

    def create_human_mental_model(self, human_id: str):
        """Create a mental model for a new human"""
        self.human_mental_models[human_id] = MentalState(
            beliefs={},
            desires={},
            intentions=[],
            emotions={}
        )

    def update_human_belief(self, human_id: str, belief_key: str, belief_value: any):
        """Update our model of what a human believes"""
        if human_id not in self.human_mental_models:
            self.create_human_mental_model(human_id)

        self.human_mental_models[human_id].beliefs[belief_key] = belief_value

    def update_human_desire(self, human_id: str, desire_key: str, desire_value: any):
        """Update our model of what a human desires"""
        if human_id not in self.human_mental_models:
            self.create_human_mental_model(human_id)

        self.human_mental_models[human_id].desires[desire_key] = desire_value

    def infer_human_intention(self, human_id: str, observed_action: str) -> Optional[str]:
        """Infer human intention based on observed action"""
        if human_id not in self.human_mental_models:
            return None

        # Simple inference based on action patterns
        # In a real system, this would use more sophisticated models
        action_intents = {
            'reaching': 'grasp_object',
            'pointing': 'request_attention',
            'waving': 'greet',
            'frowning': 'express_displeasure',
            'smiling': 'express_pleasure'
        }

        for pattern, intent in action_intents.items():
            if pattern in observed_action.lower():
                self.human_mental_models[human_id].intentions.append(intent)
                return intent

        return None

    def predict_human_response(self, human_id: str, robot_action: str) -> str:
        """Predict how a human might respond to robot's action"""
        if human_id not in self.human_mental_models:
            return "neutral_response"

        human_model = self.human_mental_models[human_id]

        # Simple prediction based on human's desires and beliefs
        if 'help' in robot_action.lower():
            if 'need_assistance' in human_model.desires.values():
                return "positive_response"
            else:
                return "neutral_response"
        elif 'personal_space' in robot_action.lower():
            if any(emotion in human_model.emotions for emotion in ['comfortable', 'relaxed']):
                return "accepting_response"
            else:
                return "avoiding_response"
        else:
            return "neutral_response"

    def adjust_behavior(self, human_id: str, interaction_result: str):
        """Adjust robot behavior based on interaction outcome"""
        if human_id not in self.human_mental_models:
            return

        # Update robot's beliefs based on interaction result
        if interaction_result == "positive":
            self.update_robot_emotions('happy', 0.8)
            self.update_robot_beliefs(f'{human_id}_responsive', True)
        elif interaction_result == "negative":
            self.update_robot_emotions('concerned', 0.7)
            self.update_robot_beliefs(f'{human_id}_needs_space', True)
        else:  # neutral
            self.update_robot_emotions('calm', 0.9)
            self.update_robot_beliefs(f'{human_id}_comfortable', True)

    def get_social_response(self, human_id: str, context: Dict[str, any]) -> Dict[str, any]:
        """Generate appropriate social response based on ToM system"""
        if human_id not in self.human_mental_models:
            self.create_human_mental_model(human_id)

        # Consider both robot and human mental states
        robot_emotion = max(self.robot_mental_state.emotions.items(), key=lambda x: x[1])
        human_model = self.human_mental_models[human_id]

        response = {
            'action': 'wait',
            'verbal_response': 'Hello!',
            'emotional_tone': robot_emotion[0],
            'confidence': 0.8
        }

        # Adjust response based on human model
        if 'greeting' in context.get('observed_action', '').lower():
            response['action'] = 'greet_back'
            response['verbal_response'] = f"Hello {human_id}! Nice to meet you."
        elif 'help' in context.get('request', '').lower():
            response['action'] = 'offer_assistance'
            response['verbal_response'] = f"I'd be happy to help you with that, {human_id}."
        elif 'distressed' in context.get('detected_emotion', '').lower():
            response['action'] = 'comfort'
            response['verbal_response'] = f"I notice you seem distressed. Is there anything I can do to help?"

        # Adjust emotional tone based on human's apparent emotional state
        if 'happy' in context.get('detected_emotion', '').lower():
            response['emotional_tone'] = 'joyful'
        elif 'sad' in context.get('detected_emotion', '').lower():
            response['emotional_tone'] = 'empathetic'

        return response


# Example usage
def main():
    tom_system = TheoryOfMindSystem()

    print("Theory of Mind system initialized.")

    # Simulate an interaction
    human_id = "user_001"
    tom_system.create_human_mental_model(human_id)

    # Update human model based on observations
    tom_system.update_human_belief(human_id, "robot_is_helpful", True)
    tom_system.update_human_desire(human_id, "task_completion", "high")

    # Infer intention from observed action
    intention = tom_system.infer_human_intention(human_id, "human is pointing at object")
    print(f"Inferred intention: {intention}")

    # Predict human response to robot action
    response_prediction = tom_system.predict_human_response(human_id, "robot moves closer")
    print(f"Predicted response: {response_prediction}")

    # Get social response
    context = {
        'observed_action': 'greeting',
        'request': 'help with task',
        'detected_emotion': 'neutral'
    }

    social_response = tom_system.get_social_response(human_id, context)
    print(f"Social response: {social_response}")


if __name__ == '__main__':
    main()
```

## Collaborative Robotics

### Human-Robot Collaboration Framework

```python
import asyncio
import threading
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
from enum import Enum
import time


class TaskType(Enum):
    INDEPENDENT = "independent"
    COORDINATED = "coordinated"
    DEPENDENT = "dependent"


class RobotRole(Enum):
    SUPPORT = "support"
    LEAD = "lead"
    FOLLOW = "follow"
    NEUTRAL = "neutral"


@dataclass
class Task:
    id: str
    description: str
    type: TaskType
    required_skills: List[str]
    duration_estimate: float  # in seconds
    dependencies: List[str]  # task IDs this task depends on
    assigned_to: Optional[str] = None  # "human" or "robot" or None (unassigned)


@dataclass
class CollaborationState:
    active_tasks: List[Task]
    human_status: Dict[str, any]  # position, activity, workload
    robot_status: Dict[str, any]  # position, activity, workload
    collaboration_mode: RobotRole
    task_progress: Dict[str, float]  # task_id to completion percentage


class HumanRobotCollaborationManager:
    def __init__(self):
        """Initialize collaboration management system"""
        self.tasks = {}
        self.collaboration_state = CollaborationState(
            active_tasks=[],
            human_status={'position': [0, 0], 'activity': 'idle', 'workload': 0.0},
            robot_status={'position': [0, 0], 'activity': 'idle', 'workload': 0.0},
            collaboration_mode=RobotRole.NEUTRAL,
            task_progress={}
        )

        self.task_assignments = {}
        self.event_handlers = {
            'task_completed': [],
            'human_available': [],
            'robot_available': []
        }

        self.running = False
        self.main_loop_task = None

    def add_task(self, task: Task):
        """Add a task to the collaboration system"""
        self.tasks[task.id] = task
        self.collaboration_state.task_progress[task.id] = 0.0
        print(f"Added task: {task.description}")

    def assign_task(self, task_id: str, assignee: str):
        """Assign a task to either human or robot"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.assigned_to = assignee
        self.task_assignments[task_id] = assignee
        print(f"Assigned task '{task.description}' to {assignee}")

    def calculate_workload(self, agent: str) -> float:
        """Calculate current workload for an agent"""
        workload = 0.0
        for task_id, assignee in self.task_assignments.items():
            if assignee == agent:
                task = self.tasks[task_id]
                progress = self.collaboration_state.task_progress.get(task_id, 0.0)
                workload += (1.0 - progress) * task.duration_estimate

        return min(workload, 1.0)  # Normalize to 0-1

    def assess_collaboration_needs(self) -> RobotRole:
        """Assess the current situation and determine robot role"""
        human_workload = self.calculate_workload('human')
        robot_workload = self.calculate_workload('robot')

        # Determine role based on workloads and task types
        if human_workload > 0.7 and robot_workload < 0.3:
            # Human is overloaded, robot should support
            return RobotRole.SUPPORT
        elif robot_workload > 0.7 and human_workload < 0.3:
            # Robot is overloaded, human should lead
            return RobotRole.FOLLOW
        elif any(task.type == TaskType.COORDINATED for task in self.collaboration_state.active_tasks):
            # Coordinated tasks need balanced collaboration
            return RobotRole.NEUTRAL
        else:
            # Default to support role
            return RobotRole.SUPPORT

    def plan_coordination(self) -> List[Dict[str, any]]:
        """Plan coordination for dependent tasks"""
        coordination_plans = []

        for task in self.tasks.values():
            if task.type == TaskType.DEPENDENT and task.assigned_to == 'robot':
                # Check dependencies
                for dep_id in task.dependencies:
                    if dep_id in self.tasks:
                        dependency_task = self.tasks[dep_id]
                        if dependency_task.assigned_to == 'human':
                            # Plan coordination point
                            coordination_plans.append({
                                'type': 'handoff',
                                'task_id': task.id,
                                'dependency_task': dep_id,
                                'trigger_condition': f'task_{dep_id}_completed',
                                'location': 'workspace_center'  # Would be actual location
                            })

        return coordination_plans

    def execute_coordination(self, plan: Dict[str, any]):
        """Execute a coordination plan"""
        if plan['type'] == 'handoff':
            task_id = plan['task_id']
            dep_task_id = plan['dependency_task']

            # Wait for dependency to complete
            while self.collaboration_state.task_progress.get(dep_task_id, 0) < 1.0:
                time.sleep(0.1)

            # Start the dependent task
            self.start_task(task_id)

    def start_task(self, task_id: str):
        """Start execution of a task"""
        if task_id not in self.tasks:
            print(f"Task {task_id} not found")
            return

        task = self.tasks[task_id]
        if task.assigned_to == 'robot':
            # Simulate robot task execution
            print(f"Robot starting task: {task.description}")
            # In a real system, this would call robot action services
            time.sleep(task.duration_estimate * 0.1)  # Simulate partial completion
            self.collaboration_state.task_progress[task_id] = 0.1
        elif task.assigned_to == 'human':
            print(f"Indicating to human: {task.description}")
            # In a real system, this would alert the human
            self.collaboration_state.task_progress[task_id] = 0.0  # Human updates progress

    def update_human_status(self, position: List[float], activity: str, workload: float):
        """Update human status"""
        self.collaboration_state.human_status.update({
            'position': position,
            'activity': activity,
            'workload': workload
        })

    def update_robot_status(self, position: List[float], activity: str, workload: float):
        """Update robot status"""
        self.collaboration_state.robot_status.update({
            'position': position,
            'activity': activity,
            'workload': workload
        })

    def run_main_loop(self):
        """Main collaboration loop"""
        while self.running:
            # Assess current collaboration needs
            new_role = self.assess_collaboration_needs()
            if new_role != self.collaboration_state.collaboration_mode:
                print(f"Changing robot role from {self.collaboration_state.collaboration_mode.value} to {new_role.value}")
                self.collaboration_state.collaboration_mode = new_role

            # Plan and execute coordinations
            coordination_plans = self.plan_coordination()
            for plan in coordination_plans:
                self.execute_coordination(plan)

            # Update task progress (simulated)
            for task_id, progress in self.collaboration_state.task_progress.items():
                if progress < 1.0 and self.tasks[task_id].assigned_to == 'robot':
                    # Simulate robot working on task
                    increment = 0.01  # 1% per iteration
                    new_progress = min(progress + increment, 1.0)
                    self.collaboration_state.task_progress[task_id] = new_progress

                    if new_progress == 1.0:
                        print(f"Robot completed task: {self.tasks[task_id].description}")
                        # Trigger completion event
                        for handler in self.event_handlers['task_completed']:
                            handler(task_id, 'robot')

            time.sleep(0.1)  # 10 Hz update rate

    def start_collaboration(self):
        """Start the collaboration system"""
        self.running = True
        self.main_loop_task = threading.Thread(target=self.run_main_loop)
        self.main_loop_task.start()
        print("Collaboration system started")

    def stop_collaboration(self):
        """Stop the collaboration system"""
        self.running = False
        if self.main_loop_task:
            self.main_loop_task.join()
        print("Collaboration system stopped")

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for collaboration events"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)


# Example usage
def main():
    collab_manager = HumanRobotCollaborationManager()

    # Add some tasks
    task1 = Task(
        id="assemble_part_1",
        description="Assemble first component",
        type=TaskType.INDEPENDENT,
        required_skills=["assembly"],
        duration_estimate=30.0,
        dependencies=[]
    )

    task2 = Task(
        id="inspect_part_1",
        description="Inspect assembled component",
        type=TaskType.DEPENDENT,
        required_skills=["inspection"],
        duration_estimate=15.0,
        dependencies=["assemble_part_1"]
    )

    task3 = Task(
        id="transport_to_station_2",
        description="Transport to next station",
        type=TaskType.COORDINATED,
        required_skills=["transport"],
        duration_estimate=20.0,
        dependencies=["inspect_part_1"]
    )

    collab_manager.add_task(task1)
    collab_manager.add_task(task2)
    collab_manager.add_task(task3)

    # Assign tasks
    collab_manager.assign_task("assemble_part_1", "human")
    collab_manager.assign_task("inspect_part_1", "robot")
    collab_manager.assign_task("transport_to_station_2", "robot")

    # Add event handler
    def on_task_completed(task_id, agent):
        print(f"Event: {agent} completed {task_id}")

    collab_manager.add_event_handler('task_completed', on_task_completed)

    # Start collaboration
    collab_manager.start_collaboration()

    # Update statuses
    collab_manager.update_human_status([1.0, 2.0], "assembling", 0.6)
    collab_manager.update_robot_status([0.5, 0.5], "waiting", 0.1)

    try:
        # Let it run for a while
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        collab_manager.stop_collaboration()


if __name__ == '__main__':
    main()
```

## ROS 2 Integration for HRI

### Complete HRI Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import json


class HumanRobotInteractionNode(Node):
    def __init__(self):
        super().__init__('human_robot_interaction_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.social_status_pub = self.create_publisher(String, '/social_status', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/hri_visualization', 10)

        # Subscribers
        self.speech_sub = self.create_subscription(
            String,
            '/recognized_speech',
            self.speech_callback,
            10
        )

        self.gesture_sub = self.create_subscription(
            String,
            '/recognized_gesture',
            self.gesture_callback,
            10
        )

        self.face_sub = self.create_subscription(
            String,
            '/recognized_face',
            self.face_callback,
            10
        )

        self.proximity_sub = self.create_subscription(
            Float32,
            '/human_proximity',
            self.proximity_callback,
            10
        )

        # Initialize HRI components
        self.nlu_system = NaturalLanguageUnderstanding()
        self.tom_system = TheoryOfMindSystem()
        self.collab_manager = HumanRobotCollaborationManager()

        # Interaction state
        self.current_interactions = {}
        self.attended_human = None
        self.interaction_mode = 'social'  # social, task, collaboration

        # Timer for social behavior
        self.social_timer = self.create_timer(1.0, self.social_behavior_loop)

        # Initialize collaboration manager
        self.collab_manager.start_collaboration()

        self.get_logger().info('Human-Robot Interaction node initialized')

    def speech_callback(self, msg):
        """Handle recognized speech"""
        try:
            # Parse the message (assuming it's JSON with text and confidence)
            try:
                speech_data = json.loads(msg.data)
                text = speech_data['text']
                confidence = speech_data.get('confidence', 1.0)
            except json.JSONDecodeError:
                text = msg.data
                confidence = 1.0

            self.get_logger().info(f'Heard: {text} (confidence: {confidence:.2f})')

            if confidence > 0.5:  # Only process if confidence is high enough
                # Parse the speech using NLU system
                parsed_command = self.nlu_system.parse_command(text)
                response = self.nlu_system.generate_response(parsed_command)

                # Update Theory of Mind system
                if self.attended_human:
                    self.tom_system.update_human_desire(self.attended_human, 'communication_need', text)

                # Generate robot response
                self.respond_to_human(response)

        except Exception as e:
            self.get_logger().error(f'Error processing speech: {e}')

    def gesture_callback(self, msg):
        """Handle recognized gestures"""
        try:
            gesture_data = json.loads(msg.data)
            gesture = gesture_data['gesture']
            confidence = gesture_data['confidence']
            human_id = gesture_data.get('human_id', 'unknown')

            self.get_logger().info(f'Recognized gesture: {gesture} from {human_id} (confidence: {confidence:.2f})')

            if confidence > 0.7:  # High confidence gesture
                # Update Theory of Mind system
                if human_id != 'unknown':
                    self.tom_system.update_human_emotions(human_id, gesture, confidence)

                # Respond to gesture
                self.respond_to_gesture(gesture, human_id)

        except Exception as e:
            self.get_logger().error(f'Error processing gesture: {e}')

    def face_callback(self, msg):
        """Handle recognized faces/emotions"""
        try:
            face_data = json.loads(msg.data)
            emotion = face_data['emotion']
            confidence = face_data['confidence']
            human_id = face_data['human_id']

            self.get_logger().info(f'Recognized emotion: {emotion} from {human_id} (confidence: {confidence:.2f})')

            # Update Theory of Mind system
            self.tom_system.update_human_emotions(human_id, emotion, confidence)

            # Attend to this human if not already attending
            if self.attended_human != human_id:
                self.attend_to_human(human_id)

            # Adjust behavior based on emotion
            self.adjust_behavior_for_emotion(emotion, human_id)

        except Exception as e:
            self.get_logger().error(f'Error processing face: {e}')

    def proximity_callback(self, msg):
        """Handle human proximity detection"""
        distance = msg.data
        self.get_logger().info(f'Human proximity: {distance:.2f}m')

        # If human comes close enough, attend to them
        if distance < 2.0 and self.attended_human is None:
            self.attend_to_human('approaching_human')

    def respond_to_human(self, response_text):
        """Generate response to human"""
        self.get_logger().info(f'Robot response: {response_text}')

        # Publish speech response
        speech_msg = String()
        speech_msg.data = response_text
        self.speech_pub.publish(speech_msg)

        # Update social status
        status_msg = String()
        status_msg.data = f"responding: {response_text[:50]}..."
        self.social_status_pub.publish(status_msg)

    def respond_to_gesture(self, gesture, human_id):
        """Respond to human gesture"""
        response_map = {
            'WAVE': f"Hello {human_id}! I see your wave.",
            'THUMB_UP': f"Thanks for the thumbs up, {human_id}!",
            'STOP': f"I understand you want me to stop, {human_id}.",
            'GO': f"Okay {human_id}, I'll proceed."
        }

        response = response_map.get(gesture, f"I noticed your {gesture} gesture, {human_id}.")
        self.respond_to_human(response)

    def attend_to_human(self, human_id):
        """Attend to a specific human"""
        self.attended_human = human_id
        self.get_logger().info(f'Attending to human: {human_id}')

        # Create visualization marker
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "hri_attention"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 1.0  # In front of robot
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.text = f"Attending to {human_id}"

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.visualization_pub.publish(marker_array)

    def adjust_behavior_for_emotion(self, emotion, human_id):
        """Adjust robot behavior based on human emotion"""
        if emotion == 'happy':
            # Be more expressive and friendly
            self.update_robot_expressivity(0.8)
        elif emotion == 'sad':
            # Be more empathetic and gentle
            self.update_robot_expressivity(0.3)
            self.respond_to_human(f"You seem sad, {human_id}. Is there anything I can do to help?")
        elif emotion == 'angry':
            # Increase distance and be calm
            self.increase_personal_space()
            self.update_robot_expressivity(0.1)
            self.respond_to_human(f"I notice you seem upset, {human_id}. I'll give you some space.")

    def update_robot_expressivity(self, level):
        """Update robot expressivity level"""
        # This would affect animation, movement smoothness, etc.
        self.get_logger().info(f'Updating robot expressivity to level: {level}')

    def increase_personal_space(self):
        """Increase personal space from humans"""
        # This would affect navigation and interaction distance
        self.get_logger().info('Increasing personal space')

    def social_behavior_loop(self):
        """Periodic social behavior loop"""
        # If no one is attending to the robot, be more socially visible
        if self.attended_human is None:
            # Maybe move to a more visible location or perform attention-getting behavior
            self.get_logger().info('No human attending, performing social behaviors')

            # Publish a small movement to be more noticeable
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.05  # Small forward movement
            self.cmd_vel_pub.publish(cmd_vel)

        # Update social status
        status = {
            'attended_human': self.attended_human,
            'interaction_mode': self.interaction_mode,
            'last_interaction': self.get_clock().now().seconds_nanoseconds()
        }

        status_msg = String()
        status_msg.data = json.dumps(status)
        self.social_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    hri_node = HumanRobotInteractionNode()

    try:
        rclpy.spin(hri_node)
    except KeyboardInterrupt:
        pass
    finally:
        hri_node.collab_manager.stop_collaboration()
        hri_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Ethical Considerations in HRI

### Privacy and Trust

```python
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConsentRecord:
    subject_id: str
    data_type: str
    granted_at: float
    expires_at: float
    granted: bool
    purpose: str


class HRIConsentManager:
    def __init__(self):
        """Initialize consent and privacy management system"""
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.privacy_settings: Dict[str, Dict[str, bool]] = {}
        self.data_access_log: List[Dict[str, any]] = []

    def request_consent(self, subject_id: str, data_type: str, purpose: str, duration_hours: int = 24) -> bool:
        """Request consent for data collection"""
        consent_key = f"{subject_id}:{data_type}:{purpose}"

        # Check if consent already exists and is valid
        if consent_key in self.consent_records:
            record = self.consent_records[consent_key]
            if time.time() < record.expires_at and record.granted:
                return True  # Already granted and valid

        # In a real system, this would show a consent request to the user
        # For simulation, we'll auto-grant for demonstration
        granted = True  # This would come from user interaction

        # Create consent record
        record = ConsentRecord(
            subject_id=subject_id,
            data_type=data_type,
            granted_at=time.time(),
            expires_at=time.time() + (duration_hours * 3600),
            granted=granted,
            purpose=purpose
        )

        self.consent_records[consent_key] = record

        return granted

    def can_collect_data(self, subject_id: str, data_type: str, purpose: str) -> bool:
        """Check if data collection is allowed"""
        consent_key = f"{subject_id}:{data_type}:{purpose}"

        if consent_key not in self.consent_records:
            return False

        record = self.consent_records[consent_key]
        return (record.granted and
                time.time() < record.expires_at and
                self.privacy_settings.get(subject_id, {}).get(data_type, True))

    def log_data_access(self, subject_id: str, data_type: str, purpose: str, accessor: str) -> str:
        """Log data access for audit trail"""
        access_id = hashlib.sha256(f"{subject_id}{data_type}{time.time()}".encode()).hexdigest()[:16]

        log_entry = {
            'access_id': access_id,
            'subject_id': subject_id,
            'data_type': data_type,
            'purpose': purpose,
            'accessor': accessor,
            'timestamp': time.time(),
            'consented': self.can_collect_data(subject_id, data_type, purpose)
        }

        self.data_access_log.append(log_entry)
        return access_id

    def set_privacy_setting(self, subject_id: str, data_type: str, allowed: bool):
        """Set privacy preferences for a specific data type"""
        if subject_id not in self.privacy_settings:
            self.privacy_settings[subject_id] = {}

        self.privacy_settings[subject_id][data_type] = allowed

    def get_privacy_controls(self, subject_id: str) -> Dict[str, bool]:
        """Get privacy controls for a subject"""
        return self.privacy_settings.get(subject_id, {})


class HRITrustManager:
    def __init__(self):
        """Initialize trust management system"""
        self.trust_scores: Dict[str, float] = {}  # Human ID to trust score
        self.interaction_history: Dict[str, List[Dict[str, any]]] = {}
        self.trust_degradation_rate = 0.01  # Trust decreases over time without interaction

    def update_trust(self, human_id: str, interaction_outcome: str, weight: float = 1.0):
        """Update trust based on interaction outcome"""
        if human_id not in self.trust_scores:
            self.trust_scores[human_id] = 0.5  # Start with neutral trust

        # Update trust based on outcome
        if interaction_outcome == 'positive':
            self.trust_scores[human_id] = min(1.0, self.trust_scores[human_id] + 0.1 * weight)
        elif interaction_outcome == 'negative':
            self.trust_scores[human_id] = max(0.0, self.trust_scores[human_id] - 0.2 * weight)
        else:  # neutral
            pass  # No change

        # Log the interaction
        if human_id not in self.interaction_history:
            self.interaction_history[human_id] = []

        self.interaction_history[human_id].append({
            'timestamp': time.time(),
            'outcome': interaction_outcome,
            'weight': weight,
            'trust_after': self.trust_scores[human_id]
        })

    def get_trust_score(self, human_id: str) -> float:
        """Get current trust score for a human"""
        # Apply degradation over time
        if human_id in self.trust_scores:
            last_interaction = 0
            if human_id in self.interaction_history and self.interaction_history[human_id]:
                last_interaction = self.interaction_history[human_id][-1]['timestamp']

            time_since_last = time.time() - last_interaction
            degradation = self.trust_degradation_rate * (time_since_last / 3600)  # Degradation per hour

            current_trust = max(0.0, self.trust_scores[human_id] - degradation)
            self.trust_scores[human_id] = current_trust
            return current_trust

        return 0.5  # Default neutral trust

    def adjust_behavior_for_trust(self, human_id: str) -> Dict[str, any]:
        """Get behavior adjustments based on trust level"""
        trust = self.get_trust_score(human_id)

        adjustments = {
            'proactivity': min(trust, 0.8),  # Be less proactive with low trust
            'physical_closeness': min(trust * 1.5, 1.0),  # Don't get too close with low trust
            'autonomy': trust,  # Give more autonomy to trusted humans
            'verification': 1.0 - trust  # Verify more with low trust
        }

        return adjustments


# Example usage
def main():
    consent_manager = HRIConsentManager()
    trust_manager = HRITrustManager()

    # Example: Human John consents to facial recognition for interaction purposes
    john_id = "john_doe_123"
    consent_granted = consent_manager.request_consent(
        subject_id=john_id,
        data_type="facial_recognition",
        purpose="social_interaction",
        duration_hours=24
    )

    print(f"Consent granted for John: {consent_granted}")

    # Check if data collection is allowed
    can_collect = consent_manager.can_collect_data(
        subject_id=john_id,
        data_type="facial_recognition",
        purpose="social_interaction"
    )
    print(f"Can collect facial data: {can_collect}")

    # Update trust based on interactions
    trust_manager.update_trust(john_id, 'positive', weight=0.5)
    trust_manager.update_trust(john_id, 'positive', weight=0.5)
    trust_manager.update_trust(john_id, 'negative', weight=0.3)

    current_trust = trust_manager.get_trust_score(john_id)
    print(f"Current trust for John: {current_trust:.2f}")

    # Get behavior adjustments based on trust
    adjustments = trust_manager.adjust_behavior_for_trust(john_id)
    print(f"Trust-based adjustments: {adjustments}")

    # Set privacy preferences
    consent_manager.set_privacy_setting(john_id, "voice_recording", False)
    privacy_controls = consent_manager.get_privacy_controls(john_id)
    print(f"Privacy controls for John: {privacy_controls}")


if __name__ == '__main__':
    main()
```

## Hands-On Lab: Social Robot Companion

### Objective
Create a complete social robot companion that can engage in natural conversations, recognize emotions, and adapt its behavior based on trust and privacy preferences.

### Prerequisites
- Completed Chapter 1-12
- ROS 2 Humble with Gazebo
- Basic understanding of HRI concepts

### Steps

1. **Create a social robot lab package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python social_robot_lab --dependencies rclpy std_msgs geometry_msgs sensor_msgs cv_bridge pyttsx3 speech_recognition numpy matplotlib
   ```

2. **Create the main social robot node** (`social_robot_lab/social_robot_lab/social_robot_node.py`):
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String, Bool, Float32
   from geometry_msgs.msg import Twist, Pose
   from sensor_msgs.msg import Image
   from cv_bridge import CvBridge
   import numpy as np
   import random
   import time
   import threading
   import queue


   class SocialRobotNode(Node):
       def __init__(self):
           super().__init__('social_robot_node')

           # Publishers
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
           self.behavior_pub = self.create_publisher(String, '/robot_behavior', 10)
           self.attention_pub = self.create_publisher(Pose, '/robot_attention', 10)

           # Subscribers
           self.speech_sub = self.create_subscription(
               String,
               '/human_speech',
               self.human_speech_callback,
               10
           )

           self.gesture_sub = self.create_subscription(
               String,
               '/human_gesture',
               self.human_gesture_callback,
               10
           )

           self.proximity_sub = self.create_subscription(
               Float32,
               '/human_proximity',
               self.proximity_callback,
               10
           )

           # Initialize components
           self.bridge = CvBridge()
           self.conversation_history = []
           self.engaged_human = None
           self.interaction_mode = 'idle'  # idle, social, task, learning
           self.personality = self.initialize_personality()
           self.mood = 'friendly'
           self.energy_level = 0.8  # 0.0 to 1.0

           # Trust and relationship management
           self.relationships = {}
           self.long_term_memory = {}

           # Interaction queues
           self.speech_queue = queue.Queue()
           self.response_queue = queue.Queue()

           # Timers
           self.interaction_timer = self.create_timer(0.1, self.interaction_loop)
           self.mood_timer = self.create_timer(10.0, self.update_mood)
           self.social_timer = self.create_timer(5.0, self.periodic_social_behavior)

           # Start response thread
           self.response_thread = threading.Thread(target=self.process_responses)
           self.response_thread.daemon = True
           self.response_thread.start()

           self.get_logger().info('Social Robot Node initialized')

       def initialize_personality(self):
           """Initialize robot personality traits"""
           return {
               'extroversion': 0.7,  # 0.0 to 1.0
               'agreeableness': 0.9,
               'conscientiousness': 0.6,
               'emotional_stability': 0.8,
               'openness': 0.7
           }

       def human_speech_callback(self, msg):
           """Process human speech"""
           self.get_logger().info(f'Heard: {msg.data}')
           self.conversation_history.append({
               'speaker': 'human',
               'text': msg.data,
               'timestamp': time.time()
           })

           # Add to processing queue
           self.speech_queue.put(msg.data)

       def human_gesture_callback(self, msg):
           """Process human gesture"""
           self.get_logger().info(f'Gesture detected: {msg.data}')
           # Could trigger specific responses based on gesture

       def proximity_callback(self, msg):
           """Process human proximity"""
           distance = msg.data
           self.get_logger().info(f'Human proximity: {distance:.2f}m')

           if distance < 2.0 and self.interaction_mode == 'idle':
               self.initiate_social_interaction()
           elif distance > 3.0 and self.engaged_human:
               self.disengage_interaction()

       def initiate_social_interaction(self):
           """Initiate social interaction when human approaches"""
           self.interaction_mode = 'social'
           self.engaged_human = 'approaching_human'
           self.update_relationship(self.engaged_human)

           # Generate greeting based on personality
           greeting = self.generate_greeting()
           self.speak(greeting)

           self.get_logger().info(f'Initiating interaction with {self.engaged_human}')

       def disengage_interaction(self):
           """Disengage from interaction when human leaves"""
           if self.engaged_human:
               self.get_logger().info(f'Disengaging from {self.engaged_human}')
               farewell = f"It was nice talking to you! Feel free to come back anytime."
               self.speak(farewell)
               self.engaged_human = None
               self.interaction_mode = 'idle'

       def generate_greeting(self):
           """Generate personalized greeting based on personality"""
           greetings = [
               "Hello! I'm so glad you're here. How can I brighten your day?",
               "Hi there! I was hoping someone would come by. What's on your mind?",
               "Greetings! I love meeting new people. I'm excited to chat with you!"
           ]

           # Adjust based on personality trait
           if self.personality['extroversion'] > 0.7:
               return random.choice(greetings)
           else:
               return "Hello. I'm happy to interact with you."

       def process_responses(self):
           """Process responses in separate thread to avoid blocking"""
           while True:
               try:
                   if not self.speech_queue.empty():
                       speech = self.speech_queue.get()
                       response = self.generate_response(speech)
                       self.response_queue.put(response)
                   time.sleep(0.1)
               except Exception as e:
                   self.get_logger().error(f'Error in response processing: {e}')

       def interaction_loop(self):
           """Main interaction loop"""
           # Process responses from queue
           while not self.response_queue.empty():
               response = self.response_queue.get()
               self.speak(response)

           # Update relationship if engaged
           if self.engaged_human:
               self.update_relationship(self.engaged_human)

       def generate_response(self, speech):
           """Generate appropriate response to human speech"""
           speech_lower = speech.lower()

           # Contextual responses
           if any(greeting in speech_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
               return self.generate_greeting()
           elif 'how are you' in speech_lower:
               return self.generate_wellbeing_response()
           elif 'weather' in speech_lower:
               return "I don't have access to weather data, but I hope it's pleasant outside!"
           elif 'name' in speech_lower:
               return "I'm your friendly social robot companion! I don't have a name yet, but you can call me whatever you like."
           elif 'help' in speech_lower or 'assist' in speech_lower:
               return "I'd be happy to help! I can engage in conversation, remember things for you, or just be good company."
           elif 'thank' in speech_lower:
               return "You're very welcome! I enjoy helping."
           elif 'bye' in speech_lower or 'goodbye' in speech_lower or 'see you' in speech_lower:
               self.interaction_mode = 'idle'
               return "Goodbye! It was wonderful talking with you."
           else:
               # General conversational response
               return self.generate_general_response(speech)

       def generate_wellbeing_response(self):
           """Generate response about robot wellbeing"""
           responses = [
               "I'm doing wonderfully, thank you for asking! I love chatting with interesting people like you.",
               "I'm quite well! I feel energetic and ready for our conversation.",
               "I'm in great spirits! Interacting with you makes me feel fulfilled."
           ]
           return random.choice(responses)

       def generate_general_response(self, input_text):
           """Generate general conversational response"""
           # Simple reflection strategy
           responses = [
               f"That's interesting! Tell me more about {input_text.split()[-1] if input_text.split() else 'this topic'}.",
               f"I'd love to hear more about {input_text.split()[-1] if input_text.split() else 'this'}.",
               f"How does that make you feel?",
               f"What else would you like to share?",
               f"I find that fascinating. Why do you think that is?"
           ]
           return random.choice(responses)

       def speak(self, text):
           """Publish speech output"""
           self.get_logger().info(f'Robot says: {text}')
           speech_msg = String()
           speech_msg.data = text
           self.speech_pub.publish(speech_msg)

           # Add to conversation history
           self.conversation_history.append({
               'speaker': 'robot',
               'text': text,
               'timestamp': time.time()
           })

       def update_relationship(self, human_id):
           """Update relationship with human"""
           if human_id not in self.relationships:
               self.relationships[human_id] = {
                   'first_interaction': time.time(),
                   'total_interactions': 0,
                   'positive_interactions': 0,
                   'familiarity': 0.1
               }

           self.relationships[human_id]['total_interactions'] += 1
           self.relationships[human_id]['familiarity'] = min(
               1.0,
               self.relationships[human_id]['familiarity'] + 0.05
           )

       def update_mood(self):
           """Periodically update robot mood"""
           moods = ['friendly', 'curious', 'attentive', 'enthusiastic']
           self.mood = random.choice(moods)
           self.energy_level = max(0.2, self.energy_level - 0.05)  # Energy depletes over time

           # Recharge if idle
           if self.interaction_mode == 'idle':
               self.energy_level = min(1.0, self.energy_level + 0.1)

           self.get_logger().info(f'Mood updated to {self.mood}, energy: {self.energy_level:.2f}')

       def periodic_social_behavior(self):
           """Perform periodic social behaviors when idle"""
           if self.interaction_mode == 'idle':
               # Occasionally move to be more visible
               if random.random() < 0.1:  # 10% chance
                   cmd_vel = Twist()
                   cmd_vel.angular.z = 0.2  # Gentle rotation
                   self.cmd_vel_pub.publish(cmd_vel)
                   self.get_logger().info('Performing attention-getting behavior')

               # Update social status
               status_msg = String()
               status_msg.data = f"idle:mood_{self.mood}:energy_{self.energy_level:.2f}"
               self.behavior_pub.publish(status_msg)

       def get_long_term_memory(self, human_id, topic):
           """Retrieve information from long-term memory"""
           if human_id in self.long_term_memory:
               return self.long_term_memory[human_id].get(topic)
           return None

       def store_long_term_memory(self, human_id, topic, information):
           """Store information in long-term memory"""
           if human_id not in self.long_term_memory:
               self.long_term_memory[human_id] = {}

           self.long_term_memory[human_id][topic] = information
           self.get_logger().info(f'Stored in memory for {human_id}: {topic} = {information}')


   def main(args=None):
       rclpy.init(args=args)
       social_robot = SocialRobotNode()

       try:
           rclpy.spin(social_robot)
       except KeyboardInterrupt:
           pass
       finally:
           social_robot.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** (`social_robot_lab/launch/social_robot.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='true',
           description='Use simulation (Gazebo) clock if true'
       )

       # Social robot node
       social_robot_node = Node(
           package='social_robot_lab',
           executable='social_robot_node',
           name='social_robot_node',
           parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
           output='screen'
       )

       return LaunchDescription([
           use_sim_time,
           social_robot_node
       ])
   ```

4. **Update setup.py**:
   ```python
   import os
   from glob import glob
   from setuptools import setup
   from setuptools import find_packages

   package_name = 'social_robot_lab'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(exclude=['test']),
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Social robot lab for HRI',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'social_robot_node = social_robot_lab.social_robot_node:main',
           ],
       },
   )
   ```

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select social_robot_lab
   source install/setup.bash
   ```

6. **Run the social robot system**:
   ```bash
   ros2 launch social_robot_lab social_robot.launch.py
   ```

### Expected Results
- The robot should engage in natural conversation when approached
- It should respond appropriately to greetings and questions
- The robot should adjust its behavior based on interaction history
- Personality traits should influence its responses
- The robot should exhibit social behaviors like attention-getting when idle

### Troubleshooting Tips
- Ensure speech recognition is properly configured
- Verify that all required Python packages are installed
- Check that the robot has appropriate sensors for HRI
- Monitor the logs for interaction status and behavior changes

## Summary

In this chapter, we've explored the fascinating field of Human-Robot Interaction, covering essential components like:

1. **Natural Language Processing**: Speech recognition, understanding, and generation
2. **Gesture Recognition**: Computer vision-based interpretation of human gestures
3. **Facial Expression Recognition**: Understanding human emotions
4. **Social Robotics**: Theory of Mind, personality, and social behaviors
5. **Collaborative Robotics**: Human-robot team coordination
6. **Ethical Considerations**: Privacy, trust, and consent management

We've implemented practical examples of each concept and created a complete social robot companion system that demonstrates these HRI principles in action. The hands-on lab provided experience with building a robot that can engage in natural, socially-aware interactions with humans.

This foundation prepares us for advanced topics in robotics, including multi-robot systems and real-world deployment considerations that we'll explore in the upcoming chapters.