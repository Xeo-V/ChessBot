import cv2
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt

# Load your trained model
model = load_model("D:\\projects and shit\\ChessBot\\trained_chess_model.h5")

# Initialize Matplotlib window
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

while True:
    time.sleep(3)  # Wait for 3 seconds before capturing the next screen
    
    # Step 1: Capture the entire screen
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Step 2: Detect the chessboard
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Step 3: Grid Localization
    cell_w = w // 8
    cell_h = h // 8
    squares = []
    for i in range(8):
        for j in range(8):
            startX = x + i * cell_w
            startY = y + j * cell_h
            endX = startX + cell_w
            endY = startY + cell_h
            squares.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 1)
    
    # Step 4: Piece recognition
    for square_coordinates in squares:
        square = frame[square_coordinates[1]:square_coordinates[3], square_coordinates[0]:square_coordinates[2]]
        processed_image = cv2.resize(square, (128, 128))
        processed_image = np.expand_dims(processed_image, axis=0)
        
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        piece_name = ["King", "Queen", "Bishop", "Knight", "Rook", "Pawn", "Empty"][predicted_class]
        
        # Step 5: Overlay the recognized piece name onto the square
        text_coordinates = (square_coordinates[0], square_coordinates[1] - 5)
        cv2.putText(frame, piece_name, text_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Step 6: Display the updated screen with overlaid piece names using Matplotlib
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)
    ax.clear()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.ioff()  # Turn off interactive mode