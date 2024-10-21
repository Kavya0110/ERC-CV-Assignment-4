import cv2
import mediapipe as mp

#starting live stream
vid = cv2.VideoCapture(0)

#using in-built utilities in the mediapipe code
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    
    #capturing the frame to analyze
    success, img = vid.read()
    
    #converting to rgb because "hands" will require rgb format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #using in-built function to process image
    results = hands.process(imgRGB)


    if results.multi_hand_landmarks:

        for handlmks in results.multi_hand_landmarks:
            
            #drawing the contour lines on the frame captured
            mpDraw.draw_landmarks(img, handlmks, mpHands.HAND_CONNECTIONS)



#displaying the  flipped live stream
    cv2.imshow("Image", cv2.flip(img, 1))
    
    #destroying the window when "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releasing the webcam and closing the program
vid.release()
cv2.destroyAllWindows()
