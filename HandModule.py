import mediapipe as mp
import cv2
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                    self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots onhands total 20 landmark points

    def findHands(self,img,draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # process the frame

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    print(id,lm)
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    print(id, cx, cy)
            if draw:
                self.mpDraw.draw_landmarks(img, handLms,
                                           self.mpHands.HAND_CONNECTIONS)

        return img
    def findPosition(self,img, handNo=0, draw=True):


         self.lmlist = []

         # check wether any landmark was detected
         if self.results.multi_hand_landmarks:
             #Which hand are we talking about
             myHand = self.results.multi_hand_landmarks[handNo]
             #myHand = self.results.multi_hand_landmarks[handNo]
             # Get id number and landmark information
             for id, lm in enumerate(myHand.landmark):
                 # id will give id of landmark in exact index number
                 # height width and channel
                 h,w,c = img.shape
                 #find the position
                 cx,cy = int(lm.x*w), int(lm.y*h) #center
                 self.lmlist.append([id, cx, cy])
                 if draw:
                     cv2.circle(img,(cx,cy), 5, (255, 0, 0), cv2.FILLED)


             # Draw circle for 0th landmark
             if draw:
                 cv2.circle(img,(cx,cy), 15 , (255,0,255), cv2.FILLED)

         return self.lmlist

    def findAngle(self, img, p1, p2, p3, draw=True):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degree(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360

            # print(angle)

            # Draw
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
                cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return angle



def main():
    #Frame rates
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()


    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
             print(lmList[14])


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime



        #cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.waitKey(1)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()