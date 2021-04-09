from cv2 import cv2 
import numpy as np 
from collections import OrderedDict
import face_recognition

#A class that tracks objects on a screen
class ObjectTracker:

    #initiliases variables
    def __init__(self, maxDissapeared=50, IDString="#"):
        self.NextObjectID = 0
        self.Objects = OrderedDict()
        self.Dissapeared = OrderedDict()

        #Number of frames an object can go missing for
        self.maxDissapeared = maxDissapeared
        #String to identify what's being tracked
        self.IDString = IDString

    #Registers a new object
    def Register(self, Object):
        self.Objects[self.NextObjectID] = Object
        self.Dissapeared[self.NextObjectID] = 0
        self.NextObjectID += 1
    
    #Deregisters the object once it has gone
    def Deregister(self, ObjectID):
        del self.Objects[ObjectID]
        del self.Dissapeared[ObjectID]
    
    def update(self, ObjectList):
        #If no objects on the screen then increase the dissapered value by one
        if len(ObjectList)==0:
            for key in list(self.Dissapeared.keys()):
                self.Dissapeared[key] += 1

                #if at max dissapeared value then deregister
                if self.Dissapeared[key] > self.maxDissapeared:
                    self.Deregister(key)
            
            #Work done, so early return
            return self.Objects
        
        #If there are no currently tracked objects, begin tracking their centers
        if len(self.Objects) == 0:
            for face in ObjectList:
                #HAAR CLASSIFIER
                #center = (face[0]+face[2]//2, face[1]+face[3]//2)

                #FACE_RECOGNITION (HOG)
                center = ((face[1]+face[3])//2, (face[0]+face[2])//2)

                self.Register(center)
        else:
            #Matrix that will store the objects in current frame, their centers, and their Euclidean distances
            EucMatrix = []
            newCenterList = []

            #Go through current frame faces
            for newface in ObjectList:
                
                #FACE_RECOGNITION (HOG)
                newCenter = ((newface[1]+newface[3])//2, (newface[0]+newface[2])//2)

                #HAAR CLASSIFIER
                #center = (face[0]+face[2]//2, face[1]+face[3]//2)
                EucList = []
                
                #Go through the tracked items and find the Euclidean distances
                for ind, oldFace in self.Objects.items():
                    eucDistance = np.sqrt((oldFace[0]-newCenter[0])**2 + (oldFace[1]-newCenter[1])**2)
                    EucList.append((ind, eucDistance, newCenter))

                #Sort the list based on the shortest distance
                EucList = sorted(EucList, key=lambda i:i[1])

                #Append row to matrix
                EucMatrix.append(EucList)

            #Sort matrix dependant on their the shortest distances, the logic here is that the first n columns
            #were we've not repeated an ID would be objects that are already tracked and are in the current frame
            EucMatrix = sorted(EucMatrix, key=lambda i:i[0][1])

            #Current cummalutive total so we know where we are at in the already
            #tracked objects list
            ConcurrentID = -1

            #ID's used for objects on the screen
            usedIdList = []
            numberOfKnownObjects = len(self.Objects)

            #Objects in the frame logic
            for eucObject in EucMatrix:
                ConcurrentID+=1

                #If we've come to new objects to add 
                if eucObject[0][0] not in usedIdList:
                    #If it's a known object and it is on the screen
                    self.Objects[eucObject[0][0]] = eucObject[0][2]
                    self.Dissapeared[eucObject[0][0]] = 0

                #If we've repeated an ID then we're at the new objects.
                else:
                    self.Register(eucObject[0][2])

                #Keep track of the ID's that have currently been used in the current frame,
                #This is so we can work out the ID's that have dissapeared 
                usedIdList.append(eucObject[0][0])
            

            #Logic for tracked objects that are currently not in the frame

            #ID's that are tracked but not on the screen 
            DissapearedIds = [obj for obj in self.Objects if obj not in usedIdList]

            #Loop through and increment their dissapeared number
            for Id in DissapearedIds:
                self.Dissapeared[Id] += 1

                if self.Dissapeared[Id] == self.maxDissapeared:
                    self.Deregister(Id)
                
    #Draw some ugly dots with text because I forgot to consider the box co-ords. 
    def DrawObjects(self, frame):
        for ind, center in self.Objects.items():
            identifierString = self.IDString + str(ind)
            frame = cv2.ellipse(frame, center, (1,1), 0, 0, 360, (0,0,255), 5)
            frame = cv2.putText(frame, identifierString , center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return frame
            


def Main():
    #Capture main webcam stream
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #HAAR CLASSIFIER

    #Load the classifier model
    #face_cascade = cv2.CascadeClassifier()
    #Load the pre-trained data
    #'env\Lib\site-packages\cv2\data\''
    #face_cascade.load(cv2.samples.findFile('env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'))

    #init a object tracking class
    OT = ObjectTracker(maxDissapeared=20, IDString="face #")

    while cap.isOpened():
        #Capture frame-by-frame
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #FOR THE HAAR_CASCADE
        #gray = cv2.equalizeHist(gray)
        #faces = face_cascade.detectMultiScale(gray)

        faces = face_recognition.face_locations(gray, model='hog')
        OT.update(faces)
        frame = OT.DrawObjects(frame)
        cv2.imshow("Face-Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    cv2.waitKey(1)

if __name__ == '__main__':
    Main()



                
        
        


