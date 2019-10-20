import cv2
import numpy as np

# load yolo
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size = (len(classes), 3))
#loading images
img = cv2.imread("IMG-20181229-WA0070.jpg")
img = cv2.resize(img, None, fx=0.7, fy=0.7)
height, width, channels = img.shape

#detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(outputlayers)

#showing info on screen
boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #obj detected
            cen_x = int(detection[0]*width)
            cen_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            # cv2.circle(img, (cen_x, cen_y), 10, (0, 255, 0), 2)
            #rectangle coordinates
            x = int(cen_x - w/2)
            y = int(cen_y - h/2)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), font, 1, (0, 0, 0), 3)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
