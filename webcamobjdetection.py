import cv2
import numpy as np
import time

# load yolo
net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size = (len(classes), 3))
#loading images
cap = cv2.VideoCapture(0)
starting_time = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    #detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

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
            if confidence > 0.4:
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255, 255, 255), 3)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 100), font, 3, (0, 0, 0), 1)
    cv2.imshow("Image", frame)
    key  = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
