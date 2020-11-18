import cv2
import numpy as np

cap = cv2.VideoCapture("Astronauts.mp4")

whT = 224

confidenceThreshold = 0.5

nmsThreshold = 0.3

classesFile = "classes.names"
classes = []

with open(classesFile, 'rt') as f:
    classNames = f.read().strip("\n").split("\n")

# print(classNames)

modelConfiguration = "yolov3_custom.cfg"

modelWeights = "yolov3_custom_best.weights"

net = cv2.dnn.readNet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def findObjects(outputs, img):
    hT = img.shape[0]
    wT = img.shape[1]
    # print(hT)
    # print(wT)
    bbox = []
    classIds = []
    confs = []
    for out in outputs:
        for det in out:
            scores = det[5:]
            # print(scores)
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                centre_x = int(det[0] * wT)
                centre_y = int(det[1] * hT)
                width = int(det[2] * wT)
                height = int(det[3] * hT)
                left = int(centre_x - width / 2)
                top = int(centre_y - height / 2)
                classIds.append(classId)
                confs.append(float(confidence))
                bbox.append([left, top, width, height])

    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confidenceThreshold, nmsThreshold)
    # print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        centre_x, centre_y, width, height = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (centre_x, centre_y), (centre_x+width, centre_y+height), (0, 225, 0), 2)
        cv2.putText(image, f"{classNames[classIds[i]].upper() } {int(confs[i]*100)}", (centre_x, centre_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    r, image = cap.read()

    blob = cv2.dnn.blobFromImage(image, 1/255, (whT, whT), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    # print(net.getUnconnectedOutLayers())
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames[0])

    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[0][0])
    findObjects(outputs, image)
    cv2.imshow("prev", image)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):
        break