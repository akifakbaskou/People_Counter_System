import cv2
import numpy as np
import pandas as pd
import time
import math

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            cx, cy = self.center_points[object_id]
            new_center_points[object_id] = (cx, cy)

        self.center_points = new_center_points.copy()

        return objects_bbs_ids


class PeopleCounter():
    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.cap = cv2.VideoCapture(videoPath)
        self.person_id_counter = 1 
        self.people_count = 0
        self.people = {}

        # Frame üzerindeki çizgilerin y koordinatları 
        # Giriş çizgisi
        # Tüm videolar için aynı değerler kullanılabilir
        self.enter_line_y = 50

        # Çıkış çizgisi
        # Tüm videolar için otomatik değer atanmalı
        #self.exit_line_y = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 30
        self.exit_line_y = 250

        self.offset = 6 # İnsanların çizgilerden geçerken hata payı
        self.counter_enter = 0 
        self.counter_exit = 0

        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True) # shadows çıkarılabilir (fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False))
                                                         # backgroundsubstractora ait başka methodlarda kullanılabilir gmg , mog, mog2 vb.
        self.min_contour_width = 10
        self.min_contour_height = 50
        self.fgmask = None
        self.kernel = None
        self.opening = None
        self.closing = None
        self.dilation = None
        self.erosion = None

    def getCentroid(self, x, y, w, h):
        # İnsanların merkez noktalarını al
        cx = int((x + x + w) / 2.0)
        cy = int((y + y + h) / 2.0)
        return cx, cy

    def filterMask(self, frame):

        global threshold

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.fgmask = self.fgbg.apply(gray)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        self.opening = cv2.morphologyEx(self.fgmask, cv2.MORPH_OPEN, self.kernel, iterations=3)
        self.erosion = cv2.erode(self.opening, self.kernel, iterations=3)
        self.closing = cv2.morphologyEx(self.erosion, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        self.dilation = cv2.dilate(self.closing, self.kernel, iterations=2)
        _, threshold = cv2.threshold(self.dilation, 100, 255, cv2.THRESH_BINARY)

        return threshold
    
    def drawLines(self, frame):
        # İnsanların giriş yapacağı çizgiyi çiz
        cv2.line(frame, (0, self.enter_line_y), (frame.shape[1], self.enter_line_y), (255, 0, 0), 2)

        # İnsanların çıkış yapacağı çizgiyi çiz
        cv2.line(frame, (0, self.exit_line_y), (frame.shape[1], self.exit_line_y), (0, 0, 255), 2)

    def drawPeopleCount(self, frame):
        # İnsan sayısını yaz
        cv2.putText(frame, "Enter: {}".format(self.counter_enter), (310, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Exit: {}".format(self.counter_exit), (310, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def findContours(self, frame, threshold):

        centroids = [] 

        contours, hierarchy = cv2.findContours(
            threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_SIMPLE veya cv2.CHAIN_APPROX_TC89_L1 kullanılabilir
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # verilen değerden büyük olan alanlar insan kabul edilir. (İnsan tespiti için ilk filtre)
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = (w >= self.min_contour_width) and (
                    h >= self.min_contour_height)   # w ve h belirli değerler arasıysa insan kabul edilir. (İnsan tespiti için ikinci filtre)

                if not contour_valid:
                    continue
                    
                # İnsanların merkez noktasını al
                centroid = self.getCentroid(x, y, w, h)              

                # İnsanların merkez noktasını çiz
                cv2.circle(frame, centroid, 2, (0, 255, 0), 2)

                # İnsanları dikdörtgen ile çerçevele
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Merkez noktalarını listeye ekle
                centroids.append(np.array(centroid))

        return centroids
    
if __name__ == "__main__":
    video_path = "test_1.mp4"
    people_counter = PeopleCounter(video_path)
    tracker = Tracker()

    count = 0
    enter_line_y = 50
    exit_line_y = 250
    offset = 6

    def calcMovementDirection(people):
        directions = {}
        for id, people[id] in people.items():
            if len(people[id]) >= 2:
                y_diff = people[id][-2][1] - people[id][-1][1]
                direction = np.subtract(people[id][-2], people[id][-1])
                if direction[1] > 0 and y_diff > 0:
                    directions[id] = "down"
                elif direction[1] < 0 and y_diff < 0:
                    directions[id] = "up"
                else:
                    directions[id] = None
        
        return directions

    while True:
        ret, frame = people_counter.cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue
        
        threshold = people_counter.filterMask(frame)
        centroids = people_counter.findContours(frame, threshold)
        people_counter.drawLines(frame)
        people_counter.drawPeopleCount(frame)

        list = []

        for index, centroid in enumerate(centroids):
            
            # Merkez noktası alınan dikdörtgenin koordinatları
            x1, y1 = centroid[0] - 30, centroid[1] - 30
            x2, y2 = (x1 + 60), (y1 + 60)

            list.append([x1,y1,x2,y2])

        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int((x3 + x4) / 2.0)
            cy = int((y3 + y4) / 2.0)
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)               

            # ID'ye göre insanların merkez noktasını tut 
            if id not in people_counter.people:
                people_counter.people[id] = []
                people_counter.people[id].append((cx, cy))
            
            # İnsanların geçmiş merkez noktalarına göre hareket yönünü hesapla
            directions = calcMovementDirection(people_counter.people)

            # Yönlere göre sayacı arttır
            if id in directions and directions[id] == "down" and enter_line_y - offset < cy < enter_line_y + offset:
                people_counter.counter_enter += 1
                del people_counter.people[id]
            elif id in directions and directions[id] == "up" and exit_line_y - offset < cy < exit_line_y + offset:
                people_counter.counter_exit += 1
                del people_counter.people[id]
            else:
                people_counter.people[id].append((cx, cy))
            
            

        cv2.imshow("Frame", frame)
        cv2.imshow("Threshold", threshold)
        cv2.imshow("Opening", people_counter.opening)
        cv2.imshow("Closing", people_counter.closing)
        cv2.imshow("Dilation", people_counter.dilation)
        cv2.imshow("Erosion", people_counter.erosion)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    people_counter.cap.release()
    cv2.destroyAllWindows()







    


                          

