import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # GORUNTUMUZU CAPTURE EDECEK

pTime = 0 # fps için koyuldu. PREVIOUS TIME 0 DAN BAŞLAMASI LAZIM


mpDraw = mp.solutions.drawing_utils# YÜZÜMÜZE BİR MESH EKLEMEK İÇİN YAZILMIŞ KODLAR -START, dijital çizim için gerekli şeyler.
mpFaceMesh = mp.solutions.face_mesh  # classı dahil et
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)# YÜZÜMÜZE BİR MESH EKLEMEK İÇİN YAZILMIŞ KODLAR -STOP, classtan obje oluştur
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1) # opsiyonel özellik çizgilerin kalınlığı, noktaların kalınlığı vs.

while True: # BİR DÖNGÜ İÇİNDE OLMA ŞARTI VAR
    success, img = cap.read() # GÖRÜNTÜMÜZÜ OKUYACAK

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # GÖRÜNTÜMÜZ BİZE BGR OLARAK GELİYOR BUNU CONVERT ETMEMİZ LAZIM, ÇÜNKÜ RGB OLARAK İŞLENMESİ ZORUNLUDUR.

    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec,drawSpec)  # neyi çizicekse onu veriyoruz yani bizim img'mizi çizicek ve neye göre çizicegini landmarkla belirliyoruz.
                                                      # drawSpec bizim kendi belirlediğimiz özellikler olmasa da olur yani
            for id, lm in enumerate(faceLms.landmark): # BURDAKİ KODLRA BİZE INFO VERİYOR Pixel olarak koordinat veriyor bir nevi -start
                ih, iw , ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*iw)
                print(id,x, y) # stop


    # FPS -START
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,
                f'fps : {int(fps)}'
                , (20,70)
                , cv2.FONT_HERSHEY_PLAIN
                , 3,
                (245, 123 ,222)
                , 2)
    # FPS - STOP


    cv2.imshow("Image", img) # BİZE GÖRÜNTÜYÜ GÖSTERİCEK
    cv2.waitKey(1) # FRAME RATEYİ AYARLAMAK

