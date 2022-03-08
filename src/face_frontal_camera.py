import cv2

# Setando a classificacao do xml da OpenCV
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Captura WebCam
capture = cv2.VideoCapture(0)

while True:
    ## Mantendo camera e frames
    conected, frame = capture.read()

    ## Transformando frame em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Detectando a face com o classificador treinado xml
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 210), 2)

    cv2.imshow('Video', frame)

    ##Fechar janela
    if cv2.waitKey(1) == ord('s'):
        break

## Liberar a captura
capture.release()
## Liberar memória
cv2.destroyAllWindows()
