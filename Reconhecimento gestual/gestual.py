# Ricardo de Paula Xavier - 2515750
# Leonardo Naime Lima - 2515660
import cv2
import numpy as np

# Inicializa a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espelha a imagem
    frame = cv2.flip(frame, 1)

    # Define a região de interesse
    roi = frame[80:300, 80:300]
    cv2.rectangle(frame, (80, 80), (300, 300), (0, 255, 0), 2)

    # Converte ROI para HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Intervalo de cor de pele em HSV (pode ajustar conforme iluminação)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    # Máscara e filtro
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)

        # Ignora ruído pequeno
        if cv2.contourArea(contour) > 2000:
            hull = cv2.convexHull(contour, returnPoints=False)

            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                count_defects = 0

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])

                        # Calcula distâncias
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        if b * c == 0:
                            continue

                        # Cálculo do ângulo com Lei dos Cossenos
                        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                        # Se for um defeito real e profundo
                        if angle <= np.pi / 2 and d > 5000:
                            count_defects += 1
                            cv2.circle(roi, far, 5, [0, 0, 255], -1)

                # Número de dedos
                fingers = count_defects + 1 if count_defects > 0 else 0

                # Classificação de gestos
                if fingers >= 4:
                    gesture = "Mão aberta"
                elif 1 <= fingers <= 3:
                    gesture = "Mão semi-aberta"
                else:
                    gesture = "Mão fechada"

                # Mostra o gesto na tela
                cv2.putText(frame, f"Gesto: {gesture}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibição
    cv2.imshow("Frame", frame)
    cv2.imshow("Mascara", mask)

    # Encerra com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
