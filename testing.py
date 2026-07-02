import cv2

# 1. Загрузка изображения или кадра с камеры
# Для Raspberry Pi с picamera2 можно использовать cv2.VideoCapture(0) или передавать numpy-массив напрямую
frame = cv2.imread("output\sections\section_14.png")
cv2.imshow("f", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Инициализация детектора (новый API для OpenCV >= 4.7.0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 3. Перевод в градации серого и детекция маркеров
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
corners, ids, rejected = detector.detectMarkers(gray)

# 4. Отрисовка и обработка результатов
if ids is not None:
    # Рисуем рамки и ID на изображении
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # Вычисляем центр каждого маркера
    for i, corner in enumerate(corners):
        c = corner[0] # 4 угла маркера
        cx = int(c[:, 0].mean())
        cy = int(c[:, 1].mean())
        print(f"Найден маркер ID: {ids[i][0]}, Центр: ({cx}, {cy})")
else:
    print("Маркеры не найдены")

cv2.imshow("Aruco Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()