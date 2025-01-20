import cv2
import mediapipe as mp

# MediaPipe’in eller (Hands) ile ilgili modüllerini hazırlıyoruz
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# El tespiti için parametreleri ayarlayabilirsiniz.
# Bu örnekte tek el veya iki el arasında otomatik ayrım yapıyor.
hands = mp_hands.Hands(
    static_image_mode=False,        # Video akışı olduğu için False
    max_num_hands=2,               # Algılanacak maksimum el sayısı
    min_detection_confidence=0.7,   # Tespit başarısı için minimum güven
    min_tracking_confidence=0.6     # Takip başarısı için minimum güven
)

# Bilgisayarınızın kamerasını açın (0, varsayılan kamera)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Kamera açılamadı veya görüntü okunamadı.")
        break
    
    # Renk formatını (BGR -> RGB) dönüştürüyoruz 
    # çünkü MediaPipe, RGB formatında görüntü istiyor.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Görüntüyü MediaPipe ile işleyelim
    results = hands.process(image)
    
    # Tekrar BGR’ye dönüştürelim (OpenCV ile gösterirken BGR istenir)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Eğer elde tespit edilmiş landmark(ler) varsa
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Landmarkları (parmak-eklem konumlarını) çizelim
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
    
    # Ekrana yansıtalım
    cv2.imshow("El Takibi (MediaPipe)", image)
    
    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kullanım sonrası her şeyi serbest bırakın
cap.release()
cv2.destroyAllWindows()
