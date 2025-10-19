# Grupo formado por Samuel Ibañez y Mateo Rendon

# Link Dataset
# https://www.kaggle.com/datasets/msambare/fer2013

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
import numpy as np
import cv2
import os
import time
import random
import tkinter as tk
from threading import Thread

# ===================== CONFIGURACIÓN DE GPU =====================
print("\n========== Verificando disponibilidad de GPU ==========\n")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ GPU detectada: {gpus[0].name}")
        print(f"Total de GPUs disponibles: {len(gpus)} (lógicas: {len(logical_gpus)})\n")
    except RuntimeError as e:
        print(f"⚠️ Error al inicializar la GPU: {e}")
else:
    print("⚠️ No se detectó GPU. TensorFlow usará la CPU.\n")

print("Versión de TensorFlow:", tf.__version__)
print("=======================================================\n")
# ===============================================================

# Definir rutas de entrenamiento y prueba
train_dir = r"C:\Users\mateo\OneDrive\Ing Sistemas\Codigos\6\IA\3\Proyecto\train"
test_dir = r"C:\Users\mateo\OneDrive\Ing Sistemas\Codigos\6\IA\3\Proyecto\test"

# Verificar si el modelo ya existe
model_path = r"C:\Users\mateo\OneDrive\Ing Sistemas\Codigos\6\IA\3\Proyecto\emotion_recognition_efficientnetb0_model.h5"
if os.path.exists(model_path):
    print("Modelo encontrado. Cargando modelo existente...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Modelo no encontrado. Entrenando el modelo desde cero...\n")
    
    # Preprocesamiento de las imágenes
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2, rotation_range=10)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=64,
        class_mode='categorical',
        color_mode='grayscale'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=64,
        class_mode='categorical',
        color_mode='grayscale'
    )

    # Crear el modelo utilizando EfficientNetB0
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(48, 48, 1))
    
    # Agregar capas personalizadas para la clasificación de emociones
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks para optimizar el entrenamiento
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Entrenar el modelo (100 épocas)
    model.fit(train_generator,
              validation_data=test_generator,
              epochs=50,
              callbacks=callbacks)

    # Guardar el modelo entrenado
    model.save(model_path)
    print("\n✅ Modelo entrenado y guardado correctamente.")

# Definir las clases de emociones
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Inicializar variables para el juego y la dificultad
start_time = time.time()
difficulty = "Normal"
difficulty_mapping = {
    "Easy": ["Sad", "Surprise", "Angry"],
    "Normal": ["Neutral"],
    "Hard": ["Happy", "Disgust", "Fear"]
}

# Función para mostrar el juego tipo arcade en una ventana separada
def arcade_game():
    root = tk.Tk()
    root.title("Juego Arcade")
    canvas = tk.Canvas(root, width=400, height=400, bg="black")
    canvas.pack()

    player = canvas.create_rectangle(180, 380, 220, 400, fill="white")
    stars = []
    score = 3
    score_text = canvas.create_text(10, 10, anchor="nw", text=f"Puntuación: {score}", fill="white", font=("Helvetica", 16))
    timer_text = canvas.create_text(10, 30, anchor="nw", text=f"Cambiando de dificultad en: 5 segundos", fill="white", font=("Helvetica", 16))
    game_over = False
    player_speed = 0

    def move_player():
        nonlocal player_speed
        if not game_over:
            x1, _, x2, _ = canvas.coords(player)
            if (x1 > 0 and player_speed < 0) or (x2 < 400 and player_speed > 0):
                canvas.move(player, player_speed, 0)
            root.after(20, move_player)

    def key_press(event):
        nonlocal player_speed
        if event.keysym == 'Left':
            player_speed = -10
        elif event.keysym == 'Right':
            player_speed = 10

    def key_release(event):
        nonlocal player_speed
        if event.keysym in ['Left', 'Right']:
            player_speed = 0

    def create_star():
        if not game_over:
            x = random.randint(10, 390)
            star_color = "green" if difficulty == "Easy" else "purple" if difficulty == "Normal" else "red"
            star = canvas.create_oval(x, 0, x+10, 10, fill=star_color)
            stars.append(star)
            root.after(1000, create_star)

    def move_stars():
        nonlocal score, game_over
        if game_over:
            return
        for star in stars[:]:
            speed = 5 if difficulty == "Easy" else 7 if difficulty == "Normal" else 15
            canvas.move(star, 0, speed)
            star_coords = canvas.coords(star)
            player_coords = canvas.coords(player)
            if star_coords[3] > 400:
                canvas.delete(star)
                stars.remove(star)
                x, y = star_coords[0], 380
                loss_text = canvas.create_text(x, y, text="-1", fill="red", font=("Helvetica", 16))
                root.after(1000, lambda: canvas.delete(loss_text))
                score -= 1
                canvas.itemconfig(score_text, text=f"Puntuación: {score}")
                if score <= 0:
                    game_over = True
                    canvas.create_text(200, 200, text="Game Over", fill="red", font=("Helvetica", 32))
                    restart_button = tk.Button(root, text="Reiniciar Juego", command=restart_game)
                    canvas.create_window(200, 250, window=restart_button)
                    return
            elif (player_coords[0] < star_coords[2] and player_coords[2] > star_coords[0] and
                  player_coords[1] < star_coords[3] and player_coords[3] > star_coords[1]):
                canvas.delete(star)
                stars.remove(star)
                score += 1
                canvas.itemconfig(score_text, text=f"Puntuación: {score}")
        if score > 0:
            root.after(50, move_stars)

    def update_timer():
        if not game_over:
            elapsed_time = int(time.time() - start_time)
            remaining_time = 5 - (elapsed_time % 5)
            canvas.itemconfig(timer_text, text=f"Cambiando de dificultad en: {remaining_time} segundos")
            root.after(1000, update_timer)

    def update_colors():
        background_color = "yellow" if difficulty == "Easy" else "blue" if difficulty == "Normal" else "black"
        star_color = "green" if difficulty == "Easy" else "purple" if difficulty == "Normal" else "red"
        canvas.config(bg=background_color)
        for star in stars:
            canvas.itemconfig(star, fill=star_color)
        root.after(500, update_colors)

    def restart_game():
        nonlocal score, game_over, stars
        score = 3
        game_over = False
        stars = []
        canvas.delete("all")
        player = canvas.create_rectangle(180, 380, 220, 400, fill="white", tags="player")
        score_text = canvas.create_text(10, 10, anchor="nw", text=f"Puntuación: {score}", fill="white", font=("Helvetica", 16))
        timer_text = canvas.create_text(10, 30, anchor="nw", text=f"Cambiando de dificultad en: 5 segundos", fill="white", font=("Helvetica", 16))
        create_star()
        move_stars()
        move_player()
        update_timer()
        update_colors()

    root.bind("<KeyPress>", key_press)
    root.bind("<KeyRelease>", key_release)
    create_star()
    move_stars()
    move_player()
    update_timer()
    update_colors()
    root.mainloop()

thread = Thread(target=arcade_game)
thread.daemon = True
thread.start()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Iniciando detección de emociones en tiempo real y juego arcade...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_emotion = ""
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray, verbose=0)
        max_index = int(np.argmax(prediction))
        current_emotion = emotion_labels[max_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if time.time() - start_time > 5:
        if current_emotion:
            for key, emotions in difficulty_mapping.items():
                if current_emotion in emotions:
                    difficulty = key
                    break
        start_time = time.time()

    cv2.putText(frame, f"Dificultad: {difficulty}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cambiando de dificultad en: {5 - int(time.time() - start_time) % 5} segundos", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Juego Arcade - Recolecta las estrellas!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Emotion Detection and Arcade Game', frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n⛔ Ejecución detenida por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()
