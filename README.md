https://youtu.be/7uv_BBNAC98

IA - Reconocimiento de Emociones con Juego Arcade  
Este proyecto implementa un sistema de detección de emociones en tiempo real mediante visión por computadora y redes neuronales convolucionales. El modelo entrenado con EfficientNetB0 analiza expresiones faciales y controla un mini juego arcade interactivo, donde las emociones del usuario influyen en el desarrollo del juego.  

Características principales  
- Reconocimiento facial en tiempo real mediante OpenCV.  
- Clasificación de emociones (feliz, triste, enojado, sorprendido, neutral, etc.) con TensorFlow/Keras.  
- Integración con un juego arcade controlado por las expresiones detectadas.  
- Soporte para GPU NVIDIA (aceleración CUDA cuando está disponible).  
- Modelo preentrenado (emotion_recognition_efficientnetb0_model.h5) incluido mediante Git LFS.
- 
Requisitos del sistema  
- Python 3.9 o superior  
- GPU NVIDIA (opcional, recomendado para entrenamiento)  
- Librerías principales: pip install tensorflow opencv-python numpy matplotlib  
Si tu sistema tiene GPU compatible con CUDA, instalá TensorFlow GPU: pip install tensorflow==2.18.0  
## Ejecución  
1. Clonar el repositorio: git clone https://github.com/mateoreca/IAReconocimientoEmocionesArcade.git && cd IAReconocimientoEmocionesArcade  
2. Instalar dependencias: pip install 
3. Ejecutar el proyecto: python proyecto.py  
4. Al iniciar, se activará la cámara y comenzará la detección en tiempo real. El juego responderá a las emociones detectadas.
   
Estructura del repositorio  
IAReconocimientoEmocionesArcade/  
├── proyecto.py  
├── emotion_recognition_efficientnetb0_model.h5  

Modelo utilizado  
El modelo se entrenó con EfficientNetB0, optimizando la precisión en la detección de emociones humanas. Fue guardado en formato .h5 y gestionado mediante Git LFS. Emociones clasificadas: Feliz, Neutral, Enojado, Triste, Sorprendido, Relajado.  
## Autor  
Mateo Rendón Cañon
Estudiante de Ingeniería de Sistemas  
Correo: mateorendonca@gmail.com  
