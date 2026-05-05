# 1. BASE: Usamos la imagen oficial de PyTorch con CUDA 12.1 que detectamos en el historial.
# Esta imagen ya tiene los drivers necesarios para que Whisper use la GPU T4 de Google Cloud.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 2. METADATA: Evita que las instalaciones se detengan pidiendo confirmación.
ENV DEBIAN_FRONTEND=noninteractive

# 3. DIRECTORIO DE TRABAJO: Donde vivirá la lógica de Notaio.
WORKDIR /app

# 4. DEPENDENCIAS DE SISTEMA: 
# FFmpeg: El corazón de la sincronización y conversión de audio clínico.
# Git: Necesario por si alguna librería (como Faster-Whisper o Pyannote) se baja de un repo.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 5. CONFIGURACIÓN NVIDIA: Asegura que el contenedor vea la GPU al arrancar.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 6. INSTALACIÓN DE LIBRERÍAS DE IA:
# Copiamos primero el requirements para aprovechar el "cache" de Docker.
# Si solo cambias el código, este paso no se repite, ahorrando minutos de build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. EL CEREBRO: Copiamos el script de lógica que recuperamos.
# Recordá que este es el archivo donde vas a aplicar los fixes de la v2.12.
COPY gpu_worker.py .

# 8. EJECUCIÓN: El comando que inicia el procesamiento discursivo.
CMD ["python", "gpu_worker.py"]
