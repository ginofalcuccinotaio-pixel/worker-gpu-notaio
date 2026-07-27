import os
import sys
import subprocess
import json
import requests
from google.cloud import storage
from faster_whisper import WhisperModel

# ==========================================
# CONFIGURACIÓN
# ==========================================

# Variables de entorno (inyectadas por el backend)
BUCKET_NAME = os.environ.get("BUCKET_NAME", "notaio-clinical-sessions")
SESSION_ID = os.environ.get("SESSION_ID")
T_PATH = os.environ.get("T_PATH")  # Path del .ogg terapeuta
T_START = os.environ.get("T_START")  # Timestamp inicio terapeuta (nanosegundos)
P_PATH = os.environ.get("P_PATH")  # Path del .ogg paciente
P_START = os.environ.get("P_START")  # Timestamp inicio paciente (nanosegundos)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
IS_PRESENTIAL = os.environ.get("IS_PRESENTIAL", "false").lower() == "true"

print(f"🚀 GPU Worker iniciado para Sesión: {SESSION_ID}")
print(f"📂 Bucket: {BUCKET_NAME}")
if IS_PRESENTIAL:
    print(f"🎤 Modo: PRESENCIAL")
    print(f"📁 Audio: {T_PATH}")
else:
    print(f"💻 Modo: VIRTUAL")
    print(f"👨‍⚕️ Terapeuta: {T_PATH} (start: {T_START})")
    print(f"🧑‍⚕️ Paciente: {P_PATH} (start: {P_START})")

# ==========================================
# UTILIDADES DE STORAGE
# ==========================================

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Descarga un archivo desde Google Cloud Storage"""
    print(f"⬇️ Descargando: {source_blob_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    
    # Validación anti-zombies
    file_size = os.path.getsize(destination_file_name)
    if file_size == 0:
        raise ValueError(f"Archivo descargado está vacío: {source_blob_name}")
    print(f"✅ Descargado: {file_size} bytes")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Sube un archivo a Google Cloud Storage"""
    print(f"⬆️ Subiendo: {destination_blob_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"✅ Subido correctamente")

# ==========================================
# PASO 1: FUSIÓN DE AUDIO (.ogg → .flac)
# ==========================================

def merge_ogg_to_flac(terapeuta_path, paciente_path, t_start_ns, p_start_ns, output_path):
    """
    Fusiona dos archivos .ogg en un .flac estéreo sincronizado.
    - Canal L (izquierda): Terapeuta
    - Canal R (derecha): Paciente
    
    Port de la lógica desde webnotaio-FFmpeg/main.py
    """
    print("🔧 === FUSIÓN DE AUDIO ===")
    
    # Calcular offset temporal (nanosegundos → milisegundos)
    try:
        t_start = int(t_start_ns)
        p_start = int(p_start_ns)
    except (ValueError, TypeError):
        print("⚠️ Timestamps no válidos, usando sincronización exacta")
        t_start = 0
        p_start = 0
    
    diff = abs(t_start - p_start)
    delay_ms = int(diff / 1000000)  # nanosegundos a milisegundos
    
    print(f"⏱️ Diferencia detectada: {delay_ms}ms")
    
    # Construir filtro FFmpeg
    if t_start < p_start:
        print("💡 Terapeuta empezó primero. Retrasando Paciente.")
        filter_complex = (
            f"[1:a]adelay={delay_ms}|{delay_ms}[p_delayed];"
            f"[0:a]pan=stereo|c0=c0[left];"
            f"[p_delayed]pan=stereo|c1=c0[right];"
            f"[left][right]amix=inputs=2:duration=longest[a]"
        )
    elif p_start < t_start:
        print("💡 Paciente empezó primero. Retrasando Terapeuta.")
        filter_complex = (
            f"[0:a]adelay={delay_ms}|{delay_ms}[t_delayed];"
            f"[t_delayed]pan=stereo|c0=c0[left];"
            f"[1:a]pan=stereo|c1=c0[right];"
            f"[left][right]amix=inputs=2:duration=longest[a]"
        )
    else:
        print("💡 Sincronización exacta.")
        filter_complex = (
            f"[0:a]pan=stereo|c0=c0[left];"
            f"[1:a]pan=stereo|c1=c0[right];"
            f"[left][right]amix=inputs=2:duration=longest[a]"
        )
    
    # Ejecutar FFmpeg
    command = [
        "ffmpeg", "-y",
        "-i", terapeuta_path,
        "-i", paciente_path,
        "-filter_complex", filter_complex,
        "-map", "[a]",
        "-ac", "2",       # Forzar estéreo
        "-ar", "16000",   # 16kHz (estándar de voz para IA)
        output_path
    ]
    
    print(f"🛠️ Ejecutando FFmpeg...")
    try:
        subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=300
        )
        print(f"✅ Fusión completada: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error FFmpeg: {e.stderr.decode()}")
        raise

# ==========================================
# PASO 2.5: CONVERSIÓN A MP3
# ==========================================

def convert_flac_to_mp3(input_flac, output_mp3):
    """
    Convierte un archivo FLAC a MP3 para consumo en frontend.
    """
    print("🎵 Convirtiendo a MP3...")
    
    cmd = [
        "ffmpeg", "-y", "-i", input_flac,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",  # Calidad VBR estándar (aprox 190kbps)
        output_mp3
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✅ Conversión completada: {output_mp3}")

# ==========================================
# PASO 2.6: CONVERSIÓN WEBM A WAV (PRESENCIAL)
# ==========================================

def convert_webm_to_wav(input_webm, output_wav):
    """
    Convierte un archivo WebM a WAV mono para Whisper.
    Usado en modo presencial.
    """
    print("🎧 Convirtiendo WebM a WAV...")
    
    cmd = [
        "ffmpeg", "-y", "-i", input_webm,
        "-ar", "16000",  # 16kHz (estándar de voz)
        "-ac", "1",      # Mono
        output_wav
    ]
    
    subprocess.run(cmd, check=True)
    print(f"✅ Conversión completada: {output_wav}")

# ==========================================
# PASO 3: SEPARACIÓN DE CANALES
# ==========================================

def split_stereo_channels(input_file):
    """
    Separa un archivo estéreo en dos archivos mono.
    - left.wav: Canal izquierdo (Terapeuta)
    - right.wav: Canal derecho (Paciente)
    """
    print("✂️ Separando canales estéreo...")
    
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-map_channel", "0.0.0", "left.wav",
        "-map_channel", "0.0.1", "right.wav"
    ]
    
    subprocess.run(cmd, check=True)
    print("✅ Canales separados: left.wav, right.wav")
    return "left.wav", "right.wav"

# ==========================================
# PASO 3: TRANSCRIPCIÓN CON WHISPER
# ==========================================

def transcribe_with_whisper(audio_path, role):
    """
    Transcribe un archivo de audio usando Faster-Whisper.
    Retorna segmentos con timestamps a nivel palabra.
    
    Guardrails anti-alucinación activos:
    - condition_on_previous_text=False : rompe bucles en silencios clínicos prolongados
    - temperature como lista            : fallback progresivo si la decodificación falla
    - compression_ratio_threshold       : descarta texto repetitivo (síntoma de loop)
    - log_prob_threshold                : descarta segmentos de baja confianza
    - no_speech_threshold               : filtra silencios y ruido de fondo
    - initial_prompt                    : calibra vocabulario clínico/argentino desde el inicio
    """
    print(f"🎙️ Transcribiendo {role} con Whisper {WHISPER_MODEL}...")
    
    # Cargar modelo (se cachea automáticamente)
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="float16")
    
    # Prompt clínico: calibra el vocabulario inicial del modelo.
    # Incluye terminología psicoanalítica, jerga argentina y contexto del encuadre
    # para reducir errores de transcripción en términos técnicos y lunfardo rioplatense.
    prompt_clinico = (
        "Psicología clínica, psicoanálisis, terapia cognitivo-conductual, psiquiatría. "
        "Terapeuta, paciente, sesión, consultorio, encuadre, setting. "
        "Angustia, ansiedad, depresión, melancolía, duelo, trauma, acting out. "
        "Transferencia, contratransferencia, inconsciente, represión, sublimación, resistencia. "
        "Superyó, yo, ello, pulsión, deseo, goce, fantasma, síntoma. "
        "Asociación libre, interpretación, señalamiento, intervención, corte de sesión. "
        "Che, vos, dale, mirá, sabés, te digo, no sé, es que, o sea, igual, "
        "capaz, tipo, re, posta, ni idea, onda, obvio, nada que ver. "
        "Buenos Aires, Argentina. Hablan dos personas: un terapeuta y su paciente."
    )
    
    # Transcribir con parámetros de alta precisión y guardrails anti-alucinación.
    # CRÍTICO: condition_on_previous_text=False es el fix principal para los bucles.
    # Sin él, Whisper con temperature=0.0 se retroalimenta de su propio output
    # durante los silencios y genera texto repetitivo o inventado de forma infinita.
    segments, info = model.transcribe(
        audio_path,
        language="es",
        beam_size=5,
        word_timestamps=True,                         # CRÍTICO para diarización
        vad_filter=True,                              # Filtro de detección de voz activo
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Fallback progresivo si falla la confianza
        condition_on_previous_text=False,             # CRÍTICO: previene bucles en silencios clínicos
        compression_ratio_threshold=2.0,              # Descarta texto repetitivo (loops)
        log_prob_threshold=-1.0,                      # Descarta segmentos de baja confianza
        no_speech_threshold=0.4,                      # Filtra silencios y ruido de fondo
        initial_prompt=prompt_clinico                 # Vocabulario clínico/argentino
    )
    
    print(f"   Idioma detectado: {info.language} (prob: {info.language_probability:.2f})")
    
    # Convertir a lista (generator)
    segments_list = list(segments)
    print(f"✅ {role}: {len(segments_list)} segmentos transcritos")
    
    return segments_list

# ==========================================
# PASO 4: DIARIZACIÓN
# ==========================================

def whisper_segments_to_diarization(terapeuta_segments, paciente_segments):
    """
    Convierte los segmentos de Whisper en un diálogo diarizado.
    
    DIFERENCIA CON PARSER ORIGINAL:
    - Google Speech API: estructura con results/alternatives/channelTag
    - Whisper: lista de segmentos con words y timestamps directos
    
    Este es un parser simplificado que aprovecha que Whisper ya nos da
    transcripciones separadas por canal.
    """
    print("🧩 Generando diarización...")
    
    dialogo = []
    
    # Procesar segmentos del terapeuta
    for segment in terapeuta_segments:
        if not segment.words:
            continue
        
        # Usar el timestamp de la primera palabra como inicio del segmento
        start_time = segment.words[0].start if segment.words else segment.start
        
        dialogo.append({
            "tiempo": f"{int(start_time//60):02}:{int(start_time%60):02}",
            "hablante": "Terapeuta",
            "texto": segment.text.strip(),
            "segundos_exactos": round(start_time, 2)
        })
    
    # Procesar segmentos del paciente
    for segment in paciente_segments:
        if not segment.words:
            continue
        
        start_time = segment.words[0].start if segment.words else segment.start
        
        dialogo.append({
            "tiempo": f"{int(start_time//60):02}:{int(start_time%60):02}",
            "hablante": "Paciente",
            "texto": segment.text.strip(),
            "segundos_exactos": round(start_time, 2)
        })
    
    # Ordenar por tiempo
    dialogo.sort(key=lambda x: x['segundos_exactos'])
    
    print(f"✅ Diarización completada: {len(dialogo)} intervenciones")
    return dialogo

# ==========================================
# PASO 4.5: DIARIZACIÓN CON PYANNOTE (PRESENCIAL)
# ==========================================

def diarize_with_pyannote(audio_path, segments):
    """
    Usa pyannote.audio para identificar speakers en el audio.
    
    Proceso:
    1. Ejecuta pyannote speaker diarization sobre el audio completo
    2. Obtiene segmentos por speaker (SPEAKER_00, SPEAKER_01, etc.)
    3. Cuenta intervenciones por speaker
    4. Asigna "Terapeuta" al speaker con MENOS intervenciones
    5. Mapea los segmentos de Whisper a los speakers identificados
    
    Args:
        audio_path: Ruta al archivo de audio WAV
        segments: Segmentos de Whisper con texto y timestamps
    
    Returns:
        Lista de diálogos con formato: [{tiempo, hablante, texto, segundos_exactos}]
    """
    print("🎯 Diarización con pyannote.audio...")
    
    from pyannote.audio import Pipeline
    import torch
    
    # Cargar modelo de diarización
    # NOTA: Requiere token de HuggingFace (variable de entorno HUGGINGFACE_TOKEN)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("❌ HUGGINGFACE_TOKEN no configurado. Obtén uno en: https://huggingface.co/settings/tokens")
    
    try:
        # ACTUALIZACIÓN: token= reemplaza use_auth_token= (deprecado en huggingface-hub>=0.22.0)
        # El requirements.txt ya tiene huggingface-hub==0.23.2, por lo que este cambio
        # elimina FutureWarnings y garantiza compatibilidad con versiones futuras.
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except TypeError:
            print("   ⚠️ Pipeline no reconoce 'token', intentando con 'use_auth_token'...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        
        # Mover a GPU si está disponible
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print("   ✅ Pipeline pyannote cargado en GPU")
        else:
            print("   ⚠️ GPU no disponible, usando CPU (será más lento)")
        
        # CRÍTICO: num_speakers=2 fuerza el clustering a exactamente 2 grupos.
        # Sin este parámetro, pyannote puede detectar ruidos ambientales del consultorio
        # (sillas, puerta, aire acondicionado) como un tercer hablante, rompiendo
        # la asignación de roles Terapeuta/Paciente. En el encuadre clínico presencial
        # siempre hay exactamente 2 personas: el parámetro es semánticamente correcto.
        print("   🔍 Procesando audio (Forzando 2 hablantes)...")
        diarization = pipeline(audio_path, num_speakers=2)
        
        # Extraer información de speakers
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        print(f"   ✅ Detectados {len(speaker_segments)} segmentos de voz")
        
        # Contar intervenciones por speaker
        speaker_counts = {}
        for seg in speaker_segments:
            speaker = seg["speaker"]
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        print(f"   📊 Conteo de intervenciones: {speaker_counts}")
        
        # Identificar terapeuta (el que tiene MENOS intervenciones)
        if len(speaker_counts) < 2:
            print("   ⚠️ Se detectó solo un speaker, usando asignación por defecto")
            speaker_to_role = {list(speaker_counts.keys())[0]: "Terapeuta"}
        else:
            # Ordenar por número de intervenciones (ascendente)
            sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1])
            terapeuta_speaker = sorted_speakers[0][0]  # El que tiene MENOS
            paciente_speaker = sorted_speakers[1][0]   # El que tiene MÁS
            
            speaker_to_role = {
                terapeuta_speaker: "Terapeuta",
                paciente_speaker: "Paciente"
            }
            
            print(f"   🎭 Asignación de roles:")
            print(f"      {terapeuta_speaker} → Terapeuta ({speaker_counts[terapeuta_speaker]} intervenciones)")
            print(f"      {paciente_speaker} → Paciente ({speaker_counts[paciente_speaker]} intervenciones)")
        
        # Mapear segmentos de Whisper a speakers de pyannote
        dialogo = []
        for segment in segments:
            if not segment.words:
                continue
            
            start_time = segment.words[0].start if segment.words else segment.start
            
            # Buscar a qué speaker corresponde este timestamp
            assigned_speaker = "Desconocido"
            for pyannote_seg in speaker_segments:
                if pyannote_seg["start"] <= start_time <= pyannote_seg["end"]:
                    assigned_speaker = speaker_to_role.get(pyannote_seg["speaker"], "Desconocido")
                    break
            
            # Si no se encontró match exacto, usar el speaker más cercano
            if assigned_speaker == "Desconocido":
                closest_seg = min(
                    speaker_segments,
                    key=lambda x: min(abs(x["start"] - start_time), abs(x["end"] - start_time))
                )
                assigned_speaker = speaker_to_role.get(closest_seg["speaker"], "Desconocido")
            
            dialogo.append({
                "tiempo": f"{int(start_time//60):02}:{int(start_time%60):02}",
                "hablante": assigned_speaker,
                "texto": segment.text.strip(),
                "segundos_exactos": round(start_time, 2)
            })
        
        # Ordenar por tiempo
        dialogo.sort(key=lambda x: x['segundos_exactos'])
        
        print(f"   ✅ Diarización completada: {len(dialogo)} intervenciones")
        return dialogo
        
    except Exception as e:
        print(f"   ❌ Error en pyannote: {e}")
        raise

# ==========================================
# PASO 4.6: DIARIZACIÓN CON GEMINI (FALLBACK)
# ==========================================

def diarize_with_gemini(segments):
    """
    [DEPRECATED - USA SOLO COMO FALLBACK]
    Usa Gemini 3.1 Flash Lite para identificar quién habla en cada segmento.
    
    NOTA: Esta función se mantiene solo como fallback de emergencia.
    El método principal es diarize_with_pyannote().
    
    Entrada: Lista de segmentos de Whisper (sin información de hablante)
    Salida: Dialogo con formato standard [{tiempo, hablante, texto, segundos_exactos}]
    """
    print("🤖 Diarización con Gemini 3.1 Flash Lite...")
    
    from google import genai
    
    # Inicializar cliente de GenAI
    client = genai.Client(vertexai=True, project="webnotaio", location="global")
    
    # Construir transcripción completa con timestamps
    transcript_lines = []
    for i, segment in enumerate(segments):
        start_time = segment.start
        tiempo_formato = f"{int(start_time//60):02}:{int(start_time%60):02}"
        transcript_lines.append(f"[{tiempo_formato}] {segment.text.strip()}")
    
    transcript_text = "\n".join(transcript_lines)
    
    # Prompt para Gemini
    prompt = f'''Eres un asistente de diarización especializado en sesiones clínicas de salud mental.

Identifica quién habla en cada línea de esta transcripción:
- "Terapeuta": El profesional de salud mental (psicólogo, psiquiatra, consejero)
- "Paciente": La persona que busca ayuda

Transcripción:
{transcript_text}

Responde SOLO con un array JSON válido (sin markdown, sin ```json):
[
  {{"tiempo": "00:01", "hablante": "Terapeuta", "texto": "...", "segundos_exactos": 1.23}},
  {{"tiempo": "00:05", "hablante": "Paciente", "texto": "...", "segundos_exactos": 5.67}}
]

IMPORTANTE:
- Mantén el texto EXACTO de cada línea
- Usa los mismos timestamps
- Solo añade el campo "hablante"'''
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        response_text = response.text.strip()
        
        # Limpiar markdown si viene envuelto
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0]
        
        # Parsear JSON
        diarized = json.loads(response_text)
        
        print(f"✅ Gemini diarizó {len(diarized)} intervenciones")
        
        # Validación: asegurar que todos los campos estén presentes
        for item in diarized:
            if 'segundos_exactos' not in item and 'tiempo' in item:
                # Calcular segundos_exactos desde tiempo si falta
                tiempo_parts = item['tiempo'].split(':')
                mins = int(tiempo_parts[0])
                secs = int(tiempo_parts[1]) if len(tiempo_parts) > 1 else 0
                item['segundos_exactos'] = mins * 60 + secs
        
        return diarized
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parseando respuesta de Gemini: {e}")
        print(f"Respuesta recibida: {response_text[:500]}")
        
        # Fallback: crear diarización básica sin IA
        print("🔄 Usando diarización fallback (alterna Terapeuta/Paciente)")
        dialogo = []
        for i, segment in enumerate(segments):
            hablante = "Terapeuta" if i % 2 == 0 else "Paciente"
            dialogo.append({
                "tiempo": f"{int(segment.start//60):02}:{int(segment.start%60):02}",
                "hablante": hablante,
                "texto": segment.text.strip(),
                "segundos_exactos": round(segment.start, 2)
            })
        return dialogo

# ==========================================
# FLUJO PRINCIPAL
# ==========================================

def main():
    """Flujo principal del worker - soporta modo VIRTUAL y PRESENCIAL"""
    
    #Validaciones básicas
    if not SESSION_ID:
        print("❌ Error: SESSION_ID es obligatorio")
        sys.exit(1)
    
    try:
        if IS_PRESENTIAL:
            # ==========================================
            # FLU JO PRESENCIAL
            # ==========================================
            print("\n" + "🎤"*30)
            print("MODO PRESENCIAL ACTIVADO")
            print("🎤"*30 + "\n")
           
            # Archivos temporales
            local_audio = "/tmp/presential_audio.webm"
            local_wav = "/tmp/presential_audio.wav"
            local_mp3 = "/tmp/session_final.mp3"
            
            # === PASO 1: DESCARGAR ARCHIVO .WEBM ===
            print("\n" + "="*60)
            print("PASO 1: DESCARGA DE AUDIO PRESENCIAL")
            print("="*60)
            download_blob(BUCKET_NAME, T_PATH, local_audio)
            
            # === PASO 2: CONVERTIR WEBM A WAV ===
            print("\n" + "="*60)
            print("PASO 2: CONVERSIÓN A WAV")
            print("="*60)
            convert_webm_to_wav(local_audio, local_wav)
            
            # === PASO 3: TRANSCRIBIR CON WHISPER ===
            print("\n" + "="*60)
            print("PASO 3: TRANSCRIPCIÓN CON WHISPER")
            print("="*60)
            print(f"🧠 Cargando modelo Whisper {WHISPER_MODEL} en {WHISPER_DEVICE}...")
            
            segments = transcribe_with_whisper(local_wav, "Presencial")
            
            # === PASO 4: DIARIZACIÓN CON PYANNOTE ===
            print("\n" + "="*60)
            print("PASO 4: DIARIZACIÓN CON PYANNOTE")
            print("="*60)
            try:
                dialogo_final = diarize_with_pyannote(local_wav, segments)
            except Exception as e:
                print(f"⚠️ Error en pyannote, intentando fallback con Gemini: {e}")
                # Fallback a Gemini si pyannote falla (por si acaso)
                dialogo_final = diarize_with_gemini(segments)
            
            # === PASO 5: CONVERTIR A MP3 ===
            print("\n" + "="*60)
            print("PASO 5: CONVERSIÓN A MP3")
            print("="*60)
            convert_flac_to_mp3(local_audio, local_mp3)  # Reutilizamos función existente
            
            # === PASO 6: GUARDAR RESULTADO ===
            print("\n" + "="*60)
            print("PASO 6: PERSISTENCIA")
            print("="*60)
            
            output_json = "/tmp/diarized_transcript.json"
            with open(output_json, "w", encoding='utf-8') as f:
                json.dump(dialogo_final, f, ensure_ascii=False, indent=2)
            
            print(f"💾 JSON generado: {len(dialogo_final)} líneas")
            
            # Subir a GCS (JSON)
            dest_blob_json = f"transcripts/{SESSION_ID}/diarized_transcript.json"
            upload_blob(BUCKET_NAME, output_json, dest_blob_json)

            # Subir a GCS (MP3)
            dest_blob_mp3 = f"processed/{SESSION_ID}/audio.mp3"
            upload_blob(BUCKET_NAME, local_mp3, dest_blob_mp3)
            
            # === LIMPIEZA ===
            temp_files = [local_audio, local_wav, local_mp3, output_json]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"🗑️ Eliminado: {f}")
            
        else:
            # ==========================================
            # FLUJO VIRTUAL (ORIGINAL)
            # ==========================================
            print("\n" + "💻"*30)
            print("MODO VIRTUAL ACTIVADO")
            print("💻"*30 + "\n")
            
            # Validaciones para modo virtual
            if not all([T_PATH, P_PATH, T_START, P_START]):
                print("❌ Error: Faltan variables de entorno para modo virtual")
                print(f"   T_PATH: {T_PATH}")
                print(f"   P_PATH: {P_PATH}")
                print(f"   T_START: {T_START}")
                print(f"   P_START: {P_START}")
                sys.exit(1)
            
            # Archivos temporales
            local_t_ogg = "/tmp/terapeuta.ogg"
            local_p_ogg = "/tmp/paciente.ogg"
            local_flac = "/tmp/session_merged.flac"
            local_mp3 = "/tmp/session_final.mp3"
            
            # === PASO 1: DESCARGAR ARCHIVOS .OGG ===
            print("\n" + "="*60)
            print("PASO 1: DESCARGA DE ARCHIVOS")
            print("="*60)
            download_blob(BUCKET_NAME, T_PATH, local_t_ogg)
            download_blob(BUCKET_NAME, P_PATH, local_p_ogg)
            
            # === PASO 2: FUSIONAR EN .FLAC ESTÉREO ===
            print("\n" + "="*60)
            print("PASO 2: FUSIÓN DE AUDIO")
            print("="*60)
            merge_ogg_to_flac(
                local_t_ogg, 
                local_p_ogg, 
                T_START, 
                P_START, 
                local_flac
            )

            # === PASO 2.5: CONVERTIR A MP3 ===
            print("\n" + "="*60)
            print("PASO 2.5: CONVERSIÓN MP3")
            print("="*60)
            convert_flac_to_mp3(local_flac, local_mp3)
            
            # === PASO 3: SEPARAR CANALES ===
            print("\n" + "="*60)
            print("PASO 3: SEPARACIÓN DE CANALES")
            print("="*60)
            left_wav, right_wav = split_stereo_channels(local_flac)
            
            # === PASO 4: TRANSCRIBIR CON WHISPER ===
            print("\n" + "="*60)
            print("PASO 4: TRANSCRIPCIÓN CON WHISPER")
            print("="*60)
            print(f"🧠 Cargando modelo Whisper {WHISPER_MODEL} en {WHISPER_DEVICE}...")
            
            terapeuta_segments = transcribe_with_whisper(left_wav, "Terapeuta")
            paciente_segments = transcribe_with_whisper(right_wav, "Paciente")
            
            # === PASO 5: GENERAR DIARIZACIÓN ===
            print("\n" + "="*60)
            print("PASO 5: DIARIZACIÓN")
            print("="*60)
            dialogo_final = whisper_segments_to_diarization(
                terapeuta_segments, 
                paciente_segments
            )
            
            # === PASO 6: GUARDAR RESULTADO ===
            print("\n" + "="*60)
            print("PASO 6: PERSISTENCIA")
            print("="*60)
            
            output_json = "/tmp/diarized_transcript.json"
            with open(output_json, "w", encoding='utf-8') as f:
                json.dump(dialogo_final, f, ensure_ascii=False, indent=2)
            
            print(f"💾 JSON generado: {len(dialogo_final)} líneas")
            
            # Subir a GCS (JSON)
            dest_blob_json = f"transcripts/{SESSION_ID}/diarized_transcript.json"
            upload_blob(BUCKET_NAME, output_json, dest_blob_json)

            # Subir a GCS (MP3)
            dest_blob_mp3 = f"processed/{SESSION_ID}/audio.mp3"
            upload_blob(BUCKET_NAME, local_mp3, dest_blob_mp3)
            
            # === LIMPIEZA ===
            temp_files = [
                local_t_ogg, local_p_ogg, local_flac, local_mp3,
                left_wav, right_wav, output_json
            ]
            
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"🗑️ Eliminado: {f}")
        
        # ==========================================
        # PASOS COMUNES (VIRTUAL Y PRESENCIAL)
        # ==========================================
        
        # NOTIFICACIÓN AL BACKEND (El "Handshake")
        print("\n" + "="*60)
        print("📡 NOTIFICACIÓN AL BACKEND")
        print("="*60)

        # Intentar obtener la URL del entorno, fallback a la URL conocida
        BACKEND_URL = os.environ.get("BACKEND_URL", "https://notaio-backend-1007838680332.us-central1.run.app")
        webhook_endpoint = f"{BACKEND_URL}/webhook/transcription-ready/{SESSION_ID}"

        # Calcular duración aproximada desde el diálogo
        duration = 0
        if dialogo_final:
            duration = dialogo_final[-1].get('segundos_exactos', 0)

        try:
            payload = {"duration": int(duration)}
            response = requests.post(webhook_endpoint, json=payload, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ Backend notificado con éxito (Duración: {duration}s): {response.json()}")
            else:
                print(f"⚠️ El Backend respondió con error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Error crítico contactando al Backend: {e}")
        
        # === FINALIZACIÓN EXITOSA ===
        print("\n" + "="*60)
        print("✅ PROCESAMIENTO COMPLETADO CON ÉXITO")
        print("="*60)
        print(f"📊 JSON disponible en: gs://{BUCKET_NAME}/{dest_blob_json}")
        print(f"🎵 MP3 disponible en: gs://{BUCKET_NAME}/{dest_blob_mp3}")
        print(f"🎉 Sesión {SESSION_ID} procesada correctamente")
        print(f"🔧 Modo: {'PRESENCIAL' if IS_PRESENTIAL else 'VIRTUAL'}")
        
        # La VM/Container se auto-destruirá al salir
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR FATAL")
        print("="*60)
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        
        import traceback
        print("\nStack Trace:")
        traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":
    main()