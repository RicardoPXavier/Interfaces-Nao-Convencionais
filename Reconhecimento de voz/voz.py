#Ricardo de Paula Xavier - 2515750
#Leonardo Naime Lima - 2515660

import torch
from transformers import pipeline
import librosa
import os

# Configurações iniciais
ARQUIVO_AUDIO = "anuncio.wav"  # Nome do seu arquivo
IDIOMA = "portuguese"          # Idioma para transcrição
MODELO = "openai/whisper-small" # Modelo balanceado entre velocidade e precisão

def transcrever_audio():
    # Verifica se o arquivo existe
    if not os.path.exists(ARQUIVO_AUDIO):
        print(f"Erro: Arquivo '{ARQUIVO_AUDIO}' não encontrado!")
        print("Certifique-se que:")
        print(f"1. O arquivo '{ARQUIVO_AUDIO}' está na mesma pasta deste script")
        print("2. O nome está escrito exatamente como acima (incluindo .wav)")
        return

    # Configura o dispositivo (GPU se disponível)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nIniciando transcrição usando {dispositivo.upper()}...")

    try:
        # Carrega o modelo Whisper
        transcriber = pipeline(
            task="automatic-speech-recognition",
            model=MODELO,
            device=dispositivo
        )

        # Carrega o áudio e ajusta a taxa de amostragem
        audio, sr = librosa.load(ARQUIVO_AUDIO, sr=16000)

        # Transcreve o áudio
        print(f"\nProcessando: '{ARQUIVO_AUDIO}'...")
        resultado = transcriber(audio, generate_kwargs={"language": IDIOMA})
        texto_transcrito = resultado["text"]

        # Mostra o resultado
        print("\n--- TRANSCRIÇÃO ---")
        print(texto_transcrito)

        # Salva em arquivo
        nome_txt = os.path.splitext(ARQUIVO_AUDIO)[0] + "_transcricao.txt"
        with open(nome_txt, "w", encoding="utf-8") as f:
            f.write(texto_transcrito)
        print(f"\nTranscrição salva em: '{nome_txt}'")

    except Exception as e:
        print(f"\nErro durante a transcrição: {str(e)}")

if __name__ == "__main__":
    transcrever_audio()