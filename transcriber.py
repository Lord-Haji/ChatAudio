import os
import speech_recognition as sr
from pydub import AudioSegment
from database import create_database, fetch_transcription, cache_transcription


def convert_mp3_to_wav(mp3_file_path):
    print("Converting to wav...")
    audio = AudioSegment.from_mp3(mp3_file_path)
    wav_file_path = os.path.splitext(mp3_file_path)[0] + '.wav'
    audio.export(wav_file_path, format="wav")
    print("Conversion done")
    return wav_file_path

def audio_to_text(file_path):
    create_database()
    
    r = sr.Recognizer()

    
    file_extension = os.path.splitext(file_path)[1]
    
    if file_extension == '.mp3':
        file_path = convert_mp3_to_wav(file_path)
    
    transcript = fetch_transcription(file_path)

    if transcript:
        print("Found transcript in cache")
        return transcript

    print("Starting Transcription")
    
    with sr.AudioFile(file_path) as source:

        # r.adjust_for_ambient_noise(source)

        audio_data = r.record(source)
        transcript = r.recognize_whisper_api(audio_data)
    
    cache_transcription(file_path, transcript)
    return transcript


def main():
    print(audio_to_text("cancel_2.mp3"))

if __name__ == "__main__":
    main()