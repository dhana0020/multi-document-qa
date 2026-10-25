import sounddevice as sd
import scipy.io.wavfile as wav
import whisper


def record_audio(
    filename="audio.wav",
    duration=5,
    sample_rate=16000
):
    print("\nRecording... Speak now!")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    wav.write(
        filename,
        sample_rate,
        audio
    )

    print("Recording completed!")


def speech_to_text(filename="audio.wav"):
    print("\nLoading Whisper model...")

    model = whisper.load_model("base")

    print("Converting speech to text...")

    result = model.transcribe(filename)

    text = result["text"].strip()

    print(f"\nRecognized question: {text}")

    return text