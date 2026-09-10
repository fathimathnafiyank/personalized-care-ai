import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification


# ==============================
# MODEL PATH
# ==============================

MODEL_PATH = "models/wav2vec2_emotion_model_deployment"

print("Loading emotion recognition model...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully!")


# ==============================
# EMOTION PREDICTION
# ==============================

def predict_emotion(audio_path):

    audio, sampling_rate = librosa.load(
        audio_path,
        sr=16000
    )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_id = torch.argmax(
        probabilities
    ).item()

    predicted_emotion = model.config.id2label[
        predicted_id
    ]

    confidence = probabilities[
        predicted_id
    ].item() * 100


    # ==============================
    # SHOW ALL PROBABILITIES
    # ==============================

    print("\nAll emotion probabilities:")

    for i, probability in enumerate(probabilities):

        emotion = model.config.id2label[i]

        print(
            f"{emotion:10s}: "
            f"{probability.item() * 100:.2f}%"
        )


    return predicted_emotion, confidence


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    audio_file = "test/audio1s.wav"

    emotion, confidence = predict_emotion(
        audio_file
    )

    print()
    print("==============================")
    print("   EMOTION PREDICTION")
    print("==============================")
    print(f"Emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}%")