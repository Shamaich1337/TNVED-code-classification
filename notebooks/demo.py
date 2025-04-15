
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_PATH = os.path.abspath("best_model_distilbert") 
ENCODER_PATH = os.path.abspath("data\\label_encoder.joblib")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_tnved_code(text):
    """
    Функция принимает текст и возвращает предсказанный TNVED-код.
    
    :param text: str - входной текст для классификации
    :return: int - предсказанный TNVED-код
    """
    # Токенизируем текст
    inputs = tokenizer(
        text,
        return_tensors="pt",  # Возвращаем тензоры PyTorch
        truncation=True,      # Обрезаем текст до максимальной длины
        padding=True,         # Дополняем текст до максимальной длины
        max_length=128        # Максимальная длина последовательности
    )
    
    # Перемещаем данные на устройство (GPU или CPU)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Выполняем предсказание
    model.eval()  
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  
    
   
    predicted_class = torch.argmax(logits, dim=-1).item()
    code = label_encoder.inverse_transform([predicted_class])[0]
    return code


if __name__ == "__main__":
    
    input_text = "The ADMV1014 is a silicon germanium (SiGe) wideband microwave downconverter designed for point-to-point microwave radio applications. It operates within a frequency range of 24 GHz to 44 GHz and provides two modes of frequency translation: direct quadrature demodulation to baseband I/Q output signals and image rejection downconversion to a complex intermediate frequency (IF) output. The device is programmable via a 4-wire SPI interface and is housed in a compact 32-terminal, 5 mm × 5 mm Land Grid Array (LGA) package."
    
    # Получаем предсказанный TNVED-код
    predicted_code = predict_tnved_code(input_text)
    print(f"Предсказанный TNVED-код: {predicted_code}")