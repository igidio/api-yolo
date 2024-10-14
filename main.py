import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

@app.get("/")
def read_root():
   return "Hola desde FastApi!"

@app.post("/classify")
async def classify_image(file: list[UploadFile]):

    model = YOLO("yolo11n-cls.pt")

    allowed_types = ["image/jpeg", "image/png", "image/gif"]
    max_file_size = 10 * 1024 * 1024 # 10MB


    for f in file:
        if f.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido.")
        if f.size > max_file_size:
            raise HTTPException(status_code=400, detail=f"Archivo demasiado grande: {f.filename}")

    if ( len(file) > 1 ):        
        images = [Image.open(io.BytesIO(f.file.read())) for f in file]
        classify = model.predict(images)
        embeddings = [result.verbose() for result in classify]
        joined_results = "".join(embeddings)

        return filter_by_repeated_name( data_to_dict(joined_results) )

    image = Image.open(io.BytesIO(file[0].file.read()))
    classify = model.predict(image, conf=0.4)
    return data_to_dict(classify[0].verbose())

def data_to_dict(data):
    parts = data.split(",")
    result = []
    for part in parts:
        if part.strip():
            name, number = part.strip().split()
            result.append({
                "name": name, 
                "conf": float(number)
            })
    return result

def filter_by_repeated_name(data):
    seen_names = set()
    unique_data = []
    
    for item in data:
        if item["name"] not in seen_names:
            unique_data.append(item)
            seen_names.add(item["name"])
    return unique_data

def filter_by_confidence(data, threshold):
    return [item for item in data if item["conf"] < threshold]