from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from processing import load_model, generate
from schemas import GenerationResponse
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image

# Инициализация приложения с метаданными для документации
app = FastAPI(
    title="Image to Anime Image API",
    description="Получи аниме версию своего изображения с помощью GAN.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели при старте сервера (выполняется один раз)
model = load_model()

# Маршрут для проверки работоспособности сервиса
@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "Image to Anime is running!"}


# Основной маршрут для получения рекомендаций
@app.post(
    "/api/generate/",
    summary="Наложить на изображение аниме стиль",
    description="Принимает изображение, пропускает его через модель и возвращает преобразованное изображение в аниме-стиле.",
    tags=["Generation"],
    response_class=StreamingResponse,
    responses={
        200: {"content": {"image/png": {}}, "description": "Успешное преобразование"},
        422: {"model": GenerationResponse, "description": "Ошибка валидации"},
    },
)
async def get_generation(
    file: UploadFile = File(..., description="Входное изображение (jpeg/png/webp и т.п.)")
):
    if not file.content_type.startswith("image/") and not file.filename.lower().endswith(('.heic', '.heif', '.avif')):
        raise HTTPException(status_code=422, detail="Файл не является поддерживаемым изображением.")

    try:
        file_bytes = await file.read() # Читаем байты загруженного файла
        input_image = Image.open(BytesIO(file_bytes)).convert("RGB") # Открываем как PIL.Image (из памяти)

        output_pil = generate(model=model, input_image=input_image, use_imagenet=False)

        # Сериализуем в буфер - в память (PNG)
        buf = BytesIO()
        output_pil.save(buf, format="PNG")
        buf.seek(0)

        # Отправляем клиенту как image/png
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="anime_{file.filename or "output"}"'
            },
        )

    except HTTPException:
        # пробрасываем наши HTTP-ошибки наверх
        raise
    except Exception as e:
        # Логика на случай непредвиденной ошибки
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {e}")