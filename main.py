import io
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from rembg import new_session, remove

app = FastAPI(title="Background Removal Service", version="1.0.0")

session = new_session("bria-rmbg")


@app.post("/remove-bg", response_model=None)
async def remove_background(
    file: UploadFile = File(..., description="Изображение (JPG/PNG/WebP)"),
    white_bg: bool = Query(False, description="Вернуть JPEG с белым фоном вместо прозрачного PNG"),
) -> StreamingResponse | dict[str, str]:
    """
    Удаляет фон с загруженного изображения и возвращает результат.

    :param file: Изображение для удаления фона
    :param white_bg: Флаг для заполнения фона белым цветом
    :return: Обработанное изображение с удалённым белым фоном
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        return {"error": "Поддерживаются только форматы JPEG, PNG, WebP"}

    input_image = await file.read()
    input_pil = Image.open(io.BytesIO(input_image)).convert("RGB")

    output_image = remove(
        input_pil,
        session=session,
        post_process_mask=True,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )

    if white_bg:
        background = Image.new("RGBA", output_image.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, output_image)
        final = composite.convert("RGB")
        img_bytes = io.BytesIO()
        final.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/jpeg")
    else:
        img_bytes = io.BytesIO()
        output_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/png")


@app.get("/")
def root() -> dict[str, str]:
    """Возвращает информацию по веб-сервису"""
    return {"message": "Background Removal API. POST /remove-bg with an image file."}


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """ВОзвращает информацию по статусу веб-сервиса"""
    return {
        "status": "healthy",
        "model_loaded": session is not None
    }
