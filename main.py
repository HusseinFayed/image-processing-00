from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

import cv2
import numpy as np
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] if you want to be
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/denoise")
async def denoise_image(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Apply denoising
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=5, templateWindowSize=5, searchWindowSize=21)

    # Save to temp file
    _, temp_path = tempfile.mkstemp(suffix=".jpg")
    cv2.imwrite(temp_path, denoised)

    return FileResponse(temp_path, media_type="image/jpeg", filename="denoised.jpg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

handler = Mangum(app)  # This is required for Deta Space


# koyeb token
# petgu3lal8ka4o6ql20qnuv1lot63qrxto1g2g1agt6lidz2r1v5ghld6wipamr6

# to deploy on koyeb
# koyeb app init my-fastapi-app \
#   --git github.com/HUSSEINFAYED/image-processing \
#   --git-branch main \
#   --git-builder docker \
#   --instance-type free \
#   --regions fra \
#   --ports 8000:http \
#   --routes /:8000