# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from tool.config import Cfg
# from tool.predictor import Predictor
# from PIL import Image
# import io

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# config = Cfg.load_config_from_file('./config/vgg-transformer.yml')
# config['device'] = 'cpu'
# detector = Predictor(config)

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     img_data = await file.read()
#     img = Image.open(io.BytesIO(img_data))

#     result = detector.predict(img)

#     return JSONResponse({
#         "prediction": result,
#     })

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)