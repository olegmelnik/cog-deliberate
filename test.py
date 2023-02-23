import os
import predict

MODEL = predict.Predictor()
MODEL.setup()

img_paths = MODEL.predict(
    prompt="Cat in space",
    negative_prompt="",
    prompt_strength=0.8,
    guidance_scale=7.5,
    width=512,
    height=512,
    num_outputs=2,
    num_inference_steps=10,
    seed=int.from_bytes(os.urandom(2), "big"),
    scheduler="K-LMS"
)

print(img_paths)