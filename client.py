import torch
import image

IMAGE_PATH = "./digit.jpg"
model = torch.load("./model.pkl")
ans, pred = image.predict(model=model, image=image.get_image(image_path=IMAGE_PATH))
print(ans)
