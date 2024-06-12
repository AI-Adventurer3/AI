
from transformers import pipeline

classifier = pipeline("image-classification", model="trpakov/vit-face-expression")

image = "happy.jpeg"

result = classifier(image)

print(result)