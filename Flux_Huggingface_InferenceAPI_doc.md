# Python hugginface_club 
from huggingface_hub import InferenceClient
client = InferenceClient("black-forest-labs/FLUX.1-dev", token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

## output is a PIL.Image object
image = client.text_to_image("Astronaut riding a horse")

# Python requests format:

import requests

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "Astronaut riding a horse",
})

# You can access the image with PIL.Image for example
import io
from PIL import Image
image = Image.open(io.BytesIO(image_bytes))

# Java Script format:
async function query(data) {
	const response = await fetch(
		"https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev",
		{
			headers: {
				Authorization: "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.blob();
	return result;
}
query({"inputs": "Astronaut riding a horse"}).then((response) => {
	// Use image
});

# cURL format:
curl https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev \
	-X POST \
	-d '{"inputs": "Astronaut riding a horse"}' \
	-H 'Content-Type: application/json' \
	-H 'Authorization: Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'