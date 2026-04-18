from PIL import Image
from io import BytesIO

import requests
import textwrap


def generate_caption(model, image, image_processor, device):
    """Generate a caption for a local image, array, or image URL."""
    model.eval()
    
    if isinstance(image, str):
        response = requests.get(image)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
    caption = model.generate(
        pixel_values,
        max_new_tokens=64,
        temperature=0.8,
        do_sample=True,
    )
    
    return caption, image

def wrap_text(text, width=40):
    """Wrap long text into multiple lines for display purposes."""
    return '\n'.join(textwrap.wrap(text, width=width))
