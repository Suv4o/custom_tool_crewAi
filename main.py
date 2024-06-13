import os, io, base64, requests
from dotenv import dotenv_values
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from PIL import Image

env = dotenv_values(".env")

api_key = env["OPENAI_API_KEY"]
model = "gpt-4o"

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_MODEL_NAME"] = model


def encode_image(image_path, max_size=(2000, 2000)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)

        byte_arr = io.BytesIO()
        img.save(byte_arr, format="JPEG")

        return base64.b64encode(byte_arr.getvalue()).decode("utf-8")


@tool("Analyse Image")
def analyse_image(image_path: str) -> str:
    """Analyse the image and provide a detailed description on what is in the image"""
    base64_image = encode_image(image_path)
    local_image = f"data:image/jpeg;base64,{base64_image}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": local_image},
                    },
                ],
            }
        ],
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    response_json = response.json()

    choices = response_json["choices"]
    message = choices[0]["message"]
    content = message["content"]

    return content


image_describer_agent = Agent(
    role="Image Describer",
    goal="To describe the following {image} in a way that is useful to people who are blind or visually impaired.",
    verbose=True,
    memory=True,
    tools=[analyse_image],
    backstory=(
        "I am a computer program that has been trained to describe images"
        "in a way that is useful to people who are blind or visually impaired."
    ),
)

image_describer_task = Task(
    description=(
        "Compose an insightful and detailed description of the following image {image}"
    ),
    expected_output="A few sentences that describe the following image {image}. The description should be detailed and insightful.",
    agent=image_describer_agent,
)

crew = Crew(
    agents=[image_describer_agent],
    tasks=[image_describer_task],
    process=Process.sequential,
    memory=True,
    max_rpm=100,
)

result = crew.kickoff(inputs={"image": "image_1.jpg"})
print(result)
