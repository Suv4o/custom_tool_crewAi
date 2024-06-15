import os, io, base64, requests
from dotenv import dotenv_values
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool, SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from PIL import Image

env = dotenv_values(".env")

api_key = env["OPENAI_API_KEY"]
model_key = "gpt-4o"
serper_key = env["SERPER_API_KEY"]

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_MODEL_NAME"] = model_key
os.environ["SERPER_API_KEY"] = serper_key

search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()


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
        "model": model_key,
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


instagram_post_agent = Agent(
    role="Instagram Post Writer",
    goal="To create engaging Instagram posts based on the provided {image}."
    "The post should be engaging and interesting, include relevant emojis, and have a personal touch."
    "The caption should not exceed 2000 characters including hashtags."
    "The post should have a personal touch, for example using 'I' in a context 'I took this image...' or 'I visited this place...'.",
    memory=True,
    tools=[analyse_image],
    verbose=False,
    backstory=(
        "I am a computer program that has been trained to create engaging Instagram posts"
        "based on the provided image."
    ),
)

hashtag_research_agent = Agent(
    role="Hashtag Research Analyst",
    goal="To review and analyse the effectiveness of hashtags used in social media posts"
    "and suggest improvements for better reach and engagement.",
    memory=True,
    tools=[search_tool, web_rag_tool],
    verbose=False,
    backstory=(
        "I am a hashtag guardian to review hashtags"
        "and provide suggestions for improvement."
    ),
)

location_research_agent = Agent(
    role="Location Research Analyst",
    goal="To research and provide detailed information about the location where the image was taken."
    "The image is titled {image_title}.",
    memory=True,
    tools=[search_tool, web_rag_tool],
    verbose=False,
    backstory=(
        "I am a location research analyst, skilled in researching locations"
        "based on the information provided in the image title."
    ),
)

instagram_post_review_agent = Agent(
    role="Instagram Post Reviewer",
    goal="To review Instagram posts and ensure they are focused on the following {image} and its contents,"
    "and that the caption does not exceed 2000 characters including hashtags."
    "Also ensure that the post is engaging and interesting and has relevant emojis."
    "The post also needs to ensure has a personal touch, for example using 'I' in a context 'I took this image...' or 'I visited this place...'.",
    memory=True,
    tools=[analyse_image],
    verbose=False,
    backstory=(
        "I am a computer program that has been trained to review Instagram posts"
        "and ensure they meet the specified criteria."
    ),
)

output_format_agent = Agent(
    role="Output Formatter",
    goal="To ensure the final output contains only the caption of the Instagram post.",
    memory=True,
    verbose=False,
    backstory=(
        "I am a computer program that has been trained to format text outputs"
        "and ensure they meet the specified format."
    ),
)

instagram_post_task = Task(
    description=(
        "Compose an engaging Instagram post using the following image {image}"
    ),
    expected_output="An Instagram post using the following image {image}."
    "The post should be engaging and interesting."
    "Should include emojis where appropriate to make the post more fun and engaging."
    "And also include relevant hashtags to increase the reach of the post."
    "The caption should have a personal touch, for example using 'I' in a context 'I took this image...' or 'I visited this place...'.",
    agent=instagram_post_agent,
)

hashtag_research_task = Task(
    description="Analyse the hashtags and provide suggestions for improvement.",
    expected_output="A list of hashtags with suggestions for improvement.",
    agent=hashtag_research_agent,
)

location_research_task = Task(
    description=(
        "Research and provide detailed information about the location where the image titled {image_title} was taken."
    ),
    expected_output="Detailed information about the location where the image titled {image_title} was taken.",
    agent=location_research_agent,
)

instagram_post_review_task = Task(
    description=(
        "Review the Instagram post and ensure it is focused on the following {image} and its contents. "
        "Also, ensure that the caption does not exceed 2000 characters including hashtags."
    ),
    expected_output="A review report indicating whether the Instagram post meets the specified criteria.",
    agent=instagram_post_review_agent,
)

output_format_task = Task(
    description=(
        "Format the final output to ensure it contains only the caption of the Instagram post. Noting else should be included. Exclude any titles or markdowns."
    ),
    expected_output="only the caption of the Instagram post.",
    agent=output_format_agent,
)

crew = Crew(
    agents=[
        instagram_post_agent,
        hashtag_research_agent,
        location_research_agent,
        instagram_post_review_agent,
        output_format_agent,
    ],
    tasks=[
        instagram_post_task,
        hashtag_research_task,
        location_research_task,
        instagram_post_review_task,
        output_format_task,
    ],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o"),
    full_output=False,
    verbose=False,
    memory=True,
    max_rpm=100,
)

result = crew.kickoff(
    inputs={
        "image": "wilson_promontory_national_park_whisky_bay.jpg",
        "image_title": "wilson_promontory_national_park_whisky_bay",
    }
)
print(result)
