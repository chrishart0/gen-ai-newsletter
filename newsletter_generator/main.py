import json
from newsletter_generator.helpers.logger_helper import get_logger
from newsletter_generator.event_extractor.data_extractor import extract_news_items
from newsletter_generator.event_extractor.webpage_loader import load_webpage
from newsletter_generator.llm.llm_setup import llm

# Setup settings
from newsletter_generator.settings import Settings

settings = Settings()

# Get the configured logger
logger = get_logger()

# List of URLs to process
# TODO: make some method of finding the right newsletter issues every week, for now we update by hand
gen_ai_newsletter_urls = [
    "https://www.deeplearning.ai/the-batch/issue-277/",
    "https://www.deeplearning.ai/the-batch/issue-276/",
    # "https://www.reddit.com/r/LocalLLaMA/"
    # Add more URLs as needed
]


def generate_news_data():
    # Initialize a list to store all events
    all_news_items = []

    for url in gen_ai_newsletter_urls:
        loaded_webpage = load_webpage(url)
        events_list = extract_news_items(loaded_webpage, url)
        all_news_items.extend(
            events_list.news_items
        )  # Add individual events to the list

    # Get the pydantic objects into a list of dicts we can store in a JSON file
    serializable_events = [event.dict() for event in all_news_items]

    # Store events in a JSON file
    logger.info(
        f"Storing {len(all_news_items)} events in {settings.OUTPUT_DIRECTORY}events.json"
    )
    with open(f"{settings.OUTPUT_DIRECTORY}events.json", "w") as f:
        json.dump(serializable_events, f, indent=4, default=str)

    logger.info(f"Stored {len(all_news_items)} events in events.json")

    # Print each event
    for event in all_news_items:
        logger.info("--------------------------------")
        for field_name, field_value in event.dict().items():
            logger.info(f"{field_name.capitalize()}: {field_value}")


def generate_newsletter_markdown():
    logger.info("Generating newsletter markdown")

    # Load events from the JSON file
    with open(f"{settings.OUTPUT_DIRECTORY}events.json", "r") as f:
        events_data = json.load(f)

    # Sort events by date, handling None values by using a default date far in the past
    events_data.sort(key=lambda x: x.get("date") or "0000-01-01")

    # Assign the formatted string to the message
    messages = [
        (
            "system",
            """You are a newsletter writer for Gen AI engineers and consultants.
Generate a newsletter for the following news items I provide.
The newsletter should be written like a 30 year old technical consultant from a bespoke technical consultancy.
Create a subtitle for the newsletter, maximum 10 words, that describes the news items. The title will be applied later.
The news items should be related to AI, machine learning, and data science, with a focus on tooling like the OpenAI and Claude APIs, LangChain, and LlamaIndex.
Keep the newsletter exciting and engaging, but don't make up any news items.
Group the news items into sections with a short introductory paragraph for each section.
When discussing models, cite evals/benchmarks/etc., compare to other popular models as much as possible so people can easily understand the relative performance of the models.
Create a short executive summary at the top, maximum 4 sentences. In the executive summary, create an open loop, exciting readers, and giving a short hint at the most important topics. Someone should be able to read the executive summary get the most important headlines.
Be sure to cite sources for all news items by linking to the original source.
""",
        ),
        ("human", json.dumps(events_data)),
    ]

    # Pass the message to the LLM
    newsletter_author_response = llm.invoke(messages)

    # Write the newsletter content to a markdown file
    with open(f"{settings.OUTPUT_DIRECTORY}newsletter.md", "w") as f:
        f.write(newsletter_author_response.content)


if __name__ == "__main__":
    # generate_news_data()
    generate_newsletter_markdown()
