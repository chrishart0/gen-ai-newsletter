from typing import Optional, List

# from datetime import datetime
from pydantic import BaseModel, Field
from newsletter_generator.helpers.logger_helper import get_logger
from langchain_core.prompts import ChatPromptTemplate

from newsletter_generator.settings import Settings
from newsletter_generator.llm.llm_setup import llm


settings = Settings()

# Get the configured logger
logger = get_logger()


class NewsItem(BaseModel):
    """Details about a technical news item intended for Gen AI engineers, enthusiasts, and consultants.
    One source may have multiple news items, especially if it is a newsletter or aggregator site.
    """

    title: str = Field(
        description="The title of the news item, give enough context so that the reader knows what the news item is about. Include meaningful numbers in the title, if possible."
    )
    # importance: int = Field(
    #     description="The importance of the news item, on a scale of 1(low) to 5(high)."
    # )
    category: Optional[str] = Field(
        description="The category of the news, such as 'AI', 'Foundational Models', 'agents', 'public news', 'policy', 'open source', 'LLMs', etc."
    )
    # source: Optional[str] = Field(description="The source publication or website of the news item.")
    # publish_date: Optional[datetime] = Field(
    #     description="The date the news item was published."
    # )
    summary: Optional[str] = Field(
        description="A summary of the news item, containing all relevant details."
    )
    full_text: Optional[str] = Field(
        description="The full text of the news item, including all relevant details."
    )
    image_links: Optional[List[str]] = Field(
        description="Links to images related to the news item."
    )
    # source_link: Optional[str] = Field(
    #     description="Link to the news source which this item was extracted from."
    # )
    # citations_links: Optional[List[dict]] = Field(description="Links to citations within the news item, if provided.")


class NewsItemsList(BaseModel):
    news_items: List[NewsItem]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Extract multiple news items from the text. "
            "If you do not know the value of an attribute asked to extract return null for the attribute's value.",
        ),
        (
            "system",
            "Example: If the text is 'AI Daily: Meta releases Llama 3.1...)', "
            "you should extract the following news item: "
            "title: 'First open weights model competitive with OpenAI! Llama 3.1 released by Meta, new 405b model evals even with closed models', "
            "...",
        ),
        ("human", "{text}"),
    ]
)

runnable = prompt | llm.with_structured_output(schema=NewsItemsList)


class NewsItemWithSource(NewsItem):
    source_link: str


def extract_news_items(
    text: str,
    original_source_url: str,
) -> NewsItemsList:
    # Log in green text that the news items are being extracted
    logger.info("\033[92mExtracting news items from text\033[0m")
    extracted_data = runnable.invoke({"text": text})
    # Log in green text that the news items were extracted
    logger.info(f"\033[92mExtracted {len(extracted_data.news_items)} news items\033[0m")

    # Convert each NewsItem to a NewsItemWithSource
    news_items_with_source = [
        NewsItemWithSource(**news_item.dict(), source_link=original_source_url)
        for news_item in extracted_data.news_items
    ]

    # Replace the old list of news items with the new one
    extracted_data.news_items = news_items_with_source

    return extracted_data
