from typing import Optional, List
from datetime import datetime
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
        description="The title of the news item, give enough context so that the reader knows what the news item is about."
    )
    importance: int = Field(
        description="The importance of the news item, on a scale of 1(low) to 5(high)."
    )
    category: Optional[str] = Field(
        description="The category of the news, such as 'AI', 'Foundational Models', 'agents', 'public news', 'policy', 'open source', 'LLMs', etc."
    )
    # source: Optional[str] = Field(description="The source publication or website of the news item.")
    publish_date: Optional[datetime] = Field(
        description="The date the news item was published."
    )
    author: Optional[str] = Field(description="The author of the news item.")
    summary: Optional[str] = Field(
        description="A summary of the news item, containing all relevant details."
    )
    full_text: Optional[str] = Field(
        description="The full text of the news item, including all relevant details."
    )
    image_links: Optional[List[str]] = Field(
        description="Links to images related to the news item."
    )
    source_link: Optional[str] = Field(
        description="Link to the news source which this item was extracted from."
    )
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
            "Each news item should include a title, source link, importance, category, source, publish date, author, summary, link, and image links. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
        (
            "system",
            "Example: If the text is 'AI Daily: New AI model released by OpenAI (source: OpenAI, publish date: 2022-01-01, author: John Doe, summary: OpenAI has released a new AI model...)', "
            "you should extract the following news item: "
            "title: 'New AI model released by OpenAI', "
            "source link: null, "
            "importance: null, "
            "category: null, "
            "source: 'OpenAI', "
            "publish date: '2022-01-01', "
            "author: 'John Doe', "
            "summary: 'OpenAI has released a new AI model...', "
            "link: null, "
            "image links: null.",
        ),
        ("human", "{text}"),
    ]
)

runnable = prompt | llm.with_structured_output(schema=NewsItemsList)


def extract_news_items(
    text: str,
    original_source_url: str,
) -> NewsItemsList:
    # Log in green text that the news items are being extracted
    logger.info("\033[92mExtracting news items from text\033[0m")
    extracted_data = runnable.invoke({"text": text})
    # Log in green text that the news items were extracted
    logger.info(f"\033[92mExtracted {len(extracted_data.news_items)} news items\033[0m")

    # Add the original source URL to each news item
    for news_item in extracted_data.news_items:
        news_item.source_link = original_source_url

        # Convert date string to datetime object
        if news_item.publish_date and isinstance(news_item.publish_date, str):
            if (
                len(news_item.publish_date) == 7
            ):  # If the publish_date string is in 'YYYY-MM' format
                news_item.publish_date += "-01"  # Add the day
            news_item.publish_date = datetime.strptime(
                news_item.publish_date, "%Y-%m-%d"
            )

    return extracted_data
