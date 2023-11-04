from flask import Flask
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
from collections import deque
import re
from multiprocessing import Pool
from functools import lru_cache
import logging
from html.parser import HTMLParser
import tiktoken
import openai
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

openai.api_key = "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"

HTTP_URL_PATTERN = r"^http[s]{0,1}://.+$"

logging.basicConfig(filename="app.log", level=logging.INFO)

with open("./config.json", "r") as f:
    config = json.load(f)

applications = config["applications"]

if len(applications) > 0:
    application = applications[0]
    app_id = application["app_id"]
    domain = application["website_domain"]
    full_url = application["website_url"]
else:
    app_id = ""
    domain = ""
    full_url = ""

app_text_folder_path = ""


class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])


def get_hyperlinks(url):
    try:
        response = requests.get(url)
        if not response.headers.get("Content-Type", "").startswith("text/html"):
            return []

        html = response.content.decode("utf-8")
    except Exception as e:
        logging.error(f"Error fetching URL: {url}, {str(e)}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    clean_links = []

    for link in soup.find_all("a", href=True):
        href = link.get("href")
        if href:
            if (
                href.startswith("#")
                or href.startswith("mailto:")
                or href.startswith("tel:")
            ):
                continue
            if not re.search(HTTP_URL_PATTERN, href):
                href = (
                    urlparse(url).scheme
                    + "://"
                    + urlparse(url).netloc
                    + "/"
                    + href.lstrip("/")
                )

            if href in application["website_urls"] and not href.endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".pdf")
            ):
                clean_links.append(href)

    return clean_links


def get_domain_hyperlinks(local_domain, url):
    clean_links = []

    for link in set(get_hyperlinks(url)):
        clean_link = None

        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                if url_obj.path.endswith((".html", ".htm")):
                    clean_link = link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    return list(set(clean_links))


def extract_urls_from_sitemap(sitemap_url):
    try:
        response = requests.get(sitemap_url)
        if response.status_code == 200:
            sitemap = response.content
            root = ET.fromstring(sitemap)
            urls = [
                elem.text
                for elem in root.iter(
                    "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                )
            ]
            return urls
        else:
            logging.error(f"Failed to retrieve sitemap: {response.status_code}")
    except requests.RequestException as e:
        logging.error(f"Error retrieving sitemap: {str(e)}")
    return []


def crawl_single_url(url, local_domain):
    logging.info(f"Crawling URL: {url}")

    with open(
        f"text/{app_id}/{domain}/{url[8:].replace('/', '_')}.txt", "w", encoding="UTF-8"
    ) as f:
        try:
            source_url_comment = f"Source URL: {url}"
            f.write(source_url_comment + "\n")
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            text = soup.get_text()

            if "You need to enable JavaScript to run this app." in text:
                logging.warning(
                    f"Unable to parse page {url} due to JavaScript being required"
                )

            f.write(text)
        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}")


def crawl(urls):
    global app_text_folder_path

    if not os.path.exists("text/"):
        os.mkdir("text/")

    app_text_folder_path = os.path.join("text", app_id, domain)
    if not os.path.exists(app_text_folder_path):
        os.makedirs(app_text_folder_path)

    visited = set()

    for url in urls:
        local_domain = urlparse(url).netloc
        queue = deque([url])
        seen = set([url])

        while queue:
            url = queue.pop()

            if url in application["website_urls"]:
                continue

            crawl_single_url(url, local_domain)

            hyperlinks = get_domain_hyperlinks(local_domain, url)
            hyperlinks_set = set(hyperlinks)

            new_links = hyperlinks_set.difference(seen, application["website_urls"])
            queue.extend(new_links)
            seen |= hyperlinks_set

            visited |= new_links

    for url in application["website_urls"]:
        if url not in visited:
            crawl_single_url(url, urlparse(url).netloc)


def parallel_crawl(urls):
    global app_text_folder_path

    if not os.path.exists("text/"):
        os.mkdir("text/")

    app_text_folder_path = os.path.join("text", app_id, domain)
    if not os.path.exists(app_text_folder_path):
        os.makedirs(app_text_folder_path)

    extracted_urls = []
    sitemap_url = application["sitemap_url"]
    website_urls = application["website_urls"]

    if sitemap_url != "N/A" and website_urls:
        logging.error(
            "Error: Both sitemap_url and website_urls found for the application."
        )
        return

    if sitemap_url == "N/A" and not website_urls:
        logging.error(
            "Error: No sitemap_url or website_urls found for the application."
        )
        return

    if sitemap_url != "N/A":
        extracted_urls = extract_urls_from_sitemap(sitemap_url)
        if not extracted_urls:
            logging.info("No URLs extracted from the sitemap.")
    else:
        extracted_urls = website_urls

    for url in extracted_urls:
        local_domain = urlparse(url).netloc
        crawl_single_url(url, local_domain)

    scrapper_urls_folder_path = os.path.join("ScrappedURLs", app_id)
    if not os.path.exists(scrapper_urls_folder_path):
        os.makedirs(scrapper_urls_folder_path)

    file_path = os.path.join(scrapper_urls_folder_path, "url.txt")
    with open(file_path, "w") as f:
        f.write("\n".join(extracted_urls))

    logging.info(
        "URLs extracted from sitemap or website_urls and saved in ScrappedURLs folder."
    )


def remove_newlines(serie):
    serie = serie.str.replace("\n", " ", regex=True)
    serie = serie.str.replace("\\n", " ", regex=True)
    serie = serie.str.replace("  ", " ", regex=True)
    serie = serie.str.replace("  ", " ", regex=True)
    return serie


def create_text_data():
    global app_text_folder_path

    if not os.path.exists("text/"):
        os.mkdir("text/")

    app_text_folder_path = os.path.join("text", app_id, domain)
    if not os.path.exists(app_text_folder_path):
        os.makedirs(app_text_folder_path)

    sitemap_url = application["sitemap_url"]
    website_urls = application["website_urls"]

    if sitemap_url != "N/A" and website_urls:
        logging.error(
            "Error: Both sitemap_url and website_urls found for the application."
        )
        return

    if sitemap_url == "N/A" and not website_urls:
        logging.error(
            "Error: No sitemap_url or website_urls found for the application."
        )
        return

    extracted_urls = []

    if sitemap_url != "N/A":
        extracted_urls = extract_urls_from_sitemap(sitemap_url)
        if not extracted_urls:
            logging.info("No URLs extracted from the sitemap.")
    else:
        extracted_urls = website_urls

    crawl(extracted_urls)
    logging.info("Text files created for all URLs.")

    scrapper_urls_folder_path = os.path.join("ScrappedURLs", app_id)
    if not os.path.exists(scrapper_urls_folder_path):
        os.makedirs(scrapper_urls_folder_path)

    file_path = os.path.join(scrapper_urls_folder_path, "url.txt")
    with open(file_path, "w") as f:
        f.write("\n".join(extracted_urls))

    logging.info(
        "URLs extracted from sitemap or website_urls and saved in ScrappedURLs folder."
    )


def create_processed_data():
    app_processed_folder_path = os.path.join("processed", app_id)
    if not os.path.exists("processed/"):
        os.mkdir("processed/")

    if not os.path.exists(app_processed_folder_path):
        os.makedirs(app_processed_folder_path)

        texts = []

        for file in os.listdir(app_text_folder_path):
            with open(f"{app_text_folder_path}/{file}", "r", encoding="UTF-8") as f:
                text = f.read()
                source_url = text.split("\n")[0].replace("Source URL: ", "")
                texts.append(
                    (
                        file[11:-4]
                        .replace("-", " ")
                        .replace("_", " ")
                        .replace("#update", ""),
                        source_url,
                        text,
                    )
                )

        df = pd.DataFrame(texts, columns=["fname", "source url", "text"])
        df["text"] = df.fname + ". " + remove_newlines(df.text)

        scraped_file_path = os.path.join(app_processed_folder_path, "scraped.csv")
        if not os.path.exists(scraped_file_path):
            df.to_csv(scraped_file_path)
            logging.info("scraped.csv file created.")
        else:
            logging.info(
                "scraped.csv file already exists. Skipping processed data creation."
            )

        embeddings_file_path = os.path.join(app_processed_folder_path, "embeddings.csv")
        if not os.path.exists(embeddings_file_path):
            df.to_csv(embeddings_file_path)
            logging.info("embeddings.csv file created.")
        else:
            logging.info(
                "embeddings.csv file already exists. Skipping processed data creation."
            )

        df.head()

        tokenizer = tiktoken.get_encoding("cl100k_base")

        df = pd.read_csv(embeddings_file_path, index_col=0)
        df.columns = ["title", "source url", "text"]

        df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

        df.n_tokens.hist()

        max_tokens = 500

        def split_into_many(text, max_tokens=max_tokens):
            sentences = text.split(". ")
            n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

            chunks = []
            tokens_so_far = 0
            chunk = []

            for sentence, token in zip(sentences, n_tokens):
                if tokens_so_far + token > max_tokens:
                    chunks.append(". ".join(chunk) + ".")
                    chunk = []
                    tokens_so_far = 0

                if token > max_tokens:
                    continue

                chunk.append(sentence)
                tokens_so_far += token + 1

            if chunk:
                chunks.append(". ".join(chunk) + ".")

            return chunks

        shortened = []

        for row in df.iterrows():
            if row[1]["text"] is None:
                continue

            if row[1]["n_tokens"] > max_tokens:
                shortened += split_into_many(row[1]["text"])
            else:
                shortened.append(row[1]["text"])

        df = pd.DataFrame(shortened, columns=["text"])
        df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
        df.n_tokens.hist()

        df["embeddings"] = df.text.apply(
            lambda x: openai.Embedding.create(
                input=x, engine="text-embedding-ada-002", timeout=600
            )["data"][0]["embedding"]
        )

        df.to_csv(embeddings_file_path)

        df.head()
    else:
        logging.info(
            "Processed folder and files already exist. Skipping processed data creation."
        )


def process_url_batch(url_batch):
    for url in url_batch:
        local_domain = urlparse(url).netloc
        crawl_single_url(url, local_domain)


@lru_cache(maxsize=None)
def measure_performance():
    start_time = time.time()

    create_text_data()
    create_processed_data()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Performance Measurement: Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    logging.info("Checking if the required files and folders already exist...")

    text_folder_exists = os.path.exists("text/") and os.path.exists(
        os.path.join("text", app_id, domain)
    )
    processed_folder_exists = os.path.exists("processed/") and os.path.exists(
        os.path.join("processed", app_id)
    )
    embeddings_file_exists = os.path.exists(
        os.path.join("processed", app_id, "embeddings.csv")
    )
    scraped_file_exists = os.path.exists(
        os.path.join("processed", app_id, "scraped.csv")
    )

    if (
        text_folder_exists
        and processed_folder_exists
        and embeddings_file_exists
        and scraped_file_exists
    ):
        logging.info(
            f"Files and folders for app ID '{app_id}' already exist. Starting the server."
        )
    else:
        if not text_folder_exists or not processed_folder_exists:
            logging.info(
                "Some of the required folders are missing. Starting the data creation process."
            )
            create_text_data()
            create_processed_data()
        else:
            if not embeddings_file_exists or not scraped_file_exists:
                logging.info(
                    "Some of the required files are missing. Starting the data creation process."
                )
                create_processed_data()

        logging.info(
            "Extracting URLs from sitemap or website_urls and saving in ScrappedURLs folder..."
        )
        parallel_crawl(application["website_urls"])

    measure_performance()

    app.run()


def run_scrapper():
    create_text_data()
    create_processed_data()
