from flask import Flask, jsonify, request
import openai
import pandas as pd
import ast
from scrapper import *
from openai.embeddings_utils import distances_from_embeddings
import time
import json
import os
import re
import logging

app = Flask(__name__)
embedding_df = None
embedding_df_numpyArray = None
processed_folder = "processed"
app_id_to_embeddings_file = {}

# Configure logging
logging.basicConfig(filename="mainApp.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the data from config.json
with open("config.json") as config_file:
    config_data = json.load(config_file)

# Initialize the app_id_to_embeddings_file dictionary
for app_info in config_data["applications"]:
    app_id = app_info["app_id"]
    embeddings_file_path = os.path.join(processed_folder, app_id, "embeddings.csv")
    app_id_to_embeddings_file[app_id] = embeddings_file_path


@app.route("/questions", methods=["POST"])
def answer_questions():
    try:
        global embedding_df, app_id, app_id_to_embeddings_file
        data = request.get_json()
        question = data.get("message", "")
        application_id = data.get("applicationId")
        logger.info("Data from Kommunicate: %s", data)
        logger.info("Incoming question: %s", question)
        logger.info("Application ID from Kommunicate: %s", application_id)

        # If application ID is provided, use it to fetch the embeddings.csv file
        if application_id:
            # Check if application ID exists in app_id_to_embeddings_file dictionary
            if application_id in app_id_to_embeddings_file:
                embeddings_file_path = app_id_to_embeddings_file[application_id]
            else:
                logger.error("Invalid application ID: %s", application_id)
                return jsonify({"error": "Invalid application ID"})

            # Load the DataFrame corresponding to the application ID
            if embedding_df is None or app_id != application_id:
                embedding_df = pd.read_csv(embeddings_file_path)
                app_id = application_id

        # If no application ID is provided, search for embeddings.csv file in every app_id folder
        else:
            app_id = None  # Initialize app_id to None initially
            for app in applications:
                app_id = app["app_id"]
                # Check if the embeddings.csv file path is already in the cache
                if app_id in app_id_to_embeddings_file:
                    embeddings_file_path = app_id_to_embeddings_file[app_id]
                else:
                    # Fetch the embeddings.csv file path for the specific app_id
                    embeddings_file_path = os.path.join(
                        processed_folder, app_id, "embeddings.csv"
                    )
                    app_id_to_embeddings_file[app_id] = embeddings_file_path

                if os.path.exists(embeddings_file_path):
                    embedding_df = pd.read_csv(embeddings_file_path)
                    break  # Exit the loop once a matching embeddings file is found

            if app_id is None:
                logger.error("No matching embeddings file found")
                return jsonify({"error": "Embeddings file not found"})

        logger.info("app_id_to_embeddings_file: %s", app_id_to_embeddings_file)
        logger.info("Folder: %s", app_id)

        start_time = time.time()
        response = answer_question(embedding_df, question)
        logger.info("--- OpenAI API time %s seconds ---", (time.time() - start_time))
        logger.info("Matched app_id: %s with applicationId: %s", app_id, application_id)

        # Extract URLs from the response
        urls = re.findall(r"http[s]?://[^\s\)]+", response)

        # Removing the source information from the response

        response = re.sub(r"Source:\s[^\n]+", "", response)
        print("response Final", response)
        # Create HTML links for the extracted URLs
        url_links = (
            "<br>".join(
                [
                    f'<a href="{url}" target="_blank" style="color:#0000FF;">{url}</a>'
                    for url in urls
                ]
            )
            or f'<a href="https://cfcc.ca.gov/" target="_blank" style="color:#0000FF;">https://cfcc.ca.gov/</a>'
        )

        content_source_link = (
            f"<div><p>{response}</p><hr><p>Source Page:</p>{url_links}</div>"
        )

        return jsonify(
            [
                {
                    "messageType": "html",
                    "platform": "kommunicate",
                    "message": content_source_link,
                    "metadata": {
                        "contentType": "300",
                        "templateId": "3",
                        "payload": [
                            {
                                "type": "link",
                                "url": "https://test.calosba.ca.gov/business-learning-center/start-up/business-quick-start-guides/",
                                "name": "Mobile Food Business",
                            },
                            {
                                "type": "link",
                                "url": "https://test.calosba.ca.gov/wp-content/uploads/BQSG_MOBILE-FOOD-VENDORS.pdf",
                                "name": "Business Guide",
                                "openLinkInNewTab": false,
                            },
                        ],
                    },
                }
            ]
        )

    except Exception as e:
        logger.error(e)
        return jsonify({"error": "Internal Server Error"})


def create_context(question, df, max_len=500):
    start_time = time.time()
    q_embedding = openai.Embedding.create(
        input=[question], engine="text-embedding-ada-002"
    )["data"][0]["embedding"]
    logger.info("--- q_embedding time %s seconds ---", (time.time() - start_time))

    # Drop empty records where embeddings are not present
    df.dropna(axis=0, how="any", subset=None, inplace=True)

    start_time = time.time()
    global embedding_df_numpyArray
    if embedding_df_numpyArray is None:
        df["embeddings"] = df["embeddings"].apply(
            ast.literal_eval
        )  # Convert string representations to NumPy arrays
        embedding_df_numpyArray = df
    logger.info(
        "--- Convert string representations to NumPy arrays time %s seconds ---",
        (time.time() - start_time),
    )

    start_time = time.time()
    df["distances"] = distances_from_embeddings(
        q_embedding,
        embedding_df_numpyArray["embeddings"].values.tolist(),
        distance_metric="cosine",
    )
    logger.info("--- Calculating Distance %s seconds ---", (time.time() - start_time))

    returns = []
    cur_len = 0
    start_time = time.time()

    for i, row in df.sort_values("distances").iterrows():
        cur_len += row["n_tokens"] + 4
        returns.append(row["text"])
        if cur_len > max_len:
            break
    logger.info("--- For loop Distance %s seconds ---", (time.time() - start_time))

    logger.info("context::::::", returns)
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    question,
    model="text-davinci-003",
    max_len=1800,
    max_tokens=200,
    stop_sequence=None,
):
    context = create_context(question, df, max_len=max_len)
    prompt = f"Customer Service Chatbot: I'm here to assist you. Please provide the best answer to the question based on the provided context from {domain}. If you don't have the information, respond with \"I apologize, but I don't have that information at the moment.\" If relevant information is available from a specific website, please include the website name or source, but only if it's a valid link.\n\nContext: {context}\n\n---\n\nQuestion: {question}\n\nAlso Add the source of the information like page name or URL or both, if it's a valid link.\n\nAdditional Information:\n- Please include any relevant names, dates, or locations related to the question.\n- If the question can be interpreted in multiple ways, please ask for clarifications or provide additional examples.\n- Preferred answer format: [concise summary/bullet points/step-by-step explanation].\n\nYour Response:"

    try:
        start_time = time.time()
        response = openai.Completion.create(
            prompt=[prompt],
            temperature=0.1,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        logger.info(
            "--- OpenAI API excluding create_context time %s seconds ---",
            (time.time() - start_time),
        )
        print("response Actual::", response)
        return response["choices"][0]["text"].strip()

    except Exception as e:
        logger.error(e)
        return ""


if __name__ == "__main__":
    run_scrapper()
    app.run(host="0.0.0.0", port=5003)
