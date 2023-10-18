# -*- coding: utf-8 -*-

import openai
from tqdm import tqdm
import ebooklib
from ebooklib import epub
import os
from bs4 import BeautifulSoup
import configparser
import random
import json
import pandas as pd
import chardet
import argparse
import re
from retry import retry
import logging

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Initialize a count variable of tokens cost.
cost_prompt_tokens = 0
cost_completion_tokens = 0
text_length = 0
gpt_model = ""

with open('settings.cfg', 'rb') as f:
    content = f.read()
    encoding = chardet.detect(content)['encoding']

with open('settings.cfg', encoding=encoding) as f:
    config_text = f.read()
    config = configparser.ConfigParser()
    config.read_string(config_text)

# Obtain OpenAI API key and language.
openai_apikey = config.get('option', 'openai-apikey')
# language_name = config.get('option', 'target-language')
prompt = config.get('option', 'prompt')
bilingual_output = config.get('option', 'bilingual-output')
language_code = config.get('option', 'langcode')
api_proxy = config.get('option', 'openai-proxy')
# Get startpage and endpage as integers with default values
startpage = config.getint('option', 'startpage', fallback=1)
endpage = config.getint('option', 'endpage', fallback=-1)
# Set the translated name table file path.
transliteration_list_file = config.get('option', 'transliteration-list')
# Is case-sensitive matching enabled in the translated name table replacement?
case_matching = config.get('option', 'case-matching')
gpt_model = config.get('option', 'gpt_model')


def get_epub_title(epub_filename):
    try:
        book = epub.read_epub(epub_filename)
        metadata = book.get_metadata('DC', {})
        if metadata:
            if 'title' in metadata:
                return metadata['title'][0]
        else:
            return "Unknown title"
    except:
        return "Unknown title"


def random_api_key():
    return random.choice(key_array)


@retry(delay=30, jitter=10, logger=logger)
def create_chat_completion(text, prompt=prompt, model=gpt_model, **kwargs):
    # openai.api_key = random_api_key()
    return openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{prompt}",
            },
            {"role": "user", "content": f"{text}"}
        ],
        # temperature=1,
        # # max_tokens=256,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        **kwargs
    )


def convert_epub_to_text(epub_filename):
    # Open EPUB file.
    book = epub.read_epub(epub_filename)

    # Get all text.
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Extract plain text using BeautifulSoup.
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += re.sub(r'\n+', '\n', soup.get_text().strip())

    return text


def text_to_epub(text, filename, language_code='en', title="Title"):
    text = text.replace("\n", "<br>")
    # Create EPUB book object.
    book = epub.EpubBook()

    # Set metadata.
    book.set_identifier(str(random.randint(100000, 999999)))
    book.set_title(title)
    book.set_language(language_code)

    # Create chapter object.
    c = epub.EpubHtml(title='Chapter 1', file_name='chap_1.xhtml', lang=language_code)
    c.content = text

    # Add the chapter to the book.
    book.add_item(c)

    # Add toc
    book.toc = (epub.Link('chap_1.xhtml', 'Chapter 1', 'chap_1'),)
    # Set the book spine order.
    book.spine = ['nav', c]
    # 添加导航
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # set cover
    # book.set_cover('image.jpg', open('image.jpg', 'rb').read())

    # Write the epub file
    epub.write_epub(filename, book, {})


# Split the text into a list of short texts, each no more than 4096 characters.
def split_text(text):
    global text_length
    text_length = len(text)

    sentence_list = re.findall(r'.+?[。！？!?.]', text)

    # Initialize the short text list.
    short_text_list = []
    # Initialize the current short text.
    short_text = ""
    limit_length = 1800
    if gpt_model == "gpt-3.5-turbo-16k":
        limit_length = limit_length * 4
    elif gpt_model == "gpt-4":
        limit_length = limit_length * 2
    elif gpt_model == "gpt-4-32k":
        limit_length = limit_length * 8

    # Traverse the sentence list.
    for s in sentence_list:
        # If the length of the current short text,
        # when combined with the length of the new sentence does not exceed 4096 characters,
        # add the new sentence to the current short text.
        if len(short_text + s) <= limit_length:
            short_text += s
        # If the current short text, when combined with the length of the new sentence, exceeds 4096 characters,
        # add the current short text to the list of short texts and reset the current short text to the new sentence.
        else:
            short_text_list.append(short_text)
            short_text = s
    # Add the last short text to the list of short texts.
    short_text_list.append(short_text)
    return short_text_list


# Replace the period with a period followed by a line break.
def return_text(text):
    text = text.replace(". ", ".\n")
    text = text.replace("。", "。\n")
    text = text.replace("！", "！\n")
    return text


# Translate text
def translate_text(text):
    global cost_prompt_tokens
    global cost_completion_tokens

    # Invoke the OpenAI API for translation.
    completion = create_chat_completion(text)
    t_text = (
        completion["choices"][0]
        .get("message")
        .get("content")
        .encode("utf8")
        .decode()
    )
    # Get the token usage from the API response
    cost_prompt_tokens += completion["usage"]["prompt_tokens"]
    cost_completion_tokens += completion["usage"]["completion_tokens"]

    return t_text


def translate_and_store(text):
    # If the text has already been translated, simply return the translated result.
    if text in translated_dict:
        return translated_dict[text]

    # Otherwise, call the translate_text function for translation and store the results in a dictionary.
    translated_text = translate_text(text)
    translated_dict[text] = translated_text

    with open(jsonfile, "w", encoding="utf-8") as f:
        json.dump(translated_dict, f, ensure_ascii=False, indent=4)

    return translated_text


def text_replace(long_string, xlsx_path, case_sensitive):
    # Read the excel file and save the first column and the second column as two separate lists.
    df = pd.read_excel(xlsx_path)
    old_words = df.iloc[:, 0].tolist()
    new_words = df.iloc[:, 1].tolist()
    # Sort the old word list in descending order of length and adjust the new word list accordingly.
    old_words, new_words = zip(*sorted(zip(old_words, new_words), key=lambda x: len(x[0]), reverse=True))
    # Iterate through both lists and perform string replacements.
    for i in range(len(old_words)):
        # If case sensitivity is not required, convert both the string and the replacement word to lowercase.
        if not case_sensitive:
            lower_string = long_string.lower()
            lower_old_word = old_words[i].lower()
            # Use regular expressions for replacement, ensuring to preserve the original case of the string.
            long_string = re.sub(r"\b" + lower_old_word + r"\b", new_words[i], long_string, flags=re.IGNORECASE)
        # If case sensitivity is required, directly use regular expressions for replacement.
        else:
            long_string = re.sub(r"\b" + old_words[i] + r"\b", new_words[i], long_string)
    return long_string


if __name__ == "__main__":

    # Set the OpenAI API key.
    openai.api_key = openai_apikey

    # Split the OpenAI API key into an array.
    key_array = openai_apikey.split(',')

    # If the configuration file is written, set the API proxy.
    # if len(api_proxy) == 0:
    #     print("OpenAI API proxy not detected, the current API address in use is: " + openai.api_base)
    # else:
    #     api_proxy_url = api_proxy + "/v1"
    #     openai.api_base = os.environ.get("OPENAI_API_BASE", api_proxy_url)
    #     print("Using OpenAI API proxy, the proxy address is: " + openai.api_base)

    # Create argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the input file")
    parser.add_argument("--test", help="Only translate the first 3 short texts", action="store_true")
    # Use the translated name table?
    parser.add_argument("--tlist", help="Use the translated name table", action="store_true")
    args = parser.parse_args()

    # Get command line arguments.
    filename = args.filename
    base_filename, file_extension = os.path.splitext(filename)
    new_filename = base_filename + "_translated.epub"
    new_filenametxt = base_filename + "_translated.txt"
    jsonfile = base_filename + "_process.json"
    # Load the translated text from the file.
    translated_dict = {}
    try:
        with open(jsonfile, "r", encoding="utf-8") as f:
            translated_dict = json.load(f)
    except FileNotFoundError:
        pass

    text = ""

    if filename.endswith('.epub'):
        print("Converting epub to text")
        book = epub.read_epub(filename)
    elif filename.endswith('.txt'):

        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()

        title = os.path.basename(filename)
    else:
        print("Unsupported file type")

    if filename.endswith('.epub'):
        # Retrieve all chapters.
        items = book.get_items()

        # Iterate through all chapters.
        translated_all = ''
        count = 0
        for item in tqdm(items):
            # If the chapter type is a document type, it needs to be translated.
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Extract the original text using BeautifulSoup.
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text().strip()
                img_html = ''
                img_tags = soup.find_all('img')
                for img_tag in img_tags:
                    img_html += str(img_tag) + '<br>'
                # If the original text is empty, skip it.
                if not text:
                    continue
                # Replace all line breaks with spaces.
                text = text.replace("\n", " ")
                text = re.sub(r"\s+", " ", text)

                # If a translation substitution table has been set,
                # perform the pre-translation substitution on the text.
                if args.tlist:
                    text = text_replace(text, transliteration_list_file, case_matching)

                # Split the text into a list of short texts, each containing no more than 1024 characters.
                short_text_list = split_text(text)
                if args.test:
                    short_text_list = short_text_list[:3]

                translated_text = ""

                # Translate each short text in the list one by one.
                for short_text in tqdm(short_text_list):
                    print(return_text(short_text))
                    count += 1
                    # Translate the current short text.
                    translated_short_text = translate_and_store(short_text)
                    short_text = return_text(short_text)
                    translated_short_text = return_text(translated_short_text)
                    # Add the current source text and its translated text to the final document.
                    if bilingual_output.lower() == 'true':
                        translated_text += f"{short_text}<br>\n{translated_short_text}<br>\n"
                    else:
                        translated_text += f"{translated_short_text}<br>\n"
                    # print(short_text)
                    print(translated_short_text)
                # Replace the original chapter content with the translated text.
                item.set_content((img_html + translated_text.replace('\n', '<br>')).encode('utf-8'))
                translated_all += translated_text
                if args.test and count >= 3:
                    break

        # write the epub file
        epub.write_epub(new_filename, book, {})
        # Write the translated text into a txt file simultaneously in case the EPUB plugin encounters a problem.
        with open(new_filenametxt, "w", encoding="utf-8") as f:
            f.write(translated_all)

    else:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        # If the translated name table replacement is set, the text is replaced before translation.
        if args.tlist:
            text = text_replace(text, transliteration_list_file, case_matching)

        # Split the text into a list of short texts, each not exceeding (4096) characters.
        short_text_list = split_text(text)
        if args.test:
            short_text_list = short_text_list[:3]
        translated_text = ""

        # Traverse the list of short texts and translate each short text in turn.
        for short_text in tqdm(short_text_list):
            print(return_text(short_text))
            # Translate the current short text.
            translated_short_text = translate_and_store(short_text)
            short_text = return_text(short_text)
            translated_short_text = return_text(translated_short_text)
            # Add the current short text and its translated text into the total text.
            if bilingual_output.lower() == 'true':
                translated_text += f"{short_text}\n{translated_short_text}\n"
            else:
                translated_text += f"{translated_short_text}\n"
            # print(short_text)
            print(translated_short_text)

        # write the translated text into an EPUB file
        with tqdm(total=10, desc="Writing translated text to epub") as pbar:
            text_to_epub(translated_text.replace('\n', '<br>'), new_filename, language_code, title)
            pbar.update(1)

        # To ensure the translation is saved in a text file in case there are issues with the EPUB plugin.
        with open(new_filenametxt, "w", encoding="utf-8") as f:
            f.write(translated_text)

    cost_prompt_tokens_rate = 0.0015
    cost_completion_tokens_rate = 0.002

    if gpt_model == "gpt-3.5-turbo-16k":
        cost_prompt_tokens_rate = 0.003
        cost_completion_tokens_rate = 0.004
    elif gpt_model == "gpt-4":
        cost_prompt_tokens_rate = 0.03
        cost_completion_tokens_rate = 0.06
    elif gpt_model == "gpt-4-32k":
        cost_prompt_tokens_rate = 0.06
        cost_completion_tokens_rate = 0.12

    cost = (cost_prompt_tokens / 1000 * cost_prompt_tokens_rate) + (
            cost_completion_tokens / 1000 * cost_completion_tokens_rate)
    print(f" {text_length} characters translation completed, model select: {gpt_model}.")
    print(f"Total cost: {cost_prompt_tokens} prompt tokens, {cost_completion_tokens} completion tokens, ${cost}.")

    try:
        os.remove(jsonfile)
        print(f"File '{jsonfile}' has been deleted.")
    except FileNotFoundError:
        print(f"File '{jsonfile}' not found. No file was deleted.")
