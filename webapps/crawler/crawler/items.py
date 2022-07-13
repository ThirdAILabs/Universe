# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader import ItemLoader
from itemloaders.processors import TakeFirst, MapCompose
from w3lib.html import remove_tags
import re
from urllib.parse import urlparse
import os
import json
from scrapy.http import HtmlResponse


def get_domain(URL):

    if URL is None:
        return None

    parsed_uri = urlparse(URL)
    domain = "{uri.scheme}://{uri.netloc}/".format(uri=parsed_uri)
    return domain


def parseURL(
    url, responseURL, skipwords, domain_type="default", restrict_to_domain=False
):

    if (
        url is None
        or (restrict_to_domain and get_domain(responseURL) not in url and url[0] != "/")
        or any(word in url for word in skipwords)
        or len(url) < 5
        or url[0] == "#"
    ):
        return None
    elif url[0:3] == "www" or url[0:5] == "https":
        return url

    if domain_type == "wikipedia":
        if url[0:5] == "/wiki" or url[0:2] == "/w":
            url = "/".join(responseURL.split("/")[:-2]) + url
            return url
        else:
            return None

    # assuming that all other pages have relative links from the domain URL. Write custom methods for changing this behaviour in the code according to the domain name
    else:
        if url[0] == "/":
            url = "/".join(responseURL.split("/")[:3]) + url
            return url
        else:
            return None


# is_scrapy_response denotes whether the resonse object is a scrapy response called from a spider or loaded from html
def parseText(response, tags=["p"], is_scrapy_response=True):
    dc = {}
    if is_scrapy_response:
        dc["metadata"] = {
            "depth": response.meta["depth"],
            "page_no": response.meta["pages"],
            "url": response.request.url,
        }
    for tag in tags:
        paras = response.xpath(f"//{tag}")
        dc[tag] = []
        # print(response.request.url)
        for i, para in enumerate(paras):
            data = " ".join(
                [x.strip() for x in para.css("*::text").extract() if len(x) > 2]
            )
            if len(data.split(" ")) < 10:
                continue
            re.sub("(\u2018|\u2019)", "'", data)
            # print(data)
            dc[tag].append({"para_no": i, "text": data})
    return dc


def parseFile(filename, tags=["p"]):
    with open(filename, "rb") as f:
        response = HtmlResponse(url="https://leadsnowhere.thirdai.com", body=f.read())
        return parseText(response, tags=tags, is_scrapy_response=False)


def parseFolder(foldername, output_folder, tags=["p"]):
    for file in os.listdir(f"./{foldername}"):
        if file.endswith(".html"):
            parsed_data = parseFile(f"./{foldername}/{file}", tags)
            with open(f"./{output_folder}/parsed_{file.replace('.html','')}", "w") as f:
                f.write(json.dumps(parsed_data))


class WikiPediaItem(scrapy.Item):

    name = scrapy.Field()
    child_urls = scrapy.Field()
    headings = scrapy.Field()
    text = scrapy.Field()
