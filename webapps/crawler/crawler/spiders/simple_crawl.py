import scrapy
import re
import os
from crawler.items import parseURL, parseText
from crawler.items import parseFolder
import random
import string
import json
import pickle


class MySpider(scrapy.Spider):

    # name of the spider. call scrapy crawl simple_crawl -o metadata.jl

    name = "simple_crawl"

    tried_urls = set()
    crawled_urls = set()

    max_pages = 3
    urls_per_page = 5
    max_depth = 3

    html_output_folder = "trial_folder"
    parse_html_folder = "trial_folder"

    fetch_html = True
    parse_dynamic = False
    parse_htmls = True

    skipwords = [
        "pinterest",
        "twitter",
        "login",
        "facebook",
        "youtube",
        "ytube",
        "shopheart",
    ]
    # args_file=None
    custom_settings = {
        "AUTOTHROTTLE_ENABLED": True,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 0.5,
        "DNS_TIMEOUT": 2,
        "DOWNLOAD_TIMEOUT": 5,
    }

    def __init__(self, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)

    def start_requests(self):

        urls = ["https://openai.com/api/"]

        # self.args_file="arguments_for_crawler.json"
        # self.args_file=None

        if getattr(self, "args_file", None) is None:
            raise Exception("Provide an args file")

        if self.args_file is not None:
            with open(self.args_file, "r") as f:
                args = json.load(f)

            self.fetch_html = args["fetch_htmls"]
            self.parse_dynamic = args["parse dynamic"]
            self.parse_htmls = args["parse htmls"]

            if "skipwords" in args.keys():
                self.skipwords = args.get("skipwords")
            if "start_urls" in args.keys():
                urls = args["start_urls"]

            if self.parse_dynamic or self.parse_htmls:
                if not "tags_to_scrape" in args.keys():
                    raise Exception("No tags specified to parse")
                self.tags_to_scrape = args["tags_to_scrape"]

            self.max_depth = args.get("max_depth", 3)
            self.max_pages = args.get("number_pages_to_crawl", 30)
            self.urls_per_page = args.get("urls_per_page", 5)
            self.html_output_folder = args.get("html_output_folder", "trial_folder")
            self.parse_html_folder = args.get("parse_html_folder", "trial_folder")

            crawled_urls_filename = args.get("load_tried_urls", None)
            if crawled_urls_filename is not None:
                with open(crawled_urls_filename, "r") as f:
                    self.crawled_urls.update(pickle.load(f))

        if not os.path.exists(self.html_output_folder):
            os.mkdir(self.html_output_folder)

        if not os.path.exists(self.parse_html_folder):
            os.mkdir(self.parse_html_folder)

        if self.fetch_html:

            print(
                f"crawling started with the parameters:\n"
                f"max_depth: {self.max_depth}\n"
                f"number_pages_to_crawl: {self.max_pages}\n"
                f"urls_per_page: {self.urls_per_page}\n"
                f"store html files in: {self.html_output_folder}\n"
                f"avoiding the following search words: {self.skipwords}\n"
                f"parsing HTML pages at the same time: {self.parse_htmls}\n"
                f"dynamic parsing of HTML pages: {self.parse_dynamic}"
            )

            if self.parse_htmls:
                print(f"parsing HTMLs to the folder: {self.parse_html_folder}")

            for url in urls:
                self.crawled_urls.add(url)
                self.parse_para = True
                yield scrapy.Request(
                    url=url,
                    meta={
                        "depth": 0,
                        "pages": 1,
                        "folder": self.html_output_folder,
                        "save_html": self.fetch_html,
                        "parent": "NA",
                    },
                    callback=self.fetch_htmls,
                )

        elif self.parse_htmls:
            print(f"parsing HTMLs from the folder: {self.html_output_folder}")
            print(f"parsing HTMLs to the folder: {self.parse_html_folder}")
            parseFolder(self.html_output_folder, self.parse_html_folder,tags=self.tags_to_scrape)
        # time.sleep(150)

    def fetch_htmls(self, response):

        if (
            any(word in response.request.url for word in self.skipwords)
            or response.status != 200
            or response.meta["depth"] > self.max_depth
            or len(self.crawled_urls) == self.max_pages
        ):
            return

        self.crawled_urls.add(response.request.url)
        yield {
            "depth": response.meta["depth"],
            "page_no": response.meta["pages"],
            "parent": response.meta["parent"],
            "url": response.request.url,
        }
        if "tag_select" in response.meta.keys():
            tag_list = response.meta["tag_select"]
        else:
            tag_list = ["a"]

        for tag in tag_list:
            links = response.xpath(f"//{tag}")
            j = 0
            for i, urls in enumerate(links):
                url = urls.css("a::attr(href)").get()
                url = parseURL(
                    url, response.request.url, self.skipwords, restrict_to_domain=False
                )
                if url is None or url in self.tried_urls:
                    continue
                j = j + 1
                if j > self.urls_per_page:
                    break
                self.tried_urls.add(url)

                yield scrapy.Request(
                    url=url,
                    meta={
                        "parent": response.request.url,
                        "depth": response.meta["depth"] + 1,
                        "pages": j + response.meta["pages"],
                        "folder": response.meta["folder"],
                        "save_html": response.meta["save_html"],
                    },
                    callback=self.fetch_htmls,
                )

        name_html = ""
        if response.meta["save_html"]:
            name_html = re.sub(
                "/",
                "_",
                f"{response.request.url}_depth_{response.meta['depth']}_pageno_{response.meta['pages']}",
            )
            if len(name_html) > 200:
                name_html = (
                    "random_html_"
                    + "".join(
                        random.choices(string.ascii_uppercase + string.digits, k=15)
                    )
                    + ".html"
                )
                with open(f"{response.meta['folder']}/" + name_html, "wb") as f:
                    f.write(f"{response.request.url}/n".encode("utf-8") + response.body)
            else:
                with open(
                    f"{response.meta['folder']}/" + name_html + ".html", "wb"
                ) as f:
                    f.write(response.body)

        if self.parse_dynamic or self.parse_htmls:
            text = parseText(response, tags=self.tags_to_scrape)
            if self.parse_dynamic:
                yield text

            if self.parse_htmls:
                if name_html == "":
                    # print("here")
                    name_html = re.sub(
                        "/",
                        "_",
                        f"{response.request.url}_depth_{response.meta['depth']}_pageno_{response.meta['pages']}",
                    ).sub(".html", "")
                    if len(name_html) > 200:
                        name_html = (
                            "random_html_"
                            + "".join(
                                random.choices(
                                    string.ascii_uppercase + string.digits, k=15
                                )
                            )
                            + ".html"
                        )
                with open(f"{self.parse_html_folder}/{name_html}", "w") as f:
                    f.write(json.dumps(text, ensure_ascii=False))
        # yield {}
