# will make a json file here and on calling scrapy crawl quotes, the arguments from json will be read for execution

# list the tags that you want to scrape data from in this section
# list the domains that you want to skip
# list the stopwords that is what sites you do not want to crawl (ex youtube, spotify, twitter) or urls that have some keyword in them (ex login, auth)
# specify the maximum depth of the crawl, number of pages to crawl, number of URLs to look for per page, where you want to store the HTMLs
# you can also specify whether you want a list of crawled URLs, and a dictionary of the text crawled

import json

args = {}

args["tags_to_scrape"] = ["p"]
args["skipwords"] = [
    "youtube",
    "twitter",
    "login",
    "auth",
    "reddit",
    "google",
    "pinterest",
    "facebook",
    "ytube",
]

# Crawl parameters
args["max_depth"] = 2
args["number_pages_to_crawl"] = 5
args["urls_per_page"] = 4


# Put true if want to fetch htmls for the webpages.
args["fetch_htmls"] = True

# Put true if you want to parse HTMLs from some folder. If fetch_htmls is true, then parsing will happen from the folder that crawler writes the fetched html to
args["parse htmls"] = True

# parsing text data while visiting a page
args["parse dynamic"] = False

args["html_output_folder"] = "openai_healthline_htmls"
args["parse_html_folder"] = "openai_healthline_parsed"


# if you have a set of previously traversed/tried URLs and don't wish to visit them again, pass a path to the URLs
args["load_tried_urls"] = None

args["start_urls"] = [
    "https://openai.com/api/",
    "https://www.healthline.com/health/food-nutrition/beetroot-juice-benefits",
]


# !! if you are being banned from sites, you may want to modify custom_settings in quotes_spider.py
# TO-DO
args["AUTOTHROTTLE_ENABLED"] = True
args["CONCURRENT_REQUESTS_PER_DOMAIN"] = 1


with open("arguments_for_crawler.json", "w") as f:
    f.write(json.dumps(args))
