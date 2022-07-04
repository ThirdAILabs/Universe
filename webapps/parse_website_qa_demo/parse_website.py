import urllib.request
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

remove_bad_chars = re.compile("[^a-zA-Z,!?.0-9 ]")


class ParsedWebsite:
    def __init__(self, base_url, max_crawl_depth):
        self.seen_urls = set()
        self.url_text_pairs = []
        self.base_url = base_url

        try:
            self._websearch_recursive(base_url, max_crawl_depth)
        except Exception as e:
            print(f"Error {e} when trying base url {base_url}")

    def _websearch_recursive(self, url, depth_left):
        if depth_left < 0:
            return
        url = url.strip().strip("/")
        if url in self.seen_urls:
            return
        self.seen_urls.add(url)
        print(f"Processing {url}")

        html = urllib.request.urlopen(url).read()

        text = self._get_clean_text(html)
        self.url_text_pairs.append((url, text))

        urls_to_recurse_to = self._get_valid_urls_from_html(html)
        for next_url in urls_to_recurse_to:
            try:
                self._websearch_recursive(next_url, depth_left - 1)
            except Exception as e:
                print(f"Error {e} when trying {next_url}, found on {url}")

    def _get_clean_text(self, html):
        clean_text = " ".join(BeautifulSoup(html, "html.parser").stripped_strings)
        without_bad_chars = remove_bad_chars.sub(" ", clean_text)
        return " ".join(without_bad_chars.split())

    # See https://python.plainenglish.io/scraping-the-subpages-on-a-website-ea2d4e3db113
    def _get_valid_urls_from_html(self, html):
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for link in soup.find_all("a", href=True):

            link_str = str(link["href"])
            if link_str.startswith("//"):
                link_str = "https:" + link_str

            # Starting with a / means it is a relative link, so we turn it
            # into an absolute one
            if link_str.startswith("/"):
                link_str = urljoin(self.base_url, link_str)

            # Strip the query parameters
            link_str = urljoin(link_str, urlparse(link_str).path)

            # Append to list if new link contains original link
            if link_str.startswith(self.base_url):
                links.append(link_str)

        return links

    def get_id_text_pairs(self, num_sentences_per_passage):
        result = []
        for url, text in self.url_text_pairs:
            sentences = text.split(".")
            for sentence_id in range(0, len(sentences), num_sentences_per_passage):
                passage = (
                    ".".join(
                        sentences[sentence_id : sentence_id + num_sentences_per_passage]
                    )
                    + "."
                )
                passage_id = f"{len(result)} : {url}"
                if self._passage_okay(passage):
                    result.append((passage_id, passage))
        return result

    # Some simple heuristics to make sure the passage is okay
    def _passage_okay(self, passage):
        words = passage.split()
        # Too short
        if len(words) < 3:
            return False

        # Too many words capitalized, probably not text
        num_capitalized = sum([w[0].isupper() for w in words])
        if num_capitalized * 3 > len(words):
            return False

        return True
