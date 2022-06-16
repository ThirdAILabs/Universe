from flask import Flask, request, render_template
from parse_website import ParsedWebsite

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html", background_color="", console_output="")


@app.route("/", methods=["POST"])
def crawl_site():
    url = request.form["url"]
    depth = int(request.form["depth"])
    if not url.startswith("http"):
        url = "https://" + url

    global parsed
    parsed = ParsedWebsite(url, depth)

    return render_template(
        "home.html",
        background_color="",
        console_output="Parsed the following urls: \n"
        + "\n".join(list(parsed.seen_urls)),
    )


@app.route("/do-qa", methods=["POST"])
def do_qa():
    print("HERE", parsed)

    return parsed.base_url
