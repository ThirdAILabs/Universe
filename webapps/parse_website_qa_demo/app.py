from flask import Flask, request, render_template
from parse_website import ParsedWebsite
from thirdai.search import EasyQA

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "home.html", background_color="", console_output="", submit_visibility="hidden"
    )


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
        submit_visibility="visible",
    )


@app.route("/do-qa", methods=["POST", "GET"])
def do_qa():
    global index
    if request.method == "GET":
        pairs = parsed.get_id_text_pairs(num_sentences_per_passage=1)
        index = EasyQA().index(pairs)
        return render_template(
            "question.html", console_output="", answer_url="", visibility="hidden"
        )
    else:
        question = request.form["question"]
        result = index.query(question, top_k=5)
        result_no_dups = []
        # TODO(josh): Why are there dupes?
        for r in result:
            if r[1] not in result_no_dups:
                result_no_dups.append(r[1])
        return render_template(
            "question.html",
            console_output="\n\n".join(
                [f"{i}: {r}" for i, r in enumerate(result_no_dups)]
            ),
            answer_url="",
            visibility="visible",
        )
