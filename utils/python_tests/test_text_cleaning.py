import re
from thirdai import data


class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """

    CONTRACTIONS2 = [
        r"(?i)\b(can)(?#X)(not)\b",
        r"(?i)\b(d)(?#X)('ye)\b",
        r"(?i)\b(gim)(?#X)(me)\b",
        r"(?i)\b(gon)(?#X)(na)\b",
        r"(?i)\b(got)(?#X)(ta)\b",
        r"(?i)\b(lem)(?#X)(me)\b",
        r"(?i)\b(more)(?#X)('n)\b",
        r"(?i)\b(wan)(?#X)(na)(?=\s)",
    ]
    CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
    CONTRACTIONS4 = [r"(?i)\b(whad)(dd)(ya)\b", r"(?i)\b(wha)(t)(cha)\b"]

# Starting quotes.
STARTING_QUOTES = [
    (re.compile("([«“‘„]|[`]+)", re.U), r" \1 "),
    (re.compile(r"^\""), r"``"),
    (re.compile(r"(``)"), r" \1 "),
    (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
    (re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b", re.U), r"\1 \2"),
]

# Ending quotes.
ENDING_QUOTES = [
    (re.compile("([»”’])", re.U), r" \1 "),
    (re.compile(r"''"), " '' "),
    (re.compile(r'"'), " '' "),
    (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
    (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
]

# For improvements for starting/closing quotes from TreebankWordTokenizer,
# see discussion on https://github.com/nltk/nltk/pull/1437
# Adding to TreebankWordTokenizer, nltk.word_tokenize now splits on
# - chervon quotes u'\xab' and u'\xbb' .
# - unicode quotes u'\u2018', u'\u2019', u'\u201c' and u'\u201d'
# See https://github.com/nltk/nltk/issues/1995#issuecomment-376741608
# Also, behavior of splitting on clitics now follows Stanford CoreNLP
# - clitics covered (?!re|ve|ll|m|t|s|d)(\w)\b

# Punctuation.
PUNCTUATION = [
    (re.compile(r'([^\.])(\.)([\]\)}>"\'' "»”’ " r"]*)\s*$", re.U), r"\1 \2 \3 "),
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (
        re.compile(r"\.{2,}", re.U),
        r" \g<0> ",
    ),  # See https://github.com/nltk/nltk/pull/2322
    (re.compile(r"[;@#$%&]"), r" \g<0> "),
    (
        re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
        r"\1 \2\3 ",
    ),  # Handles the final period.
    (re.compile(r"[?!]"), r" \g<0> "),
    (re.compile(r"([^'])' "), r"\1 ' "),
    (
        re.compile(r"[*]", re.U),
        r" \g<0> ",
    ),  # See https://github.com/nltk/nltk/pull/2322
]

# Pads parentheses
PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")

# Optionally: Convert parentheses, brackets and converts them to PTB symbols.
CONVERT_PARENTHESES = [
    (re.compile(r"\("), "-LRB-"),
    (re.compile(r"\)"), "-RRB-"),
    (re.compile(r"\["), "-LSB-"),
    (re.compile(r"\]"), "-RSB-"),
    (re.compile(r"\{"), "-LCB-"),
    (re.compile(r"\}"), "-RCB-"),
]

DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

# List of contractions adapted from Robert MacIntyre's tokenizer.
_contractions = MacIntyreContractions()
CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

def tokenize(
    self, text: str, convert_parentheses: bool = False, return_str: bool = False
) -> List[str]:
    r"""Return a tokenized copy of `text`.

    >>> from nltk.tokenize import NLTKWordTokenizer
    >>> s = '''Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.  Please buy me\ntwo of them.\nThanks.'''
    >>> NLTKWordTokenizer().tokenize(s) # doctest: +NORMALIZE_WHITESPACE
    ['Good', 'muffins', 'cost', '$', '3.88', '(', 'roughly', '3,36',
    'euros', ')', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',
    'of', 'them.', 'Thanks', '.']
    >>> NLTKWordTokenizer().tokenize(s, convert_parentheses=True) # doctest: +NORMALIZE_WHITESPACE
    ['Good', 'muffins', 'cost', '$', '3.88', '-LRB-', 'roughly', '3,36',
    'euros', '-RRB-', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',
    'of', 'them.', 'Thanks', '.']


    :param text: A string with a sentence or sentences.
    :type text: str
    :param convert_parentheses: if True, replace parentheses to PTB symbols,
        e.g. `(` to `-LRB-`. Defaults to False.
    :type convert_parentheses: bool, optional
    :param return_str: If True, return tokens as space-separated string,
        defaults to False.
    :return: List of tokens from `text`.
    :rtype: List[str]
    """
    for regexp, substitution in self.STARTING_QUOTES:
        text = regexp.sub(substitution, text)
    for regexp, substitution in self.PUNCTUATION:
        text = regexp.sub(substitution, text)
    # Handles parentheses.
    regexp, substitution = self.PARENS_BRACKETS
    text = regexp.sub(substitution, text)
    # Optionally convert parentheses
    if convert_parentheses:
        for regexp, substitution in self.CONVERT_PARENTHESES:
            text = regexp.sub(substitution, text)
    # Handles double dash.
    regexp, substitution = self.DOUBLE_DASHES
    text = regexp.sub(substitution, text)
    # add extra space to make things easier
    text = " " + text + " "
    for regexp, substitution in self.ENDING_QUOTES:
        text = regexp.sub(substitution, text)

    for regexp in self.CONTRACTIONS2:
        text = regexp.sub(r" \1 \2 ", text)
    for regexp in self.CONTRACTIONS3:
        text = regexp.sub(r" \1 \2 ", text)
    return text.split()





def test_something():

    STARTING_QUOTES = [
        (re.compile("([«“‘„]|[`]+)", re.U), r" \1 "),
        (re.compile(r"^\""), r"``"),
        (re.compile(r"(``)"), r" \1 "),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
        (re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b", re.U), r"\1 \2"),
    ]

regexp = re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b", re.U)
substitution = r"\1 \2"
regexp.sub(substitution, text)
    pass