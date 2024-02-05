import fitz
import pytest
from ndb_utils import PDF_FILE
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_pdf_highlighting(version):
    pdf = ndb.PDF(PDF_FILE, version=version)
    highlighted_doc = ndb.PDF.highlighted_doc(pdf.reference(0))
    assert isinstance(highlighted_doc, fitz.Document)
    assert not highlighted_doc.is_closed
