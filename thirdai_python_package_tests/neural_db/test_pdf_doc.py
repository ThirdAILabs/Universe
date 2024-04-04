import fitz
import pytest
from ndb_utils import PDF_FILE, PRIAXOR_PDF_FILE
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_pdf_highlighting(version):
    pdf = ndb.PDF(PDF_FILE, version=version)
    highlighted_doc = ndb.PDF.highlighted_doc(pdf.reference(0))
    assert isinstance(highlighted_doc, fitz.Document)
    assert not highlighted_doc.is_closed


def find_table_on_page(df, table_string, page):
    page = page - 1  # pages in metadata are 0 indexed
    found_table = False
    for _, row in df[df.page == page].iterrows():
        if table_string in row.display:
            found_table = True
    assert found_table


def test_pdf_table_parsing():
    pdf = ndb.PDF(PRIAXOR_PDF_FILE, version="v2", table_parsing=True)
    df = pdf.table.df

    table_string = "[['fl ozs/A', 'lb\\nfluxapyroaxad/A', 'lb\\npyraclostrobin/A'], ['2', '0 .022', '0 .043'], ['4', '0 .043', '0 .087'], ['5', '0 .054', '0 .109'], ['5 .5', '0 .060', '0 .119'], ['6', '0 .065', '0 .130'], ['6 .9', '0 .075', '0 .150'], ['7 .4', '0 .080', '0 .161'], ['8', '0 .087', '0 .174'], ['8 .2', '0 .089', '0 .178'], ['8 .8', '0 .096', '0 .191'], ['9', '0 .098', '0 .195'], ['9 .2', '0 .100', '0 .200'], ['11', '0 .119', '0 .239'], ['13 .8', '0 .150', '0 .300'], ['16', '0 .174', '0 .348'], ['16 .5', '0 .179', '0 .358'], ['18', '0 .195', '0 .391'], ['20 .7', '0 .225', '0 .450'], ['22 .2', '0 .241', '0 .482'], ['24', '0 .261', '0 .521'], ['24 .6', '0 .267', '0 .534'], ['26 .4', '0 .287', '0 .573'], ['27 .6', '0 .300', '0 .599'], ['44', '0 .478', '0 .956']]"
    find_table_on_page(df, table_string, 5)

    table_string = "[['FIRST AID', None], ['If swallowed', '• Call a poison control center or doctor immediately for treatment advice .\\n• Have person sip a glass of water if able to swallow .\\n• DO NOT induce vomiting unless told to do so by a poison control center or doctor .\\n• DO NOT give anything to an unconscious person .'], ['HOTLINE NUMBER', None], ['Have the product container or label with you when calling a poison control center or doctor or going for treatment .\\nYou may also contact BASF Corporation for emergency medical treatment information: 1-800-832-HELP (4357) .', None]]"
    find_table_on_page(df, table_string, 2)

    table_string = "['USER SAFETY RECOMMENDATIONS'], ['Users should:\\n• Wash hands before eating, drinking, chewing gum,\\nusing tobacco, or using the toilet .\\n• Remove clothing/PPE immediately if pesticide gets\\ninside . Then wash thoroughly and put on clean\\nclothing .\\n• Remove PPE immediately after handling this product .\\nAs soon as possible, wash thoroughly and change into\\nclean clothing .']]"
    find_table_on_page(df, table_string, 2)

    table_string = "[['Crop**', 'Maximum\\nProduct Rate\\nper\\nApplication\\n(fl ozs/A)', 'Maximum\\nNumber of\\nApplications\\nper Year', 'Maximum\\nNumber of\\nSequential\\nApplications', 'Maximum\\nProduct Rate\\nper Year\\n(fl ozs/A)***', 'Minimum\\ntime from\\nApplication to\\nHarvest\\n(PHI) (days)'], ['Alfalfa', '6 .9', '3', '2', '20 .7', '14'], ['Barley', '8', '2', '2', '16', '21'], ['Brassica leafy\\nvegetables crop\\nsubgroups 5A and 5B', '8 .2', '3', '2', '24 .6', '3'], ['Citrus fruit', '11', '4', '2', '44', '0'], ['Corn', '8', '2', '2', '16', '21\\n7 (sweet)'], ['Cotton', '8', '3', '2', '24', '30'], ['Dried shelled peas\\nand beans\\n(except soybeans)\\ncrop group 6C', '8', '2', '2', '16', '21'], ['Edible-podded\\nlegume vegetables\\ncrop subgroup 6A', '8', '2', '2', '16', '7'], ['Fruiting vegetables', '8', '3', '2', '24', '0'], ['Grass grown for seed', '6 .9', '2', '2', '13 .8', '14'], ['Oats', '8', '2', '2', '16', '21'], ['Oilseed crops', '8', '2', '2', '16', '21'], ['Peanut', '8', '3', '2', '24', '14'], ['Rye', '8', '2', '2', '16', '21'], ['Sorghum and millet', '8', '1', '1', '8', '21'], ['Soybean', '8', '2', '2', '16', '21'], ['Succulent shelled\\npeas and beans', '8', '2', '2', '16', '7'], ['Sugar beet', '8', '3', '2', '24', '7'], ['Sugarcane', '9', '3', '2', '18', '14'], ['Tuberous and corm\\nvegetables (potato)', '8', '3', '2', '24', '7'], ['Wheat and triticale', '8', '2', '2', '16', '21'], ['* See Table 2. Crop-specific Directions: Foliar Applications for additional directions .\\n** For a complete list of crops, see Table 2. Crop-specific Directions: Foliar Applications .\\n*** The maximum product rate per year includes the combination of in-furrow, soil-directed and foliar uses .', None, None, None, None, None]]"
    find_table_on_page(df, table_string, 10)

    table_string = "[['Crop', 'Target Disease', 'Product Use\\nRate per\\nApplication\\n(fl ozs/A)', 'Maximum\\nNumber of\\nApplications\\nper Year', 'Maximum\\nProduct Rate\\nper Year\\n(fl ozs/A)', 'Minimum\\nTime from\\nApplication\\nto Harvest\\n(PHI) (days)'], ['Barley', 'Black point\\n(Kernel blight or Head mold)\\n(Cochliobolus sativus,\\nAlternaria spp .)\\nLeaf rust\\n(Puccinia spp .)\\nNet blotch\\n(Pyrenophora teres)\\nPowdery mildew\\n(Blumeria graminis f . sp .\\nhordei)\\nScald\\n(Rhynchosporium secalis)\\nSeptoria leaf and\\nglume blotch\\n(Septoria spp .,\\nStagonospora spp .)\\nSpot blotch\\n(Cochliobolus sativus)\\nStem rust\\n(Puccinia graminis f . sp .\\ntritici)\\nStripe rust\\n(Puccinia striiformis)\\nTan spot (Yellow leaf spot)\\n(Pyrenophora spp .)', '4 to 8*', '2', '16', 'Apply no later\\nthan 50% head\\nemergence\\n(Feekes 10 .3,\\nZadok’s 55)\\nbut no less\\nthan 21 days\\nbefore harvest'], ['Application Directions. For optimal disease control, begin foliar applications of Priaxor® Xemium® brand\\nfungicide prior to disease development . To maximize yields in cereals, it is important to protect the flag leaf . Apply\\nPriaxor immediately after flag leaf emergence for optimum results .\\nPriaxor does not control Fusarium head blight (head scab) or prevent the reductions in grain quality that can result\\nfrom this disease . When head blight is a concern, growers should manage this disease with fungicides that are labeled\\nfor and effective in managing this disease, and with cultural practices like crop rotation and plowing to reduce crop\\nresidues that serve as an inoculum source .\\nDO NOT harvest barley hay or feed green-chopped barley within 14 days of last application .\\nDO NOT apply more than 16 fl ozs of Priaxor per acre per year (0 .174 lb ai/year fluxapyroxad and 0 .348 lb ai/year\\npyraclostrobin) . DO NOT make more than two (2) sequential applications of Priaxor before alternating to a labeled\\nnon-Group 7 or non-Group 11 fungicide .\\n* F or early season control of net blotch, Septoria leaf and glume blotch, spot blotch, and tan spot when conditions\\nfavor disease development, apply 2 to 4 fl ozs per acre of Priaxor either in combination with a herbicide application\\nor when conditions favor disease development . When the 2 to 4 fl ozs early season application rate is used, a second\\napplication of Priaxor may be required to protect the emerged flag leaf . Environmental conditions for disease or cur-\\nrent disease pressure at the time of flag-leaf emergence should be used to determine the Priaxor rate for the second\\napplication . For high disease pressure, use the higher rate of Priaxor .', None, None, None, None, None]]"
    find_table_on_page(df, table_string, 12)
