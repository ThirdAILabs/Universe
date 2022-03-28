

import colbert_wrapper

colbert = colbert_wrapper.Colbert("downloads/colbertv2.0")
print(colbert.encodeQuery(["what is paula deen's brother"])[0])
print(colbert.encodeDoc(["The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."])[0])

