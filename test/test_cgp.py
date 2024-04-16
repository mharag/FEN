from utils.cgp import CGPTranslator


RAW_CGP = "{2,1,0,0,2,1,0}([2]0,1,1)([3]0,2,1)(3)"

def test_cgp():
    translator = CGPTranslator(False)
    graph = translator.parse(RAW_CGP)
    reconstructed = translator.export(graph)
    assert RAW_CGP == reconstructed

    translator = CGPTranslator(True)
    graph = translator.parse(RAW_CGP)
    reconstructed = translator.export(graph)
    assert RAW_CGP == reconstructed
