from trainyolo.client import OCRResult

def read_f1_conf(client, f1_curve):
    ocr_result = OCRResult.create(client, f1_curve)
    status, result = ocr_result.get_result()
    if status == 'SUCCEEDED':
        conf = float(result.rsplit('all classes', 1)[1].rsplit('at', 1)[1])
        if conf < 1 and conf > 0:
            return conf
        else:
            return -1
    else:
        return -1