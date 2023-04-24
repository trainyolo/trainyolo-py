from trainyolo.client import OCRResult
import re

def read_f1_conf(client, f1_curve):
    ocr_result = OCRResult.create(client, f1_curve)
    status, result = ocr_result.get_result()
    if status == 'SUCCEEDED':
        pattern = re.compile(r'all\s+classes\s+(\d+\.\d+)\s+at\s+(\d+\.\d+)')
        match = pattern.search(result)
        if match:
            score, conf = match.groups()
            conf = float(conf)
            if conf < 1 and conf > 0:
                return conf
            else:
                return -1
    else:
        return -1