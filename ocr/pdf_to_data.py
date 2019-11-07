import io
import sys

import pdf2image

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

def image_convert(pdf_path, output_name):
    """
    Save each page of pdf as image.
    """
    pages = pdf2image.convert_from_path(pdf_path)
    for i, page in enumerate(pages):
        file_name = '{}.{}.jpg'.format(output_name, i)
        print('Saving', file_name)
        page.save(file_name, 'JPEG')

def extract_text_by_page(pdf_path):
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
            yield text
            # close open handles
            converter.close()
            fake_file_handle.close()

def extract_text(pdf_path, output_name):
    """
    Saved each page of pdf file as text.
    """
    for i, page in enumerate(extract_text_by_page(pdf_path)):
        file_name = '{}.{}.txt'.format(output_name, i)
        print('Writing', file_name)
        with open(file_name, 'w') as f:
            f.write(page)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: {} <pdf> <output>'.format(sys.argv[0]))
        sys.exit()

    pdf_path = sys.argv[1]
    output_name = sys.argv[2]
    image_convert(pdf_path, output_name)
    extract_text(pdf_path, output_name)
