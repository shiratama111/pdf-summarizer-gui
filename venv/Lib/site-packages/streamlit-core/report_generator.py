from docx import Document
import docx
from docxtpl import DocxTemplate
from docx2pdf import convert
from jinja2 import Undefined
import jinja2
import os
import subprocess
import random
import string
from pathlib import Path


class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        print(f"Undefined error at: {self._undefined_name}")
        return ''

    __add__ = __radd__ = __sub__ = __rsub__ = _fail_with_undefined_error
    __mul__ = __rmul__ = __div__ = __rdiv__ = _fail_with_undefined_error
    __truediv__ = __rtruediv__ = _fail_with_undefined_error
    __floordiv__ = __rfloordiv__ = _fail_with_undefined_error
    __mod__ = __rmod__ = _fail_with_undefined_error
    __pos__ = __neg__ = _fail_with_undefined_error
    __call__ = __getitem__ = _fail_with_undefined_error
    __lt__ = __le__ = __gt__ = __ge__ = _fail_with_undefined_error
    __int__ = __float__ = __complex__ = _fail_with_undefined_error
    __pow__ = __rpow__ = __round__ = __abs__ = _fail_with_undefined_error


def random_string(length=10):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class ReportGenerator:
    """
    The ReportGenerator class is used to generate a report by replacing placeholders in a Word document with variables.

    Attributes:
        assets_dir    (str): The directory where the image files are located.
        template_path (str): The path to the Word document that serves as the template for the report.
        output_path   (str): The path where the generated report should be saved.
    """

    def __init__(self, assets_dir, output_path, delete_last_report=False, keep_word_file=False):
        self.assets_dir = Path(assets_dir)
        self.output_path = Path(output_path)
        self.last_report_path = None
        self.delete_last_report = delete_last_report
        self.keep_word_file = keep_word_file

    def generate(self, template_path, variables):
        """
        The method to generate the report.

        It loads the Word document from the template path, replaces the placeholders with the images or variables,
        renders the changes, and saves the modified document to the output path.
        """

        if self.last_report_path and self.delete_last_report:
            try:
                self.last_report_path.unlink()
            except FileNotFoundError:
                print(f"Last report file not found: {self.last_report_path}")

        doc = DocxTemplate(template_path)
        ran_string = random_string()

        # Get all .png files in the assets directory
        image_files = [f.stem for f in self.assets_dir.glob('*.png')]

        # Get all alt images text from the docx file
        alt_texts = _extract_alt_text_from_docx(template_path)

        # Get list of images to replace
        images_to_replace = [f for f in image_files if f in alt_texts]

        # Replace the placeholders with the images
        for var in images_to_replace:
            print('Replacing:', var)
            image_path = self.assets_dir / f'{var}.png'
            if image_path.is_file():
                try:
                    doc.replace_pic(var, str(image_path))
                except (ValueError, FileNotFoundError):
                    pass
            else:
                print(f"Image file not found: {image_path}")

        def number_format(num):
            if abs(num) >= 100:
                res = "{:.0f}".format(num)
            elif abs(num) >= 10:
                res = "{:.1f}".format(num)
            else:
                res = "{:.2f}".format(num)
            return res.replace(',', 'X').replace('.', ',').replace('X', '.')

        print("Rendering document")
        # Render the changes
        jinja_env = jinja2.Environment(undefined=SilentUndefined)
        jinja_env.filters['nformat'] = number_format
        doc.render(context=variables, jinja_env=jinja_env)

        # If output_path folder does not exist
        if not self.output_path.exists():
            print('Output folder does not exist!', self.output_path)
            return

        docx_file = self.output_path / f'report_{ran_string}.docx'

        # Save last report path
        self.last_report_path = self.output_path / f'report_{ran_string}.pdf'

        # Save the document and convert it to PDF
        try:
            print(f'Saving to: {docx_file}')
            
            doc.save(docx_file)
        except Exception as e:
            print("Error in saving the document! \n", e)
            return

        # Convert to PDF
        try:
            command = [
                'soffice',
                '--headless',
                '--convert-to',
                'pdf',
                '--outdir',
                str(self.output_path),
                str(docx_file),
            ]
            print("Running command:", ' '.join(command))

            result = subprocess.run(command, capture_output=True, text=True, timeout=500)

            if result.returncode != 0:
                raise Exception(f"Conversion failed with exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            print("The process timed out!")
        except Exception as e:
            print("Error in converting to PDF! \n", e)

        # Remove the DOCX file
        if not self.keep_word_file:
            docx_file.unlink()
        pdf_file = docx_file.with_suffix('.pdf')

        if not pdf_file.exists():
            print(f"PDF file was not created: {pdf_file}")
            return None

        return pdf_file


def _extract_alt_text_from_docx(docx_file):
    alt_text_list = []
    doc = Document(docx_file)

    # Process headers
    for section in doc.sections:
        for header in section.header.part.element.iter():
            if header.tag.endswith('}cNvPr'):
                alt_text = header.get('descr')
                if alt_text:
                    alt_text_list.append(alt_text)

    # Process main document content
    for shape in doc.inline_shapes:
        alt_text = shape._inline.graphic.graphicData.pic.nvPicPr.cNvPr.get('descr')
        if alt_text:
            alt_text_list.append(alt_text)

    # Process footers
    for section in doc.sections:
        for footer in section.footer.part.element.iter():
            if footer.tag.endswith('}cNvPr'):
                alt_text = footer.get('descr')
                if alt_text:
                    alt_text_list.append(alt_text)

    return alt_text_list
