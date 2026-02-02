import pypdf

pdf_path = "Shivam Mishra Resume.pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")
