import PyPDF2

def extract_text_from_pdf(file_path):
    # Open the PDF file in binary mode
    with open(file_path, 'rb') as file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfFileReader(file)

        # Initialize an empty string to hold the extracted text
        text = ''

        # Loop through each page in the PDF and extract the text
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()

    return text

# Usage
file_path = 'C:\Users\Jagatheesh\OneDrive\Desktop\Heriot_Watt_University_Msc_Data_Science\F21MP_Dissertation\Report_Backups\Final_Submission\Jagatheesh_H00424118_F21RP_Dissertation_Report.pdf'
text = extract_text_from_pdf(file_path)
print(text)
