from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Introduction to Human Anatomy", ln=True, align='C')
pdf.cell(200, 10, txt="", ln=True)
pdf.cell(200, 10, txt="The human body is made up of many systems.", ln=True)
pdf.cell(200, 10, txt="The skeletal system provides structure and support.", ln=True)
pdf.cell(200, 10, txt="The nervous system controls body functions.", ln=True)
pdf.cell(200, 10, txt="The cardiovascular system pumps blood.", ln=True)
pdf.cell(200, 10, txt="The respiratory system helps in breathing.", ln=True)
pdf.cell(200, 10, txt="The digestive system breaks down food.", ln=True)

pdf.output("document.pdf")
print("PDF created successfully!")