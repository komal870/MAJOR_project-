
from fpdf import FPDF

class HealthReport:
    def __init__(self, patient_name, details: dict):
        self.patient_name = patient_name
        self.details = details

    def create_pdf(self, filename='health_report.pdf'):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0,10,'Health Report', ln=True, align='C')
        pdf.ln(6)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0,8, f'Name: {self.patient_name}', ln=True)
        pdf.ln(4)
        for k,v in self.details.items():
            pdf.multi_cell(0,8, f'{k}: {v}')
        pdf.output(filename)
        return filename

