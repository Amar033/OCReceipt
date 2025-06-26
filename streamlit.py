import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import pandas as pd
import json

# Page config
st.set_page_config(page_title="Invoice OCR Processor", page_icon="📄", layout="wide")

# Image Preprocessing Function
def preprocess_image(image):
    open_cv_image = np.array(image)
    if len(open_cv_image.shape) == 3:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

# Updated Extraction Function
def extract_invoice_info(text):
    info = {}
    text_upper = text.upper()

    # Invoice Number Patterns
    invoice_patterns = [
        r'INVOICE\s*(?:NO\.?|NUMBER)?\s*:?\s*(\d{3,})',
        r'INV\s*(?:NO\.?|#)?\s*:?\s*(\d{3,})',
        r'#\s*(\d{3,})'
    ]
    for pattern in invoice_patterns:
        match = re.search(pattern, text_upper)
        if match:
            info['invoice_number'] = match.group(1)
            break

    # Date Patterns
    date_patterns = [
        r'(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{4})',
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})'
    ]
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text_upper)
        dates_found.extend(matches)
    if dates_found:
        info['invoice_date'] = dates_found[0]
        if len(dates_found) > 1:
            info['due_date'] = dates_found[1]

    # Total Amount Patterns
    total_patterns = [
        r'TOTAL\s*\$?\s*([0-9,]+\.?\d*)',
        r'AMOUNT\s*\$?\s*([0-9,]+\.?\d*)',
        r'BALANCE\s*\$?\s*([0-9,]+\.?\d*)',
        r'\$\s*([0-9,]+\.?\d*)'
    ]
    for pattern in total_patterns:
        matches = re.findall(pattern, text_upper)
        amounts = [float(amt.replace(',', '')) for amt in matches if amt.replace(',', '').replace('.', '').isdigit()]
        if amounts:
            info['total_amount'] = max(amounts)
            break

    # Supplier Name Extraction
    supplier_match = re.search(r'BILLED\s*TO:\s*\n?(.+?)\n', text, re.IGNORECASE)
    if supplier_match:
        info['supplier_name'] = supplier_match.group(1).strip()

    # Phone Extraction
    phone_pattern = r'(\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'
    phones = re.findall(phone_pattern, text)
    if phones:
        info['phone'] = phones[0]

    # Email Extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        info['email'] = email_match.group()

    return info

# Line Items Extraction
def extract_line_items(text):
    lines = text.split('\n')
    items = []
    for line in lines:
        item_pattern = r'(.+?)\s+(\d+)\s+\$?([0-9,]+\.?\d*)\s+\$?([0-9,]+\.?\d*)$'
        match = re.search(item_pattern, line.strip())
        if match:
            items.append({
                'description': match.group(1).strip(),
                'quantity': int(match.group(2)),
                'unit_price': float(match.group(3).replace(',', '')),
                'total': float(match.group(4).replace(',', ''))
            })
    return items

# OCR Text Extraction
def extract_text_from_image(image, config):
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image, config=config)
    ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
    return text, ocr_data, processed_image

# Streamlit UI
st.title("📄 Invoice OCR Processor")

ocr_config = st.sidebar.selectbox("OCR Mode", ["--psm 6", "--psm 4", "--psm 3", "--psm 11"], index=0)
uploaded_file = st.file_uploader("Upload Invoice Image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Invoice", use_container_width=True)

    if st.button("🔍 Extract Information"):
        with st.spinner("Processing..."):
            text, ocr_data, processed_image = extract_text_from_image(image, ocr_config)
            invoice_info = extract_invoice_info(text)
            line_items = extract_line_items(text)

        st.success("Extraction Complete!")
        tab1, tab2, tab3, tab4 = st.tabs(["Invoice Info", "Line Items", "Raw Text", "Export JSON"])

        with tab1:
            st.subheader("Extracted Invoice Info")
            for key, val in invoice_info.items():
                st.write(f"**{key.replace('_',' ').title()}:** {val}")

        with tab2:
            st.subheader("Line Items")
            if line_items:
                df = pd.DataFrame(line_items)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No line items detected.")

        with tab3:
            st.subheader("Raw OCR Text")
            st.text_area("Text Output", text, height=300)

        with tab4:
            output_json = {
                "invoice_info": invoice_info,
                "line_items": line_items,
                "raw_text": text
            }
            json_str = json.dumps(output_json, indent=4)
            st.text_area("JSON Output", json_str, height=300)
            st.download_button("Download JSON", data=json_str, file_name="invoice_data.json", mime="application/json")

