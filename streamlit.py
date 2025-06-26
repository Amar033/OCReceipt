import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import pandas as pd
from datetime import datetime
import io
import json 

# Page configuration
st.set_page_config(
    page_title="OCR Invoice Processor",
    page_icon="📄",
    layout="wide"
)

# Function to preprocess image for better OCR
def preprocess_image(image):
    """Preprocess image to improve OCR accuracy"""
    # Convert PIL to OpenCV format
    open_cv_image = np.array(image)
    if len(open_cv_image.shape) == 3:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply threshold to get image with only black and white
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to PIL format
    return Image.fromarray(thresh)

# Function to extract text using Tesseract
@st.cache_data
def extract_text_from_image(_image):
    """Extract text from image using Tesseract OCR"""
    try:
        # Preprocess image
        processed_image = preprocess_image(_image)
        
        # Extract text with detailed data
        text = pytesseract.image_to_string(processed_image, config='--psm 6')
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
        
        return text, ocr_data, processed_image
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", {}, image

# Function to extract invoice information using regex patterns
def extract_invoice_info(text):
    """Extract invoice information using regex patterns"""
    info = {}
    text_upper = text.upper()
    
    # Invoice Number patterns
    invoice_patterns = [
        r'INVOICE\s*(?:NUMBER|NO|#)?\s*:?\s*([A-Z0-9\-]+)',
        r'INV\s*(?:NO|#)?\s*:?\s*([A-Z0-9\-]+)',
        r'INVOICE\s*([A-Z0-9\-]+)',
        r'#\s*([A-Z0-9\-]+)'
    ]
    
    for pattern in invoice_patterns:
        match = re.search(pattern, text_upper)
        if match:
            info['invoice_number'] = match.group(1)
            break
    
    # Date patterns
    date_patterns = [
        r'DATE\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        r'(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})'
        r'(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{2,4})'
    ]
    
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text_upper)
        dates_found.extend(matches)
    
    if dates_found:
        info['invoice_date'] = dates_found[0]
        if len(dates_found) > 1:
            info['due_date'] = dates_found[1]
    
    # Total amount patterns
    total_patterns = [
        r'TOTAL\s*:?\s*\$?([0-9,]+\.?\d*)',
        r'AMOUNT\s*:?\s*\$?([0-9,]+\.?\d*)',
        r'BALANCE\s*:?\s*\$?([0-9,]+\.?\d*)',
        r'\$\s*([0-9,]+\.?\d*)',
        r'([0-9,]+\.\d{2})\s*$'
    ]
    
    for pattern in total_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            # Get the largest amount (likely the total)
            amounts = [float(amt.replace(',', '')) for amt in matches if amt.replace(',', '').replace('.', '').isdigit()]
            if amounts:
                info['total_amount'] = max(amounts)
                break
    
    # Company/Supplier name (usually at the top)
    lines = text.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if non_empty_lines:
        # First few lines often contain company name
        for line in non_empty_lines[:5]:
            if len(line) > 3 and not re.match(r'^\d', line):
                info['supplier_name'] = line
                break
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        info['email'] = email_match.group()
    
    # Phone pattern
    phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        info['phone'] = phone_match.group()
    
    return info

# Function to extract line items
def extract_line_items(text):
    """Extract line items from invoice text"""
    lines = text.split('\n')
    items = []
    
    # Look for lines that might contain items (with quantities and prices)
    for line in lines:
        # Pattern for line items: description, quantity, price
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

# Main app
st.title("📄 OCR Invoice Processor")
st.markdown("Upload an invoice image to extract information using Tesseract OCR")

# Installation instructions
with st.expander("📋 Installation Requirements"):
    st.markdown("""
    **Required packages:**
    ```bash
    pip install streamlit pytesseract pillow opencv-python pandas numpy
    ```
    
    **Tesseract Installation:**
    - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
    - **macOS**: `brew install tesseract`
    - **Ubuntu**: `sudo apt install tesseract-ocr`
    
    **Note:** You may need to set the tesseract path in your system if it's not in PATH.
    """)

# OCR Configuration
st.sidebar.subheader("🔧 OCR Settings")
ocr_config = st.sidebar.selectbox(
    "OCR Mode",
    options=[
        "--psm 6",  # Uniform block of text
        "--psm 4",  # Single column of text
        "--psm 3",  # Fully automatic page segmentation
        "--psm 8",  # Single word
        "--psm 11", # Sparse text
        "--psm 12"  # Sparse text with OSD
    ],
    help="PSM (Page Segmentation Mode) affects how Tesseract interprets the image"
)

preprocess_option = st.sidebar.checkbox("Enable Image Preprocessing", value=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an invoice image", 
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(image, caption="Original Invoice", use_container_width=True)
        
        # Image info
        st.info(f"**File:** {uploaded_file.name}")
        st.info(f"**Size:** {uploaded_file.size:,} bytes")
        st.info(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
    
    # Process button
    if st.button("🔍 Extract Information", type="primary"):
        with st.spinner("Extracting text using OCR..."):
            try:
                # Extract text
                if preprocess_option:
                    text, ocr_data, processed_image = extract_text_from_image(image)
                    
                    with col2:
                        st.subheader("⚙️ Processed Image")
                        st.image(processed_image, caption="Preprocessed for OCR", use_container_width=True)
                else:
                    text = pytesseract.image_to_string(image, config=ocr_config)
                    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                if not text.strip():
                    st.error("No text detected in the image. Try adjusting OCR settings or using a clearer image.")
                    # return
                
                # Extract invoice information
                invoice_info = extract_invoice_info(text)
                line_items = extract_line_items(text)
                
                st.success("✅ Text extraction completed!")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 ,tab5= st.tabs(["📊 Extracted Info", "📋 Line Items", "📝 Raw Text", "🔍 OCR Details","🗂 Export Data as JSON"])
                
                with tab1:
                    st.subheader("📊 Extracted Invoice Information")
                    
                    # Display extracted info in metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total = invoice_info.get('total_amount', 'N/A')
                        st.metric("Total Amount", f"${total:.2f}" if isinstance(total, (int, float)) else total)
                    
                    with col2:
                        st.metric("Invoice Number", invoice_info.get('invoice_number', 'N/A'))
                    
                    with col3:
                        st.metric("Invoice Date", invoice_info.get('invoice_date', 'N/A'))
                    
                    with col4:
                        st.metric("Due Date", invoice_info.get('due_date', 'N/A'))
                    
                    # Additional information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏢 Company Information")
                        st.write(f"**Supplier:** {invoice_info.get('supplier_name', 'N/A')}")
                        st.write(f"**Email:** {invoice_info.get('email', 'N/A')}")
                        st.write(f"**Phone:** {invoice_info.get('phone', 'N/A')}")
                    
                    with col2:
                        st.subheader("📋 All Extracted Fields")
                        for key, value in invoice_info.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                with tab2:
                    st.subheader("📋 Line Items")
                    if line_items:
                        # Create DataFrame for line items
                        df = pd.DataFrame(line_items)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary
                        st.subheader("📊 Items Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Items", len(line_items))
                        with col2:
                            total_qty = sum(item['quantity'] for item in line_items)
                            st.metric("Total Quantity", total_qty)
                        with col3:
                            total_amount = sum(item['total'] for item in line_items)
                            st.metric("Items Total", f"${total_amount:.2f}")
                    else:
                        st.info("No line items detected. Try adjusting the image quality or OCR settings.")
                        st.markdown("**Tips for better line item detection:**")
                        st.markdown("- Ensure the image is clear and well-lit")
                        st.markdown("- Line items should be in a tabular format")
                        st.markdown("- Include quantity and price information")
                
                with tab3:
                    st.subheader("📝 Raw Extracted Text")
                    st.text_area("OCR Text Output", text, height=400)
                    
                    # Download raw text
                    st.download_button(
                        label="📥 Download Raw Text",
                        data=text,
                        file_name=f"extracted_text_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                
                with tab4:
                    st.subheader("🔍 OCR Analysis Details")
                    
                    if ocr_data:
                        # Confidence analysis
                        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                        if confidences:
                            avg_confidence = sum(confidences) / len(confidences)
                            st.metric("Average OCR Confidence", f"{avg_confidence:.1f}%")
                            
                            # Confidence distribution
                            confidence_ranges = {
                                "High (80-100%)": len([c for c in confidences if c >= 80]),
                                "Medium (60-79%)": len([c for c in confidences if 60 <= c < 80]),
                                "Low (0-59%)": len([c for c in confidences if c < 60])
                            }
                            
                            st.bar_chart(confidence_ranges)
                        
                        # Word count
                        words = [word for word in ocr_data['text'] if word.strip()]
                        st.metric("Words Detected", len(words))
                    
                    # Processing settings used
                    st.subheader("⚙️ Processing Settings")
                    st.write(f"**OCR Config:** {ocr_config}")
                    st.write(f"**Preprocessing:** {'Enabled' if preprocess_option else 'Disabled'}")
                with tab5:
                    st.subheader("🗂 Export Data as JSON")
                    output_data = {
                        "invoice_info": invoice_info,
                        "line_items": line_items,
                        "raw_text": text
                    }
                    json_output = json.dumps(output_data, indent=4)
                    st.text_area("JSON Output", json_output, height=300)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_output,
                        file_name=f"invoice_data_{uploaded_file.name}.json",
                        mime="application/json"
                    )


                                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.info("Common solutions:")
                st.markdown("""
                - Ensure Tesseract is properly installed
                - Try a different OCR mode
                - Use a clearer, higher resolution image
                - Enable/disable image preprocessing
                """)

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses **Tesseract OCR** to extract text from invoice images and then uses pattern matching to identify key information.
    
    **Features:**
    - Text extraction using OCR
    - Invoice information parsing
    - Line item detection
    - Image preprocessing
    - Confidence analysis
    """)
    
    st.header("📈 Tips for Better Results")
    st.markdown("""
    **Image Quality:**
    - Use high-resolution images
    - Ensure good lighting
    - Avoid skewed or rotated images
    - Clear, legible text
    
    **OCR Settings:**
    - Try different PSM modes
    - Enable preprocessing for noisy images
    - Adjust settings based on document layout
    """)
    
    st.header("🔧 Troubleshooting")
    st.markdown("""
    **If OCR fails:**
    1. Check Tesseract installation
    2. Verify image format is supported
    3. Try different OCR modes
    4. Ensure image text is clear
    """)