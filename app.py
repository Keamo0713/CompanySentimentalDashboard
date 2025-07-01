import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import re # Import regex for keyword highlighting
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np # For numpy arrays in confusion matrix plotting
import datetime # For timestamp in history
from fpdf import FPDF # Import FPDF for PDF generation

# Mock functions for sentiment analysis and keyword extraction
# In a real scenario, you'd integrate with the actual Gemini API.
def get_sentiment(text):
    """
    Mocks a sentiment analysis API call.
    In a real application, this would call the Google Gemini API.
    """
    text_lower = text.lower()
    if any(word in text_lower for word in ['happy', 'great', 'fantastic', 'excellent', 'love', 'enjoyed', 'amazing', 'loved']):
        return {'sentiment': 'Positive', 'confidence': 0.9}
    elif any(word in text_lower for word in ['bad', 'disappointed', 'terrible', 'broke', 'unhelpful', 'rude', 'nightmare', 'buggy', 'crashes', 'poor', 'incorrect']):
        return {'sentiment': 'Negative', 'confidence': 0.85}
    else:
        return {'sentiment': 'Neutral', 'confidence': 0.7}

def extract_keywords(text):
    """
    Mocks a keyword extraction API call.
    In a real application, this would call the Google Gemini API.
    """
    keywords = []
    text_lower = text.lower()
    common_keywords = ["fantastic", "product", "happy", "quality", "broke", "disappointed",
                       "service", "excellent", "enjoyed", "movie", "acting", "unhelpful",
                       "rude", "average", "book", "software", "buggy", "crashes", "poor", "incorrect", "order", "return process", "nightmare", "features", "price"]
    
    for kw in common_keywords:
        if kw in text_lower:
            keywords.append(kw)
    
    return list(set(keywords)) # Return unique keywords

# --- PDF Generation Functions ---
def create_single_text_analysis_pdf(analysis_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Single Text Sentiment Analysis Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Timestamp: {analysis_data['Timestamp']}")
    pdf.multi_cell(0, 10, f"Original Text: {analysis_data['Original Text']}")
    pdf.multi_cell(0, 10, f"Sentiment: {analysis_data['Sentiment']}")
    pdf.multi_cell(0, 10, f"Confidence: {analysis_data['Confidence']}")
    pdf.multi_cell(0, 10, f"Keywords: {analysis_data['Keywords']}")

    return pdf.output(dest='S') # Return bytes directly, FPDF handles encoding internally

def create_batch_analysis_pdf(df_results, file_name, sentiment_counts):
    pdf = FPDF(orientation='L') # Landscape for wider tables
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Batch Sentiment Analysis Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"File Analyzed: {file_name}")
    pdf.multi_cell(0, 10, f"Total Texts Processed: {len(df_results)}")
    pdf.multi_cell(0, 10, f"Sentiment Distribution: Positive={sentiment_counts.get('Positive', 0)}, Neutral={sentiment_counts.get('Neutral', 0)}, Negative={sentiment_counts.get('Negative', 0)}")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 10)
    # Define column widths
    col_widths = [120, 30, 25, 100] # Original Text, Sentiment, Confidence, Keywords

    # Table Header
    for i, header in enumerate(df_results.columns):
        pdf.cell(col_widths[i], 10, str(header), 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", "", 8)
    for index, row in df_results.iterrows():
        pdf.multi_cell(col_widths[0], 5, str(row["Original Text"]), 1, 'L', 0)
        # Position for next cells in the same row
        x = pdf.get_x() + col_widths[0]
        y = pdf.get_y() - 5 # move up to align with first cell's top
        
        pdf.set_xy(x, y)
        pdf.multi_cell(col_widths[1], 5, str(row["Sentiment"]), 1, 'C', 0)
        x += col_widths[1]
        pdf.set_xy(x, y)
        pdf.multi_cell(col_widths[2], 5, f"{row['Confidence']:.2f}", 1, 'C', 0)
        x += col_widths[2]
        pdf.set_xy(x, y)
        pdf.multi_cell(col_widths[3], 5, str(row["Keywords"]), 1, 'L', 0)
        pdf.ln(5) # Move to next line for the next row of data

    return pdf.output(dest='S') # Return bytes directly, FPDF handles encoding internally


def create_accuracy_report_pdf(accuracy_data, accuracy, report_df, cm_df):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Model Accuracy Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Overall Model Accuracy: {accuracy:.2%}", 0, 1)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Comparison: Manual vs. Gemini API Predictions", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    # Table for comparison data
    comparison_df = pd.DataFrame(accuracy_data['comparison_data'])
    # Reduce column width for 'Text' to fit page
    comparison_df['Text'] = comparison_df['Text'].apply(lambda x: x[:70] + "..." if len(x) > 70 else x) 
    
    # IMPORTANT FIX: Replace emojis with text equivalents to avoid UnicodeEncodeError in FPDF
    comparison_df['Match'] = comparison_df['Match'].replace({'✅ Yes': 'Yes', '❌ No': 'No'})

    # Headers
    pdf.set_font("Arial", "B", 8)
    col_widths = [75, 30, 30, 20, 15] # Text, Manual, Predicted, Confidence, Match
    for col in comparison_df.columns:
        pdf.cell(col_widths[comparison_df.columns.get_loc(col)], 7, col, 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", "", 8)
    for index, row in comparison_df.iterrows():
        for i, col_val in enumerate(row.values):
            pdf.cell(col_widths[i], 7, str(col_val), 1, 0, 'L')
        pdf.ln()
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Classification Report", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    # Classification Report Table
    # Headers
    pdf.set_font("Arial", "B", 10)
    pdf.cell(40, 7, "Metric", 1, 0, 'C')
    for col in report_df.columns:
        pdf.cell(25, 7, col, 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for index, row in report_df.iterrows():
        pdf.cell(40, 7, index, 1, 0, 'L')
        for val in row.values:
            pdf.cell(25, 7, f"{val:.2f}", 1, 0, 'C')
        pdf.ln()
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Confusion Matrix (Counts)", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    # Confusion Matrix Table
    # Headers
    pdf.set_font("Arial", "B", 10)
    pdf.cell(40, 7, "", 1, 0, 'C') # Empty cell for row header
    for col in cm_df.columns:
        pdf.cell(25, 7, col.replace("Predicted ", ""), 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for index, row in cm_df.iterrows():
        pdf.cell(40, 7, index.replace("True ", ""), 1, 0, 'L')
        for val in row.values:
            pdf.cell(25, 7, str(val), 1, 0, 'C')
        pdf.ln()
    pdf.ln(10)


    return pdf.output(dest='S') # Return bytes directly, FPDF handles encoding internally


def create_history_pdf(history_df): # This expects the flattened history_df_for_export
    pdf = FPDF(orientation='L') # Landscape for wider tables
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Analysis History Report", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 7)
    # Define headers and corresponding widths for PDF table
    # These must match the final_pdf_columns used in history_df_for_export
    headers = [
        "Timestamp", "Type", "Text Snippet", "Sentiment", "Confidence", "Keywords",
        "File Name", "Total Texts", "Positive", "Negative", "Neutral",
        "Accuracy", "Samples" # Simplified names for PDF display
    ]
    col_widths = [20, 20, 50, 18, 18, 40, 25, 18, 18, 18, 18, 20, 20] 

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", "", 6)
    
    for index, row in history_df.iterrows():
        row_height = 7 # Base height
        
        # Text Snippet and Keywords are long, might need multi_cell
        text_snippet = str(row.get("Text Snippet", "N/A"))
        keywords = str(row.get("Keywords", "N/A"))
        
        # Truncate for PDF to ensure fit
        display_text_snippet = text_snippet[:60] + "..." if len(text_snippet) > 60 else text_snippet
        display_keywords = keywords[:50] + "..." if len(keywords) > 50 else keywords

        start_x = pdf.get_x()
        start_y = pdf.get_y()

        # Iterate through actual columns of the DataFrame, matched by position to headers
        # This relies on history_df having `final_pdf_columns` in order
        
        for i, col_name in enumerate([ # Manually list columns to ensure order for get() and mapping
            "Timestamp", "Type", "Text Snippet", "Sentiment", "Confidence", "Keywords",
            "File Name", "Total Texts", "Positive", "Negative", "Neutral",
            "Overall Accuracy", "Processed Samples" # These are the DataFrame column names
        ]):
            display_value = str(row.get(col_name, "N/A"))
            width = col_widths[i]

            # Specific handling for Text Snippet and Keywords
            if col_name == "Text Snippet":
                display_value = display_text_snippet
            elif col_name == "Keywords":
                display_value = display_keywords
            elif col_name == "Overall Accuracy":
                display_value = row.get("Overall Accuracy", "N/A").replace("Accuracy:", "").strip() # Clean for PDF
            elif col_name == "Processed Samples":
                display_value = str(row.get("Processed Samples", "N/A")) # Ensure string

            # Determine cell alignment
            align = 'C' if col_name in ["Total Texts", "Positive", "Negative", "Neutral", "Overall Accuracy", "Processed Samples"] else 'L'

            if col_name in ["Text Snippet", "Keywords"]:
                current_x = pdf.get_x()
                current_y = pdf.get_y()
                pdf.multi_cell(width, 3.5, display_value, 1, 'L', 0)
                # Reset position for next cell in the same row
                pdf.set_xy(current_x + width, current_y)
            else:
                pdf.cell(width, row_height, display_value, 1, 0, align)
        pdf.ln(row_height) # Move to next line for the next row of data

    return pdf.output(dest='S')


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide", # Use wide layout for better visualization space
    initial_sidebar_state="expanded"
)

# --- Inject custom CSS for a better font and theme adjustments ---
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            font-weight: 700; /* Make headings bolder */
            color: #262730; /* Darker color for headings */
        }
        .stApp {
            background-color: #f0f2f6; /* Light grey background for a softer look */
        }
        .stButton>button {
            border-radius: 8px;
            border: 1px solid #2196F3; /* Blue border */
            color: white;
            background-color: #2196F3; /* Blue background */
            padding: 10px 20px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #1976D2; /* Darker blue on hover */
            border-color: #1976D2; /* Darker blue border on hover */
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stSelectbox>div>div>div {
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .st-emotion-cache-1avcm0n { /* Main content area padding */
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1em; /* Larger font for tab titles */
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px; /* Space between tabs */
        }
        .stTabs [data-baseweb="tab-list"] button {
            background-color: #e0e0e0; /* Light grey for inactive tabs */
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            transition: background-color 0.3s ease;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #2196F3; /* Blue for active tab */
            color: white;
            border-bottom: 3px solid #1565C0; /* Darker blue underline */
        }
        .stAlert {
            border-radius: 8px;
        }
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 25px; /* Increased padding */
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Corrected property name */
            text-align: center;
            height: 100%; /* Ensure cards have equal height in columns */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .metric-card [data-testid="stMetricLabel"] p { /* Target metric label */
            font-size: 1em; /* Keep it concise */
            color: #555;
            margin-bottom: 8px; /* Increased margin-bottom for more space */
        }
        .metric-card [data-testid="stMetricValue"] { /* Target metric value */
            font-size: 1.8em; /* Adjusted font size slightly smaller to alleviate squishing */
            font-weight: 700;
            color: #333;
            line-height: 1.2; /* Add line-height to prevent vertical squishing */
        }
        .stSpinner > div > div {
            border-top-color: #2196F3; /* Customize spinner color to blue */
        }
        .dataframe-container {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden; /* Ensures borders are respected */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Initialize session state for history ---
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = []

# --- Title and Description ---
st.title("😊 AI-Powered Sentiment Analysis Dashboard")
st.markdown(
    """
    Uncover the **emotional tone** of your text data with the power of **Google Gemini**.
    Quickly classify reviews, social media posts, and more into positive, negative, or neutral sentiments,
    and pinpoint the key phrases driving those emotions.
    """
)

st.divider() # Adds a nice visual separator

# --- Create Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✍️ Direct Text Analysis", "📊 Batch File Analysis", "📈 Accuracy Report", "⏳ History", "ℹ️ Model Limitations"])

with tab1: # Direct Text Analysis Tab
    st.header("Analyze a Single Piece of Text")
    st.markdown("Enter any text below to instantly get its sentiment and extract important keywords.")

    text_input = st.text_area(
        "Your Text Here:",
        "This is a fantastic product! I am so happy with my purchase. The quality is great, and the support team was excellent.",
        height=180,
        placeholder="Type or paste the text you want to analyze (e.g., a customer review, email, or social media post).",
        help="The model will determine if the text is Positive, Negative, or Neutral."
    )

    col_btn = st.columns([1, 4]) # Use columns to center the button visually
    with col_btn[0]:
        if st.button("Analyze Sentiment", key="analyze_single_text_btn", use_container_width=True):
            if text_input:
                with st.spinner("Processing your text..."):
                    sentiment_result = get_sentiment(text_input)
                    keywords = extract_keywords(text_input)
                
                # Define highlighted_text before using it
                highlighted_text = text_input
                if keywords:
                    for kw in keywords:
                        # Use regex to find and replace all occurrences of the keyword, case-insensitive
                        highlighted_text = re.sub(
                            r'\b(' + re.escape(kw) + r')\b', # \b for word boundaries, re.escape for special characters
                            r'<span style="background-color: #FFF9C4; font-weight: bold;">\1</span>', # \1 refers to the matched keyword
                            highlighted_text,
                            flags=re.IGNORECASE
                        )

                st.success("Analysis Complete!")

                st.markdown("### Analysis Summary")
                
                # Using columns for metrics with custom styling
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown(f'<div class="metric-card"><h4>Overall Sentiment</h4><p style="color: {"#4CAF50" if sentiment_result["sentiment"] == "Positive" else "#F44336" if sentiment_result["sentiment"] == "Negative" else "#FFC107"}; font-size: 1.8em; font-weight: 700;">{sentiment_result["sentiment"]}</p></div>', unsafe_allow_html=True)
                with metric_col2:
                    st.markdown(f'<div class="metric-card"><h4>Confidence Score</h4><p style="font-size: 1.8em; font-weight: 700;">{sentiment_result["confidence"]:.2f}</p></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Key Insights:")
                st.write("**Identified Keywords (Sentiment Drivers):**")
                if keywords:
                    # Display keywords as badges or a nice list
                    st.write(" ".join([f'<span style="background-color: #e0f7fa; color: #00796b; border-radius: 5px; padding: 5px 10px; margin: 0 5px 5px 0; display: inline-block;">{kw}</span>' for kw in keywords]), unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.write("**Original Text with Keywords Highlighted:**")
                    st.markdown(f'<div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">{highlighted_text}</div>', unsafe_allow_html=True)
                else:
                    st.info("No significant keywords were extracted for this text. The model may not have found specific terms strongly influencing the sentiment.")

                st.markdown("---")
                st.subheader("How was this sentiment determined?")
                if sentiment_result['sentiment'] == 'Positive':
                    st.info("The model detected several positive words and phrases (like 'fantastic', 'happy', 'great', 'excellent') contributing to a favorable overall tone. The high confidence score suggests clear positive indicators.")
                elif sentiment_result['sentiment'] == 'Negative':
                    st.warning("The model identified strong negative indicators (e.g., 'broke', 'disappointed', 'unhelpful', 'rude'). This indicates a critical or unfavorable tone. The confidence score reflects the model's certainty in this negative classification.")
                elif sentiment_result['sentiment'] == 'Neutral':
                    st.info("The text did not contain strong positive or negative cues, or it had a balance of both, leading to a neutral classification. The confidence score of {:.2f} suggests the model found the text largely objective or factual.".format(sentiment_result['confidence']))
                
                # Add download button for single text analysis
                st.markdown("---")
                st.subheader("Download Single Text Analysis Results")
                
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                single_analysis_data = {
                    "Timestamp": current_time,
                    "Original Text": text_input,
                    "Sentiment": sentiment_result['sentiment'],
                    "Confidence": f"{sentiment_result['confidence']:.2f}",
                    "Keywords": ", ".join(keywords) if keywords else "None"
                }

                col_single_dl_1, col_single_dl_2 = st.columns(2)

                analysis_output = f"""
Analysis Type: Single Text Analysis
Timestamp: {single_analysis_data['Timestamp']}
Original Text: {single_analysis_data['Original Text']}
Sentiment: {single_analysis_data['Sentiment']}
Confidence: {single_analysis_data['Confidence']}
Keywords: {single_analysis_data['Keywords']}
"""
                col_single_dl_1.download_button(
                    label="Download as TXT",
                    data=analysis_output.encode("utf-8"),
                    file_name="single_text_sentiment_analysis.txt",
                    mime="text/plain",
                    key="download_single_text_analysis_txt_btn",
                    use_container_width=True
                )
                
                pdf_output_single = create_single_text_analysis_pdf(single_analysis_data)
                col_single_dl_2.download_button(
                    label="Download as PDF",
                    data=pdf_output_single,
                    file_name="single_text_sentiment_analysis.pdf",
                    mime="application/pdf",
                    key="download_single_text_analysis_pdf_btn",
                    use_container_width=True
                )

                # Add to history
                st.session_state['analysis_history'].insert(0, {
                    "Type": "Single Text Analysis",
                    "Timestamp": current_time,
                    "Details": {
                        "Original Text Snippet": text_input[:200] + ("..." if len(text_input) > 200 else ""),
                        "Sentiment": sentiment_result['sentiment'],
                        "Confidence": f"{sentiment_result['confidence']:.2f}",
                        "Keywords": ", ".join(keywords) if keywords else "None"
                    }
                })

            else:
                st.warning("Please enter some text to analyze before clicking the button.")
                st.toast("Input field is empty!", icon="⚠️")

with tab2: # Batch File Analysis Tab
    st.header("Upload a File for Batch Sentiment Analysis")
    st.markdown("Upload a `.txt` or `.csv` file to analyze multiple texts at once. This is ideal for datasets like customer feedback or survey responses.")

    uploaded_file = st.file_uploader(
        "Choose a file:",
        type=["txt", "csv"],
        help="Supported formats: .txt (one line per text) or .csv (select text column)."
    )

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
        st.info(f"File uploaded: **{file_details['FileName']}** ({file_details['FileSize']})")

        df_results = pd.DataFrame()
        sentiment_counts = pd.Series() # Initialize sentiment_counts

        texts_to_process = []
        if uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            texts_to_process = [line.strip() for line in stringio.readlines() if line.strip()]
            st.info(f"Found **{len(texts_to_process)}** lines in the text file. Each line will be analyzed as a separate entry.")
            
            if st.button("Process Text File", key="process_txt_btn", use_container_width=True):
                if not texts_to_process:
                    st.warning("The text file appears to be empty or contains no valid lines.")
                else:
                    st.toast("Starting batch analysis...", icon="🚀")
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, text in enumerate(texts_to_process):
                        status_text.text(f"Processing text {i+1} of {len(texts_to_process)}...")
                        sentiment_res = get_sentiment(text)
                        keywords_res = extract_keywords(text)
                        results_list.append({
                            "Original Text": text,
                            "Sentiment": sentiment_res['sentiment'],
                            "Confidence": sentiment_res['confidence'],
                            "Keywords": ", ".join(keywords_res)
                        })
                        progress_bar.progress((i + 1) / len(texts_to_process))
                    
                    df_results = pd.DataFrame(results_list)
                    st.session_state['df_results'] = df_results # Store in session state
                    status_text.success("Batch processing complete!")
                    st.toast("Batch analysis finished!", icon="✅")

                    # Add to history
                    current_batch_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sentiment_counts_for_history = df_results['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
                    st.session_state['analysis_history'].insert(0, {
                        "Type": "Batch File Analysis",
                        "Timestamp": current_batch_time,
                        "Details": {
                            "File Name": uploaded_file.name,
                            "Total Texts Processed": len(df_results),
                            "Positive Count": int(sentiment_counts_for_history.get('Positive', 0)),
                            "Negative Count": int(sentiment_counts_for_history.get('Negative', 0)),
                            "Neutral Count": int(sentiment_counts_for_history.get('Neutral', 0))
                        }
                    })


        elif uploaded_file.type == "text/csv":
            df_uploaded = pd.read_csv(uploaded_file)
            st.subheader("Preview of Uploaded CSV Data:")
            st.dataframe(df_uploaded.head(), use_container_width=True)

            text_column = st.selectbox(
                "Select the column containing the text to analyze:",
                df_uploaded.columns,
                help="Choose the column in your CSV that holds the text content for sentiment analysis."
            )

            if text_column and st.button("Process CSV File", key="process_csv_btn", use_container_width=True):
                texts_to_process = df_uploaded[text_column].astype(str).tolist()
                st.info(f"Processing **{len(texts_to_process)}** entries from the CSV file using column '**{text_column}**'.")
                
                st.toast("Starting batch analysis...", icon="🚀")
                results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, text in enumerate(texts_to_process):
                    status_text.text(f"Processing text {i+1} of {len(texts_to_process)}...")
                    sentiment_res = get_sentiment(text)
                    keywords_res = extract_keywords(text)
                    results_list.append({
                        "Original Text": text,
                        "Sentiment": sentiment_res['sentiment'],
                        "Confidence": sentiment_res['confidence'],
                        "Keywords": ", ".join(keywords_res)
                    })
                    progress_bar.progress((i + 1) / len(texts_to_process))
                
                df_results = pd.DataFrame(results_list)
                st.session_state['df_results'] = df_results # Store in session state
                status_text.success("Batch processing complete!")
                st.toast("Batch analysis finished!", icon="✅")

                # Add to history
                current_batch_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sentiment_counts_for_history = df_results['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
                st.session_state['analysis_history'].insert(0, {
                    "Type": "Batch File Analysis",
                    "Timestamp": current_batch_time,
                    "Details": {
                        "File Name": uploaded_file.name,
                        "Total Texts Processed": len(df_results),
                        "Positive Count": int(sentiment_counts_for_history.get('Positive', 0)),
                        "Negative Count": int(sentiment_counts_for_history.get('Negative', 0)),
                        "Neutral Count": int(sentiment_counts_for_history.get('Neutral', 0))
                    }
                })

        if 'df_results' in st.session_state and not st.session_state['df_results'].empty:
            df_results = st.session_state['df_results'] # Retrieve from session state
            st.subheader("Batch Analysis Results")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(df_results, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- Visualization Components for Batch Processing ---
            st.subheader("Sentiment Distribution Overview")
            st.markdown("Visualizing the overall breakdown of positive, neutral, and negative sentiments in your dataset.")
            sentiment_counts = df_results['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}) # Green, Amber, Red
            ax.set_title('Overall Sentiment Distribution', fontsize=16)
            ax.set_xlabel('Sentiment Category', fontsize=12)
            ax.set_ylabel('Number of Texts', fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("Export Your Results")
            st.markdown("Download the analyzed data for further use or reporting.")
            col1, col2, col3, col4 = st.columns(4) # Added col4 for PDF

            # CSV Export
            csv_file = df_results.to_csv(index=False).encode('utf-8')
            col1.download_button(
                label="Download as CSV",
                data=csv_file,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
                key="download_csv_batch",
                use_container_width=True
            )

            # JSON Export
            json_file = df_results.to_json(orient="records", indent=4).encode('utf-8')
            col2.download_button(
                label="Download as JSON",
                data=json_file,
                file_name="sentiment_analysis_results.json",
                mime="application/json",
                key="download_json_batch",
                use_container_width=True
            )
            # TXT Export for Batch (basic CSV-like text)
            txt_file = df_results.to_string(index=False).encode('utf-8')
            col3.download_button(
                label="Download as TXT",
                data=txt_file,
                file_name="sentiment_analysis_results.txt",
                mime="text/plain",
                key="download_txt_batch",
                use_container_width=True
            )

            # PDF Export for Batch
            if uploaded_file: # Ensure file was uploaded for the name
                batch_pdf_output = create_batch_analysis_pdf(df_results, uploaded_file.name, sentiment_counts)
                col4.download_button(
                    label="Download as PDF",
                    data=batch_pdf_output,
                    file_name="batch_sentiment_analysis_results.pdf",
                    mime="application/pdf",
                    key="download_batch_pdf",
                    use_container_width=True
                )
            
        elif uploaded_file is None:
             st.info("Upload a .txt or .csv file above to begin batch analysis.")


with tab3: # Accuracy Report Tab
    st.header("Evaluate Model Accuracy: Gemini API vs. Manual Labels")
    st.markdown("""
        This section helps you assess the performance of the sentiment analysis model.
        It compares the sentiments predicted by the **Gemini API** against a **manually labeled "ground truth" dataset**.
        This is crucial for understanding how well the model performs on real-world data and identifying areas for improvement.

        **To get meaningful results, ensure your sample data includes diverse examples for each sentiment category.**
        For a robust evaluation, we recommend using **at least 50 sample texts** that you have carefully analyzed and labeled.
    """)

    # Sample data for demonstration (replace with your 50+ manually analyzed texts)
    # Added more diverse examples to better illustrate a real dataset
    sample_texts_with_labels = [
        {"text": "The service was absolutely excellent and I am very satisfied with the prompt response.", "true_sentiment": "Positive"},
        {"text": "This product broke after only a week, truly disappointed and frustrated with the quality.", "true_sentiment": "Negative"},
        {"text": "The delivery arrived on time, no issues with the packaging or contents.", "true_sentiment": "Neutral"},
        {"text": "I really enjoyed the movie, especially the acting and the fantastic storyline.", "true_sentiment": "Positive"},
        {"text": "The support agent was unhelpful and incredibly rude, making the situation worse.", "true_sentiment": "Negative"},
        {"text": "It's an average book, nothing spectacular but not bad either. Just okay.", "true_sentiment": "Neutral"},
        {"text": "Fantastic experience from start to finish! Highly recommend this place.", "true_sentiment": "Positive"},
        {"text": "The software is buggy and constantly crashes, making it impossible to work.", "true_sentiment": "Negative"},
        {"text": "The meeting lasted an hour and covered all the agenda items as planned.", "true_sentiment": "Neutral"},
        {"text": "This coffee tastes amazing, a real treat to start the day!", "true_sentiment": "Positive"},
        {"text": "My order was incorrect and the return process was a nightmare, very frustrating.", "true_sentiment": "Negative"},
        {"text": "The weather today is cloudy with a chance of light rain later this afternoon.", "true_sentiment": "Neutral"},
        {"text": "Absolutely loved the new features! They are a game-changer.", "true_sentiment": "Positive"},
        {"text": "The price is too high for such poor quality, not worth the investment.", "true_sentiment": "Negative"},
        {"text": "The document contains essential information on company policies and procedures.", "true_sentiment": "Neutral"},
        {"text": "Couldn't be happier with the results! Everything worked perfectly.", "true_sentiment": "Positive"},
        {"text": "Such a terrible experience, I will never use this service again.", "true_sentiment": "Negative"},
        {"text": "The report provides a summary of the quarterly financial performance.", "true_sentiment": "Neutral"},
        {"text": "A truly wonderful discovery, I'm so glad I tried it.", "true_sentiment": "Positive"},
        {"text": "Completely unsatisfied, the product failed expectations.", "true_sentiment": "Negative"},
        {"text": "The tutorial explained the basic concepts clearly.", "true_sentiment": "Neutral"},
        {"text": "I had a pleasant time, the staff were very friendly.", "true_sentiment": "Positive"},
        {"text": "This is the worst customer service I've ever encountered.", "true_sentiment": "Negative"},
        {"text": "The data collected will be used for future analysis.", "true_sentiment": "Neutral"},
        {"text": "An excellent choice, very robust and reliable.", "true_sentiment": "Positive"},
        {"text": "The constant glitches make this app unusable.", "true_sentiment": "Negative"},
        {"text": "The presentation covered all key areas of the project.", "true_sentiment": "Neutral"},
    ]

    st.subheader("Upload Your Own Labeled Data (TXT or CSV)")
    st.markdown("""
        Upload a `.txt` file where each line contains a text followed by its true sentiment label, separated by a delimiter (e.g., `Text ###Sentiment`).
        Or, upload a `.csv` file with columns for 'text' and 'true_sentiment'.
        """)

    uploaded_accuracy_file = st.file_uploader(
        "Choose a file for accuracy report:",
        type=["txt", "csv"],
        key="upload_accuracy_file",
        help="For .txt: Each line should be 'Your text here###True Sentiment'. For .csv: must have 'text' and 'true_sentiment' columns."
    )

    data_to_process = None # Initialize data_to_process
    
    col_upload_btn_txt, col_upload_btn_csv = st.columns(2)

    if uploaded_accuracy_file is not None:
        st.info(f"Using uploaded file: **{uploaded_accuracy_file.name}**")
        if uploaded_accuracy_file.type == "text/plain":
            stringio_accuracy = StringIO(uploaded_accuracy_file.getvalue().decode("utf-8"))
            lines = stringio_accuracy.readlines()
            parsed_data = []
            for line in lines:
                parts = line.strip().split("###", 1) # Split only on the first "###"
                if len(parts) == 2:
                    parsed_data.append({"text": parts[0].strip(), "true_sentiment": parts[1].strip()})
                elif len(parts) == 1 and parts[0].strip():
                    st.warning(f"Line '{line.strip()}' in TXT file missing '###' separator or sentiment label. Skipping.")
            if not parsed_data:
                st.error("Uploaded TXT file could not be parsed. Please ensure each line is 'Your text here###True Sentiment'.")
            else:
                if col_upload_btn_txt.button("Generate Accuracy Report from Uploaded TXT", key="generate_accuracy_report_uploaded_txt_btn", use_container_width=True):
                    data_to_process = parsed_data
        elif uploaded_accuracy_file.type == "text/csv":
            df_accuracy_uploaded = pd.read_csv(uploaded_accuracy_file)
            if 'text' in df_accuracy_uploaded.columns and 'true_sentiment' in df_accuracy_uploaded.columns:
                st.subheader("Preview of Uploaded Accuracy CSV Data:")
                st.dataframe(df_accuracy_uploaded.head(), use_container_width=True)
                if col_upload_btn_csv.button("Generate Accuracy Report from Uploaded CSV", key="generate_accuracy_report_uploaded_csv_btn", use_container_width=True):
                    data_to_process = df_accuracy_uploaded[['text', 'true_sentiment']].to_dict(orient='records')
            else:
                st.error("Uploaded CSV must contain 'text' and 'true_sentiment' columns for accuracy evaluation.")
    
    st.markdown("---")
    st.subheader("Or use Sample Data for Demonstration")
    if st.button("Generate Accuracy Report for Sample Data", key="generate_accuracy_report_sample_btn", use_container_width=True):
        data_to_process = sample_texts_with_labels
        st.info("Using sample data.") # Indicate which data is being used

    # Process data if any of the "Generate" buttons were clicked and data_to_process is set
    if data_to_process and (
        st.session_state.get("generate_accuracy_report_uploaded_txt_btn", False) or
        st.session_state.get("generate_accuracy_report_uploaded_csv_btn", False) or
        st.session_state.get("generate_accuracy_report_sample_btn", False)
    ):
        st.info("Running Gemini API on texts and comparing to your manual labels. This may take a moment...")
        true_labels = []
        predicted_labels = []
        comparison_data = []

        progress_text = st.empty()
        progress_bar = st.progress(0)

        for i, item in enumerate(data_to_process):
            text = item['text']
            true_sentiment = item['true_sentiment']

            gemini_result = get_sentiment(text)
            predicted_sentiment = gemini_result['sentiment']

            true_labels.append(true_sentiment)
            predicted_labels.append(predicted_sentiment)
            comparison_data.append({
                "Text": text,
                "Manual Sentiment": true_sentiment,
                "Gemini Predicted Sentiment": predicted_sentiment,
                "Confidence": f"{gemini_result['confidence']:.2f}",
                "Match": "✅ Yes" if true_sentiment == predicted_sentiment else "❌ No"
            })

            progress_bar.progress((i + 1) / len(data_to_process))
            progress_text.text(f"Processing item {i+1} of {len(data_to_process)}...")
        
        st.session_state['accuracy_data'] = {
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'comparison_data': comparison_data
        }
        
        # Add to history (only if it hasn't been added already in this session/run)
        current_time_accuracy = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_exists = any(
            entry.get('Type') == "Accuracy Report" and entry.get('Timestamp') == current_time_accuracy
            for entry in st.session_state['analysis_history']
        )
        
        if not history_exists:
            st.session_state['analysis_history'].insert(0, {
                "Type": "Accuracy Report",
                "Timestamp": current_time_accuracy,
                "Details": {
                    "Overall Accuracy": f"{accuracy_score(true_labels, predicted_labels):.2%}", # Calculate accuracy here
                    "Processed Samples": len(true_labels)
                }
            })
            st.toast("Accuracy Report added to history!", icon="⏳")

        st.toast("Accuracy report generated!", icon="📊")
        progress_text.empty() # Clear the processing text

    if 'accuracy_data' in st.session_state:
        true_labels = st.session_state['accuracy_data']['true_labels']
        predicted_labels = st.session_state['accuracy_data']['predicted_labels']
        comparison_data = st.session_state['accuracy_data']['comparison_data']

        if true_labels: # Ensure there's data to avoid errors
            accuracy = accuracy_score(true_labels, predicted_labels)
            st.subheader(f"Overall Model Accuracy: **{accuracy:.2%}**")

            st.markdown("---")
            st.subheader("Detailed Comparison: Manual Labels vs. Gemini API Predictions")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("💡 **Match** indicates if the Gemini API's prediction aligns with your manual 'true sentiment' label.")

            # Classification Report
            st.markdown("---")
            st.subheader("Classification Report")
            st.markdown("Provides precision, recall, and F1-score for each sentiment class.")
            report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().round(2)

            # Apply custom styles to the Classification Report DataFrame
            def highlight_accuracy(s):
                if s.name == 'f1-score':
                    return ['background-color: #e6f7ff' if val >= 0.8 else '' for val in s]
                return [''] * len(s)

            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(report_df.style.apply(highlight_accuracy, axis=0), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.info("🔍 **Precision:** Proportion of positive identifications that were actually correct. **Recall:** Proportion of actual positives that were identified correctly. **F1-Score:** Harmonic mean of precision and recall.")

            # Confusion Matrix
            st.markdown("---")
            st.subheader("Confusion Matrix")
            st.markdown("Visualizes the performance of the classification model, showing actual vs. predicted labels.")
            
            # Get unique labels from both true and predicted, then sort them consistently
            labels = sorted(list(set(true_labels + predicted_labels)))
            
            cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f'True {l}' for l in labels], columns=[f'Predicted {l}' for l in labels])

            fig_cm, ax_cm = plt.subplots(figsize=(9, 7))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm, linewidths=.5, linecolor='lightgray')
            ax_cm.set_title('Confusion Matrix: True vs. Predicted Sentiments', fontsize=16)
            ax_cm.set_xlabel('Predicted Label', fontsize=12)
            ax_cm.set_ylabel('True Label', fontsize=12)
            st.pyplot(fig_cm)
            st.info("🔢 **Interpretation:** Rows represent true labels, columns represent predicted labels. Diagonal values show correct predictions. Off-diagonal values show misclassifications (e.g., a text that was truly 'Positive' but predicted as 'Negative').")

            st.markdown("---")
            st.subheader("Download Accuracy Report")
            accuracy_pdf_output = create_accuracy_report_pdf(st.session_state['accuracy_data'], accuracy, report_df, cm_df)
            st.download_button(
                label="Download Accuracy Report as PDF",
                data=accuracy_pdf_output,
                file_name="sentiment_accuracy_report.pdf",
                mime="application/pdf",
                key="download_accuracy_pdf",
                use_container_width=True
            )
        else:
            st.warning("No data processed for accuracy report. Please upload a file or use sample data.")


with tab4: # History Tab
    st.header("Analysis History")
    st.markdown("Review your past sentiment analysis sessions.")

    if st.session_state['analysis_history']:
        for i, entry in enumerate(st.session_state['analysis_history']):
            with st.expander(f"**{entry['Type']}** - {entry['Timestamp']}"):
                for key, value in entry['Details'].items():
                    st.markdown(f"**{key}:** {value}")
                
                # Option to clear specific history entry
                if st.button(f"Clear this entry", key=f"clear_history_{i}"):
                    st.session_state['analysis_history'].pop(i)
                    if 'df_results' in st.session_state: # Clear batch results if related
                        del st.session_state['df_results']
                    if 'accuracy_data' in st.session_state: # Clear accuracy data if related
                        del st.session_state['accuracy_data']
                    st.rerun() # Rerun to update the display

        st.markdown("---")
        if st.button("Clear All History", key="clear_all_history_btn", type="secondary"):
            st.session_state['analysis_history'] = []
            if 'df_results' in st.session_state:
                del st.session_state['df_results']
            if 'accuracy_data' in st.session_state:
                del st.session_state['accuracy_data']
            st.rerun()
    else:
        st.info("No analysis history yet. Perform some analyses to see them here!")

with tab5: # Model Limitations Tab
    st.header("Understanding Model Limitations")
    st.markdown("""
        While powerful, sentiment analysis models, including those based on Gemini, have inherent limitations:

        * **Sarcasm and Irony:** Models often struggle to detect sarcasm, irony, or subtle humor, misinterpreting the true sentiment.
        * **Contextual Nuance:** Without broader context, a model might misunderstand industry-specific jargon, cultural references, or highly nuanced language.
        * **Domain Specificity:** A model trained on general text might perform poorly on highly specialized text (e.g., medical reports, legal documents) where words have different connotations.
        * **Double Negatives and Complex Structures:** Sentences with multiple negatives ("not unwelcome") or convoluted structures can confuse models.
        * **Mixed Sentiments:** A single piece of text can contain both positive and negative aspects. Models might assign an overall neutral or dominant sentiment, missing the full spectrum.
        * **New Slang and Evolving Language:** Language is constantly changing. Models require continuous updates to accurately interpret new slang, idioms, or evolving sentiment expressions.
        * **Subjectivity of "Neutral":** What constitutes "neutral" can be subjective. Some texts are genuinely neutral, while others might be deemed neutral due to a balance of positive and negative words.
        * **Emoji and Emoticon Interpretation:** While many models are improving, interpreting the full range of emojis and their context can still be a challenge.
        * **Language Specificity:** Performance can vary significantly across different languages, depending on the training data availability and linguistic complexities.

        **Best Practices:**
        To mitigate these limitations, consider:
        * **Human Oversight:** Always use automated sentiment analysis as a tool, not a definitive answer. Human review of critical or ambiguous cases is essential.
        * **Domain-Specific Training (if possible):** For highly specialized applications, fine-tuning the model with domain-specific labeled data can significantly improve accuracy.
        * **Clear Data Labeling:** When providing ground truth for accuracy reports, ensure your manual labels are consistent and reflect the intended sentiment accurately.
    """)

# --- Documentation/Information Sidebar ---
st.sidebar.title("🚀 Project Information")
st.sidebar.markdown(
    """
    This interactive dashboard is developed as part of the
    **Demand Program 3 by CAPACITI**.
    It demonstrates the practical application of advanced
    Natural Language Processing (NLP) for sentiment analysis.
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("✨ Key Features:")
st.sidebar.markdown(
    """
    -   **Flexible Input:** Analyze single texts or upload files (.txt, .csv).
    -   **Multi-Class Sentiment:** Classifies text as Positive, Negative, or Neutral.
    -   **Confidence Scoring:** Provides a score indicating model certainty.
    -   **Keyword Extraction:** Pinpoints key terms driving sentiment.
    -   **Batch Processing:** Efficiently analyze large datasets.
    -   **Intuitive Visualizations:** Understand sentiment distribution at a glance.
    -   **Data Export:** Download results in convenient CSV, JSON, and TXT formats.
    -   **Accuracy Reporting:** Evaluate model performance against ground truth.
    -   **Analysis History:** Keep track of past analysis sessions.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Technical Details:")
st.sidebar.info(
    """
    **Core NLP Engine:** Google Gemini API (`gemini-1.5-flash` for description, `gemini-2.0-flash` for mock API calls in this demo).
    **Frontend Framework:** Streamlit.
    **Styling:** Custom HTML/CSS for enhanced user experience.
    **Libraries:** `pandas` for data handling, `matplotlib` & `seaborn` for visualizations, `scikit-learn` for metrics.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("📞 Contact Us:")
st.sidebar.write("selekeamogetsoe@gmail.com")
st.sidebar.write("mvubum26@gmail.com")
st.sidebar.write("dlakavusiseko@gmail.com")
st.sidebar.write("cynthiamotaung015@gmail.com")
st.sidebar.markdown("---")
st.sidebar.write("[Visit CAPACITI](https://www.capaciti.org.za)") 