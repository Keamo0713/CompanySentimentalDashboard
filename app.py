import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

# Import our sentiment analysis and keyword extraction functions
from sentiment_analyzer import get_sentiment, extract_keywords

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide", # Use wide layout for better visualization space
    initial_sidebar_state="expanded"
)

# --- Title and Description ---
st.title("😊 Sentiment Analysis Dashboard")
st.markdown(
    """
    Analyze the emotional tone of your text data. Understand customer reviews,
    social media posts, or any other text content by classifying sentiment
    as positive, negative, or neutral.
    """
)

st.divider() # Adds a nice visual separator

# --- Input Section ---
st.header("1. Input Your Text")

input_option = st.radio(
    "Choose input method:",
    ("Direct Text Entry", "Upload Text File (.txt, .csv)"),
    key="input_method"
)

text_input = ""
uploaded_file = None

if input_option == "Direct Text Entry":
    text_input = st.text_area(
        "Enter text here for sentiment analysis:",
        "This is a fantastic product! I am so happy with my purchase. The quality is great.",
        height=200,
        help="Type or paste the text you want to analyze."
    )
    if st.button("Analyze Sentiment", key="analyze_single_text"):
        if text_input:
            # --- Perform analysis for single text ---
            st.subheader("Analysis Results:")
            sentiment_result = get_sentiment(text_input)
            st.write(f"**Overall Sentiment:** {sentiment_result['sentiment']}")
            st.write(f"**Confidence:** {sentiment_result['confidence']:.2f}")

            keywords = extract_keywords(text_input)
            if keywords:
                st.write("**Identified Keywords (Sentiment Drivers):**")
                # Highlight keywords in the original text
                highlighted_text = text_input
                for keyword in keywords:
                    highlighted_text = highlighted_text.replace(keyword, f"**{keyword}**")
                st.markdown(f"Original Text with Keywords Highlighted: {highlighted_text}")
            else:
                st.info("No significant keywords extracted for this text.")

            # --- Explanation Feature (basic) ---
            st.markdown("---")
            st.subheader("Why this sentiment?")
            if sentiment_result['sentiment'] == 'Positive':
                st.info("The model likely detected positive words/phrases and entities.")
            elif sentiment_result['sentiment'] == 'Negative':
                st.warning("The model likely detected negative words/phrases and strong entities.")
            elif sentiment_result['sentiment'] == 'Neutral':
                st.info("The model found no strong positive or negative indicators, or it found a balance of both.")
            st.write(f"The confidence score of {sentiment_result['confidence']:.2f} indicates how certain the model is about its classification.")


        else:
            st.warning("Please enter some text to analyze.")

elif input_option == "Upload Text File (.txt, .csv)":
    uploaded_file = st.file_uploader(
        "Upload a .txt or .csv file:",
        type=["txt", "csv"],
        help="For CSV, ensure one column contains the text to be analyzed."
    )

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        df_results = pd.DataFrame()

        if uploaded_file.type == "text/plain":
            # Read plain text file line by line or as a single block
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            texts = [line.strip() for line in stringio.readlines() if line.strip()]
            st.info(f"Processing {len(texts)} lines from the text file.")

            if st.button("Process Batch Text File", key="process_txt"):
                results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, text in enumerate(texts):
                    sentiment_res = get_sentiment(text)
                    keywords_res = extract_keywords(text)
                    results_list.append({
                        "Original Text": text,
                        "Sentiment": sentiment_res['sentiment'],
                        "Confidence": sentiment_res['confidence'],
                        "Keywords": ", ".join(keywords_res)
                    })
                    progress_bar.progress((i + 1) / len(texts))
                    status_text.text(f"Processing text {i+1}/{len(texts)}")
                df_results = pd.DataFrame(results_list)
                st.success("Batch processing complete!")

        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            text_column = st.selectbox(
                "Select the column containing text for analysis:",
                df.columns,
                help="Choose the column in your CSV that holds the text content."
            )

            if text_column and st.button("Process Batch CSV File", key="process_csv"):
                texts = df[text_column].astype(str).tolist()
                st.info(f"Processing {len(texts)} entries from the CSV file.")

                results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, text in enumerate(texts):
                    sentiment_res = get_sentiment(text)
                    keywords_res = extract_keywords(text)
                    results_list.append({
                        "Original Text": text,
                        "Sentiment": sentiment_res['sentiment'],
                        "Confidence": sentiment_res['confidence'],
                        "Keywords": ", ".join(keywords_res)
                    })
                    progress_bar.progress((i + 1) / len(texts))
                    status_text.text(f"Processing text {i+1}/{len(texts)}")
                df_results = pd.DataFrame(results_list)
                st.success("Batch processing complete!")

        if not df_results.empty:
            st.subheader("Batch Analysis Results Table:")
            st.dataframe(df_results)

            # --- Visualization Components for Batch Processing ---
            st.subheader("2. Sentiment Distribution Overview")
            sentiment_counts = df_results['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette=['green', 'grey', 'red'])
            ax.set_title('Overall Sentiment Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Number of Texts')
            st.pyplot(fig)

            # --- Comparative Analysis (Basic) ---
            st.subheader("3. Comparative Analysis (Batch)")
            st.info("You can sort the table above by 'Confidence' to see high-confidence classifications, or filter by 'Sentiment' for specific categories.")
            # More advanced comparative analysis could involve plotting confidence distributions per sentiment,
            # or allowing selection of specific texts for side-by-side comparison.

            # --- Export Results ---
            st.subheader("4. Export Results")
            col1, col2, col3 = st.columns(3)

            # CSV Export
            csv_file = df_results.to_csv(index=False).encode('utf-8')
            col1.download_button(
                label="Download as CSV",
                data=csv_file,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
                key="download_csv"
            )

            # JSON Export
            json_file = df_results.to_json(orient="records", indent=4).encode('utf-8')
            col2.download_button(
                label="Download as JSON",
                data=json_file,
                file_name="sentiment_analysis_results.json",
                mime="application/json",
                key="download_json"
            )

            # PDF Export (Placeholder - requires more complex implementation)
            # You would need a library like reportlab for this.
            # For now, it's a dummy button.
            # Example for PDF generation:
            # from reportlab.lib.pagesizes import letter
            # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            # from reportlab.lib.styles import getSampleStyleSheet
            # def generate_pdf(df):
            #    doc = SimpleDocTemplate("sentiment_report.pdf", pagesize=letter)
            #    styles = getSampleStyleSheet()
            #    elements = [Paragraph("Sentiment Analysis Report", styles['h1'])]
            #    elements.append(Spacer(1, 12))
            #    # Add df.to_string() or a more structured table
            #    elements.append(Paragraph(df.to_string(), styles['Normal']))
            #    doc.build(elements)
            #
            # if col3.button("Download as PDF", key="download_pdf"):
            #    generate_pdf(df_results)
            #    st.success("PDF report generated (check your project folder).")
            # This is a more complex implementation for direct download in Streamlit, often you'd
            # create the file and then provide a link or specific Streamlit download button for it.
            col3.info("PDF Export requires additional library setup (e.g., ReportLab). Not implemented in this basic example.")


# --- Documentation/Information Sidebar ---
st.sidebar.title("Project Information")
st.sidebar.markdown(
    """
    This dashboard is part of the **Tech Career Accelerator** program by CAPACITI.

    **Key Features:**
    - Direct text entry or file upload
    - Multi-class sentiment (Positive, Negative, Neutral)
    - Confidence scoring
    - Keyword extraction
    - Batch processing for files
    - Sentiment distribution visualization
    - Export results (CSV, JSON)
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Technical Details:")
st.sidebar.info(
    """
    **NLP API Integration:** Hugging Face `transformers` library (running a local model).
    **Web Interface:** Streamlit
    **Keyword Extraction:** SpaCy
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Developed by Your Name/Team Name")
st.sidebar.write("Contact: hello@capaciti.org.za")
st.sidebar.write("www.capaciti.org.za")