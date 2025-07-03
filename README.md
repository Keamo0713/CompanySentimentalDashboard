Link to this dashboard: [https://companysentimentaldashboard.onrender.com ]

Link to the deployed dashboard that does not work with API: [https://companysentimentaldashboardmock.onrender.com] 

Link to the mock dashboards github : [https://github.com/Keamo0713/CompanySentimentalDashboardMock]

A Streamlit-based web application for comprehensive sentiment analysis and keyword extraction from text data, powered by the Google Gemini API. This dashboard allows users to analyze single pieces of text or upload datasets for batch processing, providing insights into sentiment distribution and key drivers.

## ✨ Features

* **Multi-Class Sentiment Analysis:** Classifies text into Positive, Negative, or Neutral sentiment categories.
* **Confidence Scoring:** Provides a numerical confidence score for each sentiment classification.
* **Keyword Extraction:** Identifies and extracts keywords or phrases that are most influential in determining the sentiment of the text.
* **Batch Processing:** Upload `.txt` or `.csv` files for efficient analysis of multiple entries.
* **Intuitive Visualizations:** Presents sentiment distribution through interactive charts (e.g., pie charts, bar charts).
* **Data Export:** Download analysis results in various formats (CSV, JSON, PDF reports).
* **Analysis History:** Keeps a record of past analysis sessions for quick review.
* **Error Handling:** Robust error reporting for API communication issues and quota limits.

## 🚀 Technologies Used

* **Frontend Framework:** Streamlit
* **Core NLP Engine:** Google Gemini API (`gemini-2.0-flash` or `gemini-1.5-flash` for analysis)
* **Data Handling:** `pandas`
* **Visualizations:** `matplotlib`, `seaborn`
* **Utility:** `fpdf` for PDF generation, `scikit-learn` for classification metrics.
* **Version Control:** Git / GitHub

## 🛠️ Setup and Local Installation

Follow these steps to set up and run the application on your local machine.

### Prerequisites

* Python 3.9+
* Git

### 1. Clone the Repository

Bash
git clone [https://github.com/Keamo0713/CompanySentimentalDashboard.git](https://github.com/Keamo0713/CompanySentimentalDashboard.git)
cd CompanySentimentalDashboard
2. Create a Virtual Environment (Recommended)
Bash

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
Install all required Python packages using pip:

Bash

pip install -r requirements.txt
Your requirements.txt should contain (at minimum):

streamlit
pandas
matplotlib
seaborn
scikit-learn
numpy
fpdf
google-generativeai
requests # If still used
4. Configure Google Gemini API Key
This application requires a Google Gemini API key to function.

Go to the Google AI Studio to generate an API key.

Create a .streamlit folder in your project's root directory if it doesn't exist.

Inside the .streamlit folder, create a file named secrets.toml.

Add your API key to secrets.toml in the following format:

Ini, TOML

GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
Replace YOUR_ACTUAL_GEMINI_API_KEY_HERE with your generated key.

Note: For local development, you can also set GOOGLE_API_KEY as an environment variable in your system.

5. Run the Application
Once all dependencies are installed and the API key is configured, run the Streamlit app:

Bash

streamlit run app.py
This command will open the application in your default web browser.

🌍 Deployment
This application can be deployed on various cloud platforms.

Common Requirements for Deployment
All code (app.py, sentiment_analyzer.py, etc.) must be pushed to your GitHub repository.

A requirements.txt file listing all Python dependencies.

Your GOOGLE_API_KEY must be securely configured as a secret or environment variable on the chosen deployment platform, NOT hardcoded in your public repository.

Streamlit Community Cloud
Platform: share.streamlit.io

Setup: Connect your GitHub repository, specify the branch (e.g., master), and app.py as the main file.

Secrets: Add GOOGLE_API_KEY="YOUR_KEY" in the app's settings under "Secrets".

Hugging Face Spaces
Platform: huggingface.co/spaces/new

SDK: If "Streamlit" SDK is not directly available, choose "Docker".

If using Docker, include a Dockerfile in your repository root:

Dockerfile

FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
Secrets: Add GOOGLE_API_KEY as a "Repository secret" in your Space's settings.

Render
Platform: render.com

Setup: Create a new "Web Service", connect your GitHub repository.

Procfile: Create a Procfile in your repository root:

web: streamlit run app.py --server.port=$PORT --server.enableCORS=false --server.enableXsrfProtection=false
Environment Variables: Add GOOGLE_API_KEY as an environment variable in your Render service settings.

⚠️ Troubleshooting Common Deployment Issues
DNS_PROBE_FINISHED_NXDOMAIN: This usually means the app failed to start or crashed immediately. Check deployment logs.

429 Quota Exceeded (Google Gemini API): You have hit your usage limits for the Gemini API.

Solution: Check your Google Cloud Project's "APIs & Services" > "Quotas" to monitor usage or upgrade your plan. Waiting a day often resolves daily quota limits.

AttributeError related to e.response.text: This indicates an error in the API error handling. Ensure you have the latest sentiment_analyzer.py code with robust try-except blocks.

Deployment Logs are your best friend! Always review the logs on your chosen deployment platform (Streamlit Cloud, Hugging Face Spaces, or Render) for detailed error messages.

📞 Contact
If you have any questions or need further assistance, feel free to reach out:

selekeamogetsoe@gmail.com

mvubum26@gmail.com

dlakavusiseko@gmail.com

![image](https://github.com/user-attachments/assets/fb4aad58-4370-42fd-bf15-7670089a2ff3)
![image](https://github.com/user-attachments/assets/83462bba-55b8-4103-9306-91dd1978b011)
![image](https://github.com/user-attachments/assets/5ab62903-b681-4fd4-8630-24ff111453fc)


