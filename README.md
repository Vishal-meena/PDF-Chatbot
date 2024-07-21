# Chat With PDF

## Description
A chatbot that interacts with users by answering questions based on the content of a provided PDF.



https://github.com/user-attachments/assets/3bbc8871-4f3d-4fed-90d0-fdbb1e5fa3e7



## Features
- Upload a PDF file
- Preprocess the PDF to create a vector store
- Ask questions about the PDF content

## Requirements
- Python 3.7+
- Streamlit
- LangChain
- OpenAI API Key
- dotenv

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Vishal-meena/PDF-Chatbot.git
    cd PDF-Chatbot
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your OpenAI API key in the `.env` file:
        ```env
        OPENAI_API_KEY=your_openai_api_key
        ```

## Running the App
1. Start the Streamlit app:
    ```bash
    streamlit run app.py --server.enableXsrfProtection false
    ```

2. Open your browser and go to `http://localhost:8501`.

## Usage
1. Upload a PDF file using the sidebar.
2. Click on the "Preprocess" button to process the PDF.
3. Type your questions in the text input at the bottom.

## Troubleshooting
- If you encounter any issues, ensure all dependencies are installed and your OpenAI API key is correct.
