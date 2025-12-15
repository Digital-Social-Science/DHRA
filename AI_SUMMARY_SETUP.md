# AI Summary Feature Setup Guide

## Overview
The AI Summary feature has been integrated into the expert's profile. Experts can now click an "AI Summary" button next to each article to get an AI-generated summary of the PDF content.

## What Has Been Implemented

### Backend (app.py)
1. **Added Dependencies:**
   - `pdfminer.six` for PDF text extraction
   - `python-dotenv` for environment variable management
   - `logging` for better debugging

2. **New Classes & Functions:**
   - `CerebrasLLM` class: Handles communication with Cerebras AI API
   - `extract_text_with_pdfminer()`: Extracts text from PDF files
   - `summarize_pdf_with_cerebras()`: Generates AI summaries from PDF text

3. **New Endpoint:**
   - `GET /get_ai_summary/<post_id>`: Fetches post details, reads the PDF, and generates AI summary

### Frontend (templates/expert.html)
1. **New Modal:** Added "AI Summary Modal" to display summaries
2. **Updated Article Lists:** Each article now has an "AI Summary" button
3. **Enhanced JavaScript:** `showAISummary()` function fetches and displays summaries with loading states

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```bash
# .env
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

**To get a Cerebras API key:**
1. Visit https://cloud.cerebras.ai/
2. Sign up or log in
3. Navigate to API Keys section
4. Generate a new API key
5. Copy it to your `.env` file

### 3. Restart Your Flask Application
```bash
python app.py
```

## How It Works

1. **User clicks "AI Summary" button** on any article in the expert's profile
2. **Frontend sends request** to `/get_ai_summary/<post_id>`
3. **Backend:**
   - Queries database for post details (email, title)
   - Locates PDF file: `researchers/<email>/<title>.pdf`
   - Extracts text from PDF using pdfminer.six
   - Sends text to Cerebras LLM for summarization
   - Returns summary (approximately 150 words in English)
4. **Frontend displays** the summary in a modal popup

## Features

- **Multi-language Support:** PDFs in any language are summarized in English
- **Loading States:** Shows spinner while generating summary
- **Error Handling:** Displays user-friendly error messages
- **Professional Summaries:** Captures main points and key details
- **Non-blocking:** Article list remains clickable for full details

## File Structure
```
researchers/
  └── <email>/
      └── <article_title>.pdf  ← PDF files are read from here
```

## Troubleshooting

### "AI service not configured" error
- Ensure `CEREBRAS_API_KEY` is set in `.env` file
- Run `source .env` or restart your application

### "PDF file not found" error
- Verify the PDF exists at: `researchers/<email>/<title>.pdf`
- Check that the title matches exactly (including special characters)

### "Failed to extract text" error
- PDF might be image-based (scanned) - requires OCR
- PDF might be corrupted or password-protected

### Slow response
- Normal for large PDFs (can take 10-60 seconds)
- Cerebras API might be experiencing high load

## API Response Format

**Success Response:**
```json
{
  "success": true,
  "summary": "The AI-generated summary text...",
  "title": "Article Title"
}
```

**Error Response:**
```json
{
  "error": "Error message description"
}
```

## Notes

- Summaries are generated on-demand (not cached)
- Each request makes a call to Cerebras API
- Summary length is approximately 150 words
- Works with documents in any language (output is always English)

