import markdown
import pdfkit
import os

MARKDOWN_PATH = r"C:\Users\KIIT0001\.gemini\antigravity\brain\312b7f27-98de-4bb4-8979-6edbd236bade\walkthrough.md"
PDF_PATH = r"d:\AntiGravity\zomathon\Zomaton_PS2_Submission.pdf"

if not os.path.exists(MARKDOWN_PATH):
    print("Error: Markdown walkthrough not found.")
    exit(1)

with open(MARKDOWN_PATH, 'r', encoding='utf-8') as f:
    text = f.read()
    
# Convert MM to HTML
html_text = markdown.markdown(text, extensions=['extra', 'tables'])

# Add some basic styling
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; color: #333; }}
        h1, h2, h3 {{ color: #E23744; }} /* Zomato Red */
        h1 {{ border-bottom: 2px solid #E23744; padding-bottom: 10px; }}
        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 4px; font-family: monospace; }}
        ul {{ margin-top: 5px; }}
        li {{ margin-bottom: 5px; }}
    </style>
</head>
<body>
    {html_text}
</body>
</html>
"""

print(f"Converting markdown to PDF...")
try:
    # Requires wkhtmltopdf installed on the system. If not, we will just save the HTML
    pdfkit.from_string(html_content, PDF_PATH)
    print(f"SUCCESS: Created {PDF_PATH}")
except Exception as e:
    print(f"PDF creation failed (likely missing wkhtmltopdf): {e}")
    # Fallback to HTML
    HTML_PATH = r"d:\AntiGravity\zomathon\Zomaton_PS2_Submission.html"
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Fallback: Saved as HTML instead to {HTML_PATH}")
