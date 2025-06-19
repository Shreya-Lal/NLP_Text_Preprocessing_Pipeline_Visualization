import gradio as gr
import re
# import shutil
# import os
import string
import unicodedata
import difflib
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# download the necessary resources if haven't done so
nltk.download('omw-1.4')

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
# nltk.download('wordnet')
nltk.download('stopwords')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to expand contractions
def expand_contractions(text):
    contractions = {
        "can't": "cannot", "won't": "will not", "it's": "it is", "I'm": "I am",
        "you're": "you are", "they're": "they are", "he's": "he is", "she's": "she is",
        "we're": "we are", "that's": "that is", "what's": "what is", "where's": "where is",
        "who's": "who is", "how's": "how is", "let's": "let us", "I've": "I have",
        "you've": "you have", "we've": "we have", "they've": "they have", "I'd": "I would",
        "you'd": "you would", "he'd": "he would", "she'd": "she would", "we'd": "we would",
        "they'd": "they would", "I'll": "I will", "you'll": "you will", "he'll": "he will",
        "she'll": "she will", "we'll": "we will", "they'll": "they will", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
        "haven't": "have not", "hadn't": "had not", "doesn't": "does not", "don't": "do not",
        "didn't": "did not", "shouldn't": "should not", "wouldn't": "would not",
        "couldn't": "could not", "mustn't": "must not"
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text

# Function to remove accents
def remove_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# Function to normalize text
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Function to generate HTML with color-coded differences
def highlight_differences(original, processed):
    if original == processed:
        return f'<span style="color:black">{processed}</span>'
    
    diff = difflib.ndiff(original.split(), processed.split())
    html_output = []
    
    for word in diff:
        if word.startswith('  '):  # Unchanged
            html_output.append(f'<span style="color:black">{word[2:]}</span>')
        elif word.startswith('- '):  # Removed
            html_output.append(f'<span style="color:red; text-decoration:line-through">{word[2:]}</span>')
        elif word.startswith('+ '):  # Added
            html_output.append(f'<span style="color:green; font-weight:bold">{word[2:]}</span>')
    
    return ' '.join(html_output)

# Function to tokenize and highlight stopwords
def tokenize_and_highlight(text, remove_stopwords=False):
    tokens = word_tokenize(text)
    
    if not remove_stopwords:
        return tokens, " ".join(tokens)
    
    # Highlight stopwords in red
    highlighted_tokens = []
    for token in tokens:
        if token in stop_words:
            highlighted_tokens.append(f'<span style="color:red; font-weight:bold">{token}</span>')
        else:
            highlighted_tokens.append(token)
    
    return tokens, " ".join(highlighted_tokens)

# Function to apply stemming with highlighting
def apply_stemming(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    highlighted = []
    for original, stemmed in zip(tokens, stemmed_tokens):
        if original != stemmed:
            highlighted.append(f'<span style="color:black">{original}</span> → '
                              f'<span style="color:purple; font-weight:bold">{stemmed}</span>')
        else:
            highlighted.append(f'<span style="color:black">{original}</span>')
    
    return stemmed_tokens, " ".join(highlighted)

# Function to apply lemmatization with highlighting
def apply_lemmatization(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    highlighted = []
    for original, lemma in zip(tokens, lemmatized_tokens):
        if original != lemma:
            highlighted.append(f'<span style="color:black">{original}</span> → '
                              f'<span style="color:blue; font-weight:bold">{lemma}</span>')
        else:
            highlighted.append(f'<span style="color:black">{original}</span>')
    
    return lemmatized_tokens, " ".join(highlighted)

# Main processing function
def process_text(text, normalization, contraction, accents, stopword_removal, stemming, lemmatization):
    if not text:
        return "Please enter some text to process."
    
    steps = []
    results_html = []
    current_text = text
    
    # Display original text
    results_html.append(f"<h3>Original Text:</h3>"
                       f"<div style='background-color:#f8f8f8; padding:10px; border-radius:5px;'>"
                       f"{text}</div>")
    
    # Step 1: Expand Contractions
    if contraction:
        expanded_text = expand_contractions(current_text)
        highlighted = highlight_differences(current_text, expanded_text)
        results_html.append(f"<h3>1. After Contraction Expansion:</h3>"
                           f"<div style='background-color:#f0f8ff; padding:10px; border-radius:5px;'>"
                           f"{highlighted}</div>")
        current_text = expanded_text
        steps.append("contraction")
    
    # Step 2: Remove Accents
    if accents:
        accent_removed = remove_accents(current_text)
        highlighted = highlight_differences(current_text, accent_removed)
        results_html.append(f"<h3>2. After Accent Removal:</h3>"
                           f"<div style='background-color:#f0fff0; padding:10px; border-radius:5px;'>"
                           f"{highlighted}</div>")
        current_text = accent_removed
        steps.append("accents")
    
    # Step 3: Normalization
    if normalization:
        normalized = normalize_text(current_text)
        highlighted = highlight_differences(current_text, normalized)
        results_html.append(f"<h3>3. After Normalization:</h3>"
                           f"<div style='background-color:#fff8f0; padding:10px; border-radius:5px;'>"
                           f"{highlighted}</div>")
        current_text = normalized
        steps.append("normalization")
    
    # Tokenization
    tokens, tokens_html = tokenize_and_highlight(current_text, stopword_removal)
    results_html.append(f"<h3>4. Tokenization:</h3>"
                       f"<div style='background-color:#f8f0ff; padding:10px; border-radius:5px;'>"
                       f"{tokens_html}</div>")
    
    # Stopword Removal
    if stopword_removal:
        # Get tokens without stopwords
        filtered_tokens = [token for token in tokens if token not in stop_words]
        _, tokens_html = tokenize_and_highlight(" ".join(tokens), True)
        results_html.append(f"<h3>5. After Stopword Removal:</h3>"
                           f"<div style='background-color:#fff0f5; padding:10px; border-radius:5px;'>"
                           f"{tokens_html}</div>")
        tokens = filtered_tokens
        steps.append("stopword_removal")
    
    # Stemming
    if stemming:
        stemmed_tokens, highlighted = apply_stemming(tokens)
        results_html.append(f"<h3>6. After Stemming:</h3>"
                           f"<div style='background-color:#f0f8ff; padding:10px; border-radius:5px;'>"
                           f"{highlighted}</div>")
        tokens = stemmed_tokens
        steps.append("stemming")
    
    # Lemmatization
    if lemmatization:
        lemmatized_tokens, highlighted = apply_lemmatization(tokens)
        results_html.append(f"<h3>7. After Lemmatization:</h3>"
                           f"<div style='background-color:#f0fff0; padding:10px; border-radius:5px;'>"
                           f"{highlighted}</div>")
        tokens = lemmatized_tokens
        steps.append("lemmatization")
    
    # Final tokens
    results_html.append(f"<h3>Final Tokens:</h3>"
                       f"<div style='background-color:#e6e6fa; padding:10px; border-radius:5px;'>"
                       f"{(tokens)}</div>")
    
    return "<br>".join(results_html)

# Create Gradio interface
with gr.Blocks(title="Text Preprocessing App", theme="soft", css=".highlight { background-color: yellow; }") as app:
    gr.Markdown(
        "<h1 style='text-align: center;'>Text Preprocessing Pipeline with Visual Transformation</h1>"
    )
    gr.Markdown("""
    ### See how each step transforms your text with color-coded changes:
    <div style="margin-bottom:20px">
        <span style="color:green; font-weight:bold">Green</span> = Added content, 
        <span style="color:red; text-decoration:line-through">Red</span> = Removed content, 
        <span style="color:purple; font-weight:bold">Purple</span> = Stemmed words, 
        <span style="color:blue; font-weight:bold">Blue</span> = Lemmatized words
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Text", lines=5, placeholder="Enter your text here...")
            with gr.Accordion("Preprocessing Options", open=True):
                contraction = gr.Checkbox(label="1. Expand Contractions", value=True)
                accents = gr.Checkbox(label="2. Remove Accents/Dialects")
                normalization = gr.Checkbox(label="3. Text Normalization", value=True)
                stopword_removal = gr.Checkbox(label="4. Stopword Removal", value=True)
                stemming = gr.Checkbox(label="5. Stemming")
                lemmatization = gr.Checkbox(label="6. Lemmatization")
            submit_btn = gr.Button("Process Text", variant="primary")
        
        with gr.Column():
            output_html = gr.HTML(label="Processing Results")
    
    submit_btn.click(
        fn=process_text,
        inputs=[text_input, normalization, contraction, accents, stopword_removal, stemming, lemmatization],
        outputs=output_html
    )
    
    examples = gr.Examples(
        examples=[
            ["I'm studying NLP since 2020 - it's fascinating! Can't wait to apply it."],
            ["The café's spécialités are delicious, but they don't serve them after 9 PM."],
            ["We've been working on this project for months, and we're making good progress!"],
            ["She doesn't know that her résumé needs updating for the job application process."]
        ],
        inputs=[text_input],
        label="Try these examples (contains contractions, accents, punctuation)"
    )
    
    # Add a collapsible references section
    with gr.Accordion("References & Resources", open=False):
        gr.Markdown("""
        ## Text Preprocessing References:
        
        - [NLTK Documentation](https://www.nltk.org/) - Natural Language Toolkit
        - [Stemming vs Lemmatization](https://www.baeldung.com/cs/stemming-vs-lemmatization) - Comparison article
        - [Regular Expressions Guide](https://docs.python.org/3/howto/regex.html) - Python regex documentation
        - [Unicode Normalization](https://unicode.org/reports/tr15/) - About accent removal
        - [Contraction Handling in NLP](https://towardsdatascience.com/why-and-how-to-handle-contractions-in-nlp-28ca3a2c5d4a) - Article on contractions
        
        ## Academic Papers:
        
        - [Text Normalization Techniques](https://aclanthology.org/W17-3202.pdf) - ACL Anthology paper
        - [Preprocessing for NLP Tasks](https://arxiv.org/abs/2005.03684) - Survey of text preprocessing
        """)

app.launch()