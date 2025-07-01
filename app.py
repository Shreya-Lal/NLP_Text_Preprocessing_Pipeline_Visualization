import gradio as gr
import re
import string
import unicodedata
import difflib
# We are using Snowball stemmer from the NLTK package to implement this. 
# This is a rectified version of Porter’s stemmer algorithm.The Snowball stemmer 
# can stem texts in a number of other Roman script languages, such as Dutch, German, French, and even Russian.
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import contractions

# Download required NLTK data
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer and lemmatizer
#stemmer = PorterStemmer()
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to expand contractions
def expand_contractions(text):
    expanded_text = []   
    for word in text.split():
        expanded_text.append(contractions.fix(word))  
    expanded_text = ' '.join(expanded_text)
    return expanded_text

# Function to remove accents
def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

# Text cleaning functions
def convert_to_lowercase(text):
    return text.lower()

def remove_urls(text):
    # Pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_non_word_non_whitespace(text):
    # Remove any character that's not a word character (alphanumeric + underscore) or whitespace
    return re.sub(r'[^\w\s]', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_digits(text):
    return re.sub(r'\d+', '', text)

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def replace_repeated_punctuation(text):
    # Replace repeated punctuation with single instance
    return re.sub(r'([!?.])\1+', r'\1', text)

def remove_twitter_handlers(text):
    # Remove @mentions and RT tags
    return re.sub(r'\bRT\b|@\w+', '', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

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

# Function to generate token IDs
def generate_token_ids(tokens):
    # Create a vocabulary of unique tokens
    vocab = sorted(set(tokens))
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    # Map each token to its ID
    token_ids = [token_to_id[token] for token in tokens]
    return token_ids, token_to_id

# Main processing function
def process_text(text, text_cleaning, lowercase, expand_contractions_opt, remove_accents_opt, 
                 remove_urls_opt, remove_non_word, remove_punct, remove_digits_opt, 
                 remove_whitespace, replace_punct_repetition, remove_twitter_handlers_opt, remove_emoji_opt,
                 stopword_removal, stemming, lemmatization):
    if not text:
        return "Please enter some text to process."
    
    #step_index = 1
    results_html = []
    current_text = text
    
    # Display original text
    results_html.append(f"<h3>Original Text:</h3>"
                       f"<div style='background-color:#f8f8f8; padding:10px; border-radius:5px;'>"
                       f"{text}</div>")
    
    # Text Cleaning (Step 1)
    if text_cleaning:
        results_html.append(f"<h2>Text Cleaning:</h2>")
        
        # Convert to lowercase
        if lowercase:
            new_text = convert_to_lowercase(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Convert to Lowercase:</h3>"
                               f"<div style='background-color:#e0f7fa; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Expand contractions
        if expand_contractions_opt:
            new_text = expand_contractions(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Expanding Contractions:</h3>"
                               f"<div style='background-color:#f0f8ff; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Remove accents
        if remove_accents_opt:
            new_text = remove_accents(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing Accents:</h3>"
                               f"<div style='background-color:#f0fff0; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Remove URLs
        if remove_urls_opt:
            new_text = remove_urls(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing URLs:</h3>"
                               f"<div style='background-color:#fff3e0; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Remove non-word/non-whitespace characters
        if remove_non_word:
            new_text = remove_non_word_non_whitespace(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing Non-Word/Non-Whitespace Characters:</h3>"
                               f"<div style='background-color:#e8f5e9; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Remove punctuation
        if remove_punct:
            new_text = remove_punctuation(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing Punctuation:</h3>"
                               f"<div style='background-color:#fff8e1; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Remove digits
        if remove_digits_opt:
            new_text = remove_digits(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing Digits:</h3>"
                               f"<div style='background-color:#fce4ec; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Remove extra whitespace
        if remove_whitespace:
            new_text = remove_extra_whitespace(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing Extra Whitespace:</h3>"
                               f"<div style='background-color:#f3e5f5; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Replace repeated punctuation characters by keeping just one of the same punctuation character 
        if replace_punct_repetition:
            new_text = replace_repeated_punctuation(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Replacing Repeated Punctuation(Keeping one punctuation):</h3>"
                               f"<div style='background-color:#e3f2fd; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        # Remove twitter handlers
        if remove_twitter_handlers_opt:
            new_text = remove_twitter_handlers(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing Twitter Handlers and RT tags:</h3>"
                               f"<div style='background-color:#ffebee; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text

        # Remove emoji
        if remove_emoji_opt:
            new_text = remove_emoji(current_text)
            highlighted = highlight_differences(current_text, new_text)
            results_html.append(f"<h3> After Removing Emojis:</h3>"
                               f"<div style='background-color:#f0fff0; padding:10px; border-radius:5px;'>"
                               f"{highlighted}</div>")
            current_text = new_text
        
        #step_index += 1
    
    # Step 2: Tokenization
    tokens, tokens_html = tokenize_and_highlight(current_text, stopword_removal)
    results_html.append(f"<h3> Tokenization Input:</h3>"
                       f"<div style='background-color:#f8f0ff; padding:10px; border-radius:5px;'>"
                       f"{tokens_html}</div>")
    
    # Generate token IDs
    token_ids, token_id_map = generate_token_ids(tokens)
    token_id_html = ", ".join([f"<span style='color:#6a0dad'>{token}</span>: <span style='color:#008080'>{tid}</span>" 
                              for token, tid in token_id_map.items()])
    token_id_list = ", ".join(map(str, token_ids))
    
    results_html.append(f"<h3> Token IDs:</h3>"
                       f"<div style='background-color:#e6e6fa; padding:10px; border-radius:5px;'>"
                       f"<b>Token → ID Mapping:</b> {token_id_html}<br><br>"
                       f"<b>Token ID Sequence:</b> [{token_id_list}]</div>")
    #step_index += 1
    
    # Step 3: Stopword Removal
    if stopword_removal:
        # Get tokens without stopwords
        filtered_tokens = [token for token in tokens if token not in stop_words]
        _, tokens_html = tokenize_and_highlight(" ".join(tokens), True)
        results_html.append(f"<h3> After Stopword Removal:</h3>"
                           f"<div style='background-color:#fff0f5; padding:10px; border-radius:5px;'>"
                           f"{tokens_html}</div>")
        tokens = filtered_tokens
        #step_index += 1
    
    # Step 4: Stemming
    if stemming:
        stemmed_tokens, highlighted = apply_stemming(tokens)
        results_html.append(f"<h3> After Stemming:</h3>"
                           f"<div style='background-color:#f0f8ff; padding:10px; border-radius:5px;'>"
                           f"{highlighted}</div>")
        tokens = stemmed_tokens
        #step_index += 1
    
    # Step 5: Lemmatization
    if lemmatization:
        lemmatized_tokens, highlighted = apply_lemmatization(tokens)
        results_html.append(f"<h3> After Lemmatization:</h3>"
                           f"<div style='background-color:#f0fff0; padding:10px; border-radius:5px;'>"
                           f"{highlighted}</div>")
        tokens = lemmatized_tokens
        #step_index += 1
    
    # Final tokens with IDs
    final_token_ids, _ = generate_token_ids(tokens)
    final_id_list = ", ".join(map(str, final_token_ids))
    
    results_html.append(f"<h3>Final Tokens:</h3>"
                       f"<div style='background-color:#e6e6fa; padding:10px; border-radius:5px;'>"
                       f"{tokens}</div>"
                       f"<h3>Final Token IDs:</h3>"
                       f"<div style='background-color:#e6e6fa; padding:10px; border-radius:5px;'>"
                       f"[{final_id_list}]</div>")
    
    return "<br>".join(results_html)

custom_css = """
.gr-accordion-header {
    background-color: #f89d8a;
    color: #0c63e4; 
}
.cleaning-option {
    background-color: #f0f7ff;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
"""

# Create Gradio interface
with gr.Blocks(title="Advanced Text Preprocessing", theme=gr.themes.Default(primary_hue="red", secondary_hue="pink"), css=custom_css) as app:
    gr.Markdown(
        "<h1 style='text-align: center;'>🧠 NLP Text Preprocessing Playground</h1>"
    )
    gr.Markdown("""
                <div style="text-align: justify">
                    <span style="color:gray; font-weight:bold">Welcome!</span> 
                    This app visually demonstrates essential preprocessing steps in NLP—like tokenization, stemming, and lemmatization— <b>before</b> diving into code.By using this visualization tool, you'll develop an intuitive understanding of how each 
                    preprocessing step affects your text data, helping you make better 
                    decisions when preparing text for your NLP applications. 
                    </div>
                """)
    with gr.Accordion("Explanation and Usage", open=False):
        gr.Markdown("""
---
                      
## 🧹 Complete Text Cleaning Options:
1. **Lowercase Conversion**: Standardizes text to lowercase
2. **Contraction Expansion**: Converts shortened forms to full forms (e.g., "can't" → "cannot")
3. **Accent Removal**: Normalizes special characters (e.g., "café" → "cafe")
4. **URL Removal**: Eliminates web addresses
5. **Non-Word/Whitespace Removal**: Removes special characters such as hashtags(#)
6. **Punctuation Removal**: Strips punctuation marks
7. **Digit Removal**: Eliminates numerical values
8. **Whitespace Normalization**: Reduces multiple spaces to single spaces
9. **Repeated Punctuation Handling**: Replaces multiple punctuation with single same punctuation
10. **Twitter Handler and RT tags Removal**: Removes @mentions and retweet tags RT
11. **Remove Emojis**: Removes Removes emoticons, symbols & pictographs, transport & map symbols, flags etc.   
        
## What is Text Pre-processing?
Text preprocessing is the essential first step in Natural Language Processing (NLP) that transforms 
raw text into a clean, standardized format suitable for analysis.
                    
## Why Comprehensive Text Cleaning?
Text cleaning is crucial for preparing raw text for NLP tasks. Different text sources (social media, documents, web content) 
contain various types of noise that need to be addressed before analysis.

## Processing Workflow
1. **Text Cleaning**: Optional step with various sub-operations
2. **Tokenization**: Splitting text into individual tokens/words
3. **Stopword Removal**: Filtering out common but unimportant words (e.g., "the", "and")
4. **Stemming**: Reducing words to their root form (e.g., "running" → "run")
5. **Lemmatization**: More sophisticated word normalization, uses dictionary forms (e.g., "better" → "good")
6. **Token ID Generation**: Creating numerical representations
                    
---

## Why is Preprocessing Important?
- Standardization: Different sources format text differently (capitalization, punctuation, etc.)
- Noise Reduction: Removes irrelevant elements like special characters and stopwords
- Dimensionality Reduction: Simplifies text while preserving meaning
- Improved Accuracy: Helps models focus on meaningful patterns rather than artifacts
- Consistency: Ensures all text follows the same format for fair comparison

## How to Use This Visualization Tool
#### Understanding the Color Coding 
Our app uses a visual system to help you understand each transformation:

🟢 Green text: New content added by the transformation <br>

🔴 Red strikethrough: Content removed by the transformation <br>

🟣 Purple: Stemmed versions of words (reduced to root form) <br>

🔵 Blue: Lemmatized words (dictionary form) <br>

⚫ Black: Unchanged content <br>
    """)
        
    gr.Markdown("""
    > This is where the journey to building LLMs begins! 🚀
    """)

    gr.Markdown("""
    ### Visual Guide to Transformations:
    <div style="margin-bottom:20px">
        <span style="color:green; font-weight:bold">Green</span> = Added content, 
        <span style="color:red; text-decoration:line-through">Red</span> = Removed content, 
        <span style="color:purple; font-weight:bold">Purple</span> = Stemmed words, 
        <span style="color:blue; font-weight:bold">Blue</span> = Lemmatized words
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Text", lines=5, placeholder="Enter text with URLs, numbers, @mentions, and special characters...")
            
            with gr.Accordion("Text Cleaning Options (Step 1)", open=True) as cleaning_accordion:
                text_cleaning = gr.Checkbox(label="Enable Text Cleaning", value=True)
                with gr.Group(visible=True) as cleaning_group:
                    lowercase = gr.Checkbox(label=" Convert to lowercase", value=True)
                    expand_contractions_opt = gr.Checkbox(label=" Expand contractions", value=True)
                    remove_accents_opt = gr.Checkbox(label=" Remove accents/dialects")
                    remove_urls_opt = gr.Checkbox(label=" Remove URLs")
                    remove_non_word = gr.Checkbox(label=" Remove non-word/non-whitespace characters")
                    remove_punct = gr.Checkbox(label=" Remove punctuation")
                    remove_digits_opt = gr.Checkbox(label=" Remove digits")
                    remove_whitespace = gr.Checkbox(label=" Remove extra whitespace")
                    replace_punct_repetition = gr.Checkbox(label=" Replace repeated punctuation")
                    remove_twitter_handlers_opt = gr.Checkbox(label=" Remove Twitter handlers and RT tags")
                    remove_emoji_opt = gr.Checkbox(label=" Remove Emojis")
            
            with gr.Accordion("Advanced NLP Processing", open=True):
                stopword_removal = gr.Checkbox(label=" Stopword Removal", value=True)
                stemming = gr.Checkbox(label=" Stemming")
                lemmatization = gr.Checkbox(label=" Lemmatization")
                
            submit_btn = gr.Button("Process Text", variant="primary")
        
        with gr.Column():
            output_html = gr.HTML(label="Processing Results")
    
    # Show/hide cleaning options based on main checkbox
    text_cleaning.change(
        fn=lambda x: gr.Group(visible=x),
        inputs=text_cleaning,
        outputs=cleaning_group
    )
    
    submit_btn.click(
        fn=process_text,
        inputs=[
            text_input, 
            text_cleaning, lowercase, expand_contractions_opt, remove_accents_opt, 
            remove_urls_opt, remove_non_word, remove_punct, remove_digits_opt, 
            remove_whitespace, replace_punct_repetition, remove_twitter_handlers_opt, remove_emoji_opt,
            stopword_removal, stemming, lemmatization
        ],
        outputs=output_html
    )
    
    examples = gr.Examples(
        examples=[
            ["Check out https://nlp.stanford.edu!!! I've been studying NLP since 2020. @StanfordNLP rocks!"],
            ["The café's spécialités cost $25.99 each - don't miss out!!! 😊 #delicious @foodie"],
            ["Meeting at 5:30 PM.   Please   be   on time!!! We're excited @team!!!"],
            ["Her résumé shows 5+ years of Python. Contact: email@example.com @dev"]
        ],
        inputs=[text_input],
        label="Try these examples (contains URLs, digits, @mentions, repeated punctuation)"
    )
    
    # Add a collapsible references section
    with gr.Accordion("References & Resources", open=False):
        gr.Markdown("""
        ## Text Preprocessing References:
        
        - [Text Cleaning Techniques in NLP](https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/)
        - [Twitter Text Processing](https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis/)
        - [Stemming vs Lemmatization](https://www.baeldung.com/cs/stemming-vs-lemmatization)
        - [Contraction Handling in NLP](https://how.dev/answers/how-to-deal-with-contractions-in-nlp)
        - [List of Emojis Unicode](https://www.unicode.org/Public/emoji/1.0//emoji-data.txt) --> use this link to get the unicode of every emojis
                    
        """)

app.launch(share=True)