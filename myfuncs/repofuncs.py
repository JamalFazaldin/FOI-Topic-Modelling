from tqdm import tqdm
import re

from concurrent.futures import ThreadPoolExecutor

from queue import Queue
results_queue = Queue()

import nltk
nltk_data_path = os.getenv('', 'default/path')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import spacy
from spellchecker import SpellChecker
from wordsegment import load, segment
from wordsegment import Segmenter
from collections import Counter

# Initialize external tools
spell = SpellChecker()
custom_segmenter = Segmenter()
custom_segmenter.load()  # Load data for wordsegment
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

import gensim
from gensim.models import CoherenceModel

def get_pos_tags(text):
    doc = nlp(text)
    return {token.text: token.tag_ for token in doc}

def is_valid_correction(original, pos_tags):
    # Skip correction if the original word matches these patterns
    if re.match(r"\d+(st|nd|rd|th)", original):  # Ordinal numbers
        return False
    if pos_tags and pos_tags.get(original) == 'NNP' or pos_tags.get(original) == 'PROPN':
        return False
    return True

def is_valid_token(word):
    if len(word) > 2:
        return True
    if any(char.isdigit() for char in word):
        return False
    return False

# Tokenization function
def safe_tokenize(text):
    if isinstance(text, str):
        return word_tokenize(text)
    return []

def segementing_and_spelling(tokens):
    # Step 3: Spelling correction and word segmentation
    corrected_tokens = []
    for word in tokens:
        split_words = custom_segmenter.segment(word)  # Split merged words
        for subword in split_words:
            corrected_word = spell.correction(subword) or word # Correct spelling
            pos_tags = get_pos_tags(subword)
            if not is_valid_correction(subword, pos_tags):  # Skip invalid corrections
                corrected_word = subword
            corrected_tokens.append(corrected_word)
        
    filtered_tokens = [word for word in corrected_tokens if is_valid_token(word)]

    
    return filtered_tokens

def process_text(row,details):
    text = row[f'{details}']
    if not isinstance(text, str):  # Convert non-string types to string
        text = str(text)
    if not text.strip():  # Skip empty or whitespace strings
        return ""
 
    try:
        # Step 1: Text cleaning
        text = text.lower()  # Lowercase
        text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b", "", text)  # Remove email addresses
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
 
        # Step 2: Tokenization
        tokens = text.split()
        
        # Step 3 & 4: Word segmenting and spelling
        corrected_tokens = segementing_and_spelling(tokens=tokens)
 
        # Step 5: Lemmatization and stop word removal
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in corrected_tokens if word not in stop_words]

        lemmatized_tokens = [word for word in lemmatized_tokens if word not in stop_words]

        # Step 6: Word segmenting and spelling (again)
        final_tokens = segementing_and_spelling(tokens=lemmatized_tokens)

        final_tokens = [word for word in final_tokens if word not in stop_words]

        # Step 7: Join processed tokens into a string
        return " ".join(final_tokens)
 
    except Exception as e:
        print(f"Error processing text: {repr(text)}")
        raise e
 
 
def process_text_with_progress(df,details):
    # Initialize tqdm progress bar
    pbar = tqdm.tqdm(total=len(df), desc="Processing rows", unit="row", dynamic_ncols=True)
    def process_row(row):
        try:
            return process_text(row,details=details)
        finally:
            # Update the progress bar for every row processed
            pbar.update(1)
    
    df['processed_details'] = df.apply(process_row, axis=1)
    df['processed_tokens'] = df['processed_details'].apply(safe_tokenize)

    pbar.close()
    
    return df

def prepare_corpus(tokens, threshold=100, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # Generate bigrams and trigrams
    bigram = gensim.models.Phrases(tokens, min_count=5, threshold=threshold)
    trigram = gensim.models.Phrases(bigram[tokens], threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    bigram_texts = (bigram_mod[doc] for doc in tokens)
    trigram_texts = (trigram_mod[bigram_mod[doc]] for doc in bigram_texts)

    # Verify POS of each word
    texts_out = []
    for text in trigram_texts:
        # Use SpaCy to process the text
        doc = nlp(" ".join(text))  # Join tokens into a string and process with SpaCy
        filtered_tokens = [token.text for token in doc if token.pos_ in allowed_postags]
        texts_out.append(filtered_tokens)
    return texts_out

def count_words(tokens):
    # Flatten the tokenized details
    all_tokens = [token for sublist in tokens for token in sublist]
    # Count word frequencies
    word_counts = Counter(all_tokens)
    # Print the top 10 most common words
    word_counts = pd.DataFrame([word_counts])
    word_counts.reset_index(inplace=True)
    word_counts = word_counts.melt(id_vars='index',var_name='Word',value_name='Count')
    word_counts.sort_values(by='Count',ascending=False,inplace=True)
    word_counts.reset_index(inplace=True,drop=['index'])
    word_counts.drop(columns='index',inplace=True)
    display(word_counts.head(25))
    display(word_counts)

# Model Evaluation Function
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        random_state=42,
        chunksize=100,
        passes=10,
        alpha=a,
        eta=b,
        workers=4
    )
    coherence_model_lda = CoherenceModel(model=lda_model, texts=FOI_corpus, dictionary=id2words, coherence='c_v')

    return coherence_model_lda.get_coherence()

# Define the function for computing coherence for parameter sets
def compute_for_params(corpus_set, k, a, b, corpus_title):
    try:
        # Thread-local coherence value computation
        cv = compute_coherence_values(corpus=corpus_sets[corpus_set], dictionary=id2words, k=k, a=a, b=b)
        # The result is created with thread-local variables
        result = {
            'Validation_Set': corpus_title,
            'Topics': k,
            'Alpha': a,
            'Beta': b,
            'Coherence': cv
        }
        results_queue.put(result)
    except Exception as e:
        print(f"Error processing (k={k}, a={a}, b={b}, corpus_title={corpus_title}): {str(e)}")

def run_io_task_in_parallel(tasks):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in futures:
            future.result()

def find_element(corpus_title):
    if corpus_title == '75% Corpus':
        return 0
    return 1

# Function to extract the domain part
def extract_domain(email):
    try:
        email = email.lower()
        match = re.search(r'@([\w.-]+)', email)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        print(f"Error extracting domain from email: {email}")
        raise e