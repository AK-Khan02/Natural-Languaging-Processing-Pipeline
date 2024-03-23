import spacy
from transformers import pipeline as hf_pipeline
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer

# Load spaCy's transformer-based model for better accuracy in tasks
nlp = spacy.load("en_core_web_trf")

# Initialize Hugging Face's pipelines
sentiment_pipeline = hf_pipeline("sentiment-analysis")
summarization_pipeline = hf_pipeline("summarization", model="t5-small", tokenizer="t5-small")

class AdvancedNLPPipeline:
    def __init__(self, text):
        self.text = text
        self.doc = nlp(text)
        self.parser = PlaintextParser.from_string(text, Tokenizer("english"))

    def advanced_sentiment_analysis(self):
        """Advanced sentiment analysis using Transformers."""
        return sentiment_pipeline(self.text)

    def topic_modeling(self, num_topics=5, passes=10):
        """Perform LDA Topic Modeling."""
        processed_docs = [[token.lemma_ for token in sent if token.is_alpha and not token.is_stop] for sent in self.doc.sents]
        dictionary = Dictionary(processed_docs)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        
        lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
        return {f"Topic {i+1}": [word for word, _ in topics[i][1]] for i in range(num_topics)}

    def text_summarization(self):
        """Text summarization using Sumy's LSA summarizer."""
        summarizer = Summarizer()
        summary = summarizer(self.parser.document, sentences_count=3)
        return " ".join([str(sentence) for sentence in summary])

    def relation_extraction(self):
        """Placeholder for Relation Extraction - assumes integration of a trained RE model."""
        # Example: Extracting 'SUBJ' and 'OBJ' relations and the verb connecting them
        relations = []
        for token in self.doc:
            if "subj" in token.dep_:
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if "obj" in child.dep_:
                        relations.append((subject, verb, child.text))
        return relations

    def summarize(self):
        """Summarizes all NLP tasks."""
        return {
            'sentiment_analysis': self.advanced_sentiment_analysis(),
            'topic_modeling': self.topic_modeling(),
            'text_summarization': self.text_summarization(),
            'relation_extraction': self.relation_extraction(),
        }

# Example Usage
text = """Tesla Inc. is an American electric vehicle and clean energy company based in Palo Alto, California. Tesla's current products include electric cars, battery energy storage from home to grid-scale, solar panels and solar roof tiles, as well as other related products and services."""
pipeline = AdvancedNLPPipeline(text)
summary = pipeline.summarize()

for key, value in summary.items():
    print(f"{key.capitalize()}: {value}\n")
