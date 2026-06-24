import nltk
import string
import tkinter as tk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {

"What is Python?":
"Python is a high-level programming language used for AI, web development and automation.",

"What is machine learning?":
"Machine learning is a branch of AI where computers learn from data.",

"What is artificial intelligence?":
"Artificial Intelligence allows machines to perform tasks that require human intelligence.",

"What is NLP?":
"NLP stands for Natural Language Processing. It helps computers understand human language.",

"What is VS Code?":
"VS Code is a source code editor developed by Microsoft.",

"What is a chatbot?":
"A chatbot is a program that communicates with users through text or voice.",

"What is a database?":
"A database is an organized collection of data.",

"What is cybersecurity?":
"Cybersecurity protects computers and data from cyber threats."

}
def preprocess(text):

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)
questions = list(faqs.keys())

processed_questions = [
    preprocess(q) for q in questions
]
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    processed_questions
)
def chatbot(user_input):

    user_input = preprocess(user_input)

    user_vector = vectorizer.transform(
        [user_input]
    )

    similarity = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match = similarity.argmax()

    score = similarity[0][best_match]


    if score < 0.2:
        return "Sorry, I don't understand your question."

    return faqs[questions[best_match]]
# GUI Chat Window

window = tk.Tk()
window.title("FAQ Chatbot")
window.geometry("500x500")


# Chat display area

chat_box = tk.Text(
    window,
    height=20,
    width=55
)

chat_box.pack(pady=10)


# User input box

entry = tk.Entry(
    window,
    width=50
)

entry.pack(pady=5)



def send_message():

    user_message = entry.get()

    if user_message.strip() == "":
        return

    chat_box.insert(
        tk.END,
        "You: " + user_message + "\n"
    )

    response = chatbot(user_message)

    chat_box.insert(
        tk.END,
        "Bot: " + response + "\n\n"
    )

    entry.delete(0, tk.END)



# Send button

button = tk.Button(
    window,
    text="Send",
    command=send_message
)

button.pack()


window.mainloop()