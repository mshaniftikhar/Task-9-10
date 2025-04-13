from flask import Flask, render_template, request, jsonify
import nltk
from nltk.chat.util import Chat, reflections
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

# Chatbot pairs
pairs = [
    [r"(?i).*hello.*|.*hi.*|.*hey.*",
     ["Hello! How can I assist you with admissions?",
      "Hi there! Do you need admission information?",
      "Hey! Welcome to Superior University Admissions, how can I help?"]],

    [r"(?i).*programs.*offer.*|.*degree programs.*|.*courses available.*",
     ["Superior University offers undergraduate, graduate, and PhD programs in Engineering, Business, IT, Social Sciences, and more.",
      "We offer a variety of programs, including BS, MS, and PhD in multiple fields. You can visit our website for a complete list.",
      "Our university provides degrees in disciplines such as Computer Science, Business Administration, Media Studies, and Engineering."]],

    [r"(?i).*admission requirements.*|.*entry requirements.*|.*eligibility.*",
     ["Admission requirements vary by program. Generally, you need academic transcripts, an entry test (if applicable), and supporting documents.",
      "You must have the required qualifications based on your desired program. Some programs also require an entry test.",
      "Each program has specific requirements. Visit our website or contact the admission office for details."]],

    [r"(?i).*apply.*admission.*|.*admission process.*|.*steps to apply.*",
     ["You can apply online through the official Superior University website or visit the admission office for assistance.",
      "The admission process is simple: fill out the online application form, submit required documents, and appear for an entry test (if required).",
      "To apply, visit our website, choose your program, and follow the instructions to complete the application process."]],

    [r"(?i).*last date.*apply.*|.*admission deadline.*|.*when.*admission.*close.*",
     ["Admission deadlines vary each semester. Please check the official website or contact the admission office for updated information.",
      "The last date to apply depends on the academic session. Keep an eye on our website for important dates.",
      "Superior University announces deadlines for each intake separately. Visit our admissions page for the latest updates."]],

    [r"(?i).*entry test.*|.*admission test.*|.*exam required.*",
     ["Yes, some programs require an entry test, while others may consider merit-based admissions.",
      "Entry tests are mandatory for specific programs. Check the eligibility criteria for your selected course.",
      "Depending on the program, you may need to appear for an entry test or submit prior qualifications."]],

    [r"(?i).*scholarship.*|.*financial aid.*|.*tuition support.*",
     ["Superior University offers merit-based and need-based scholarships to deserving students.",
      "We provide financial aid options, including scholarships for high achievers and students in need.",
      "Our scholarships include merit-based, need-based, sports, and extracurricular excellence awards."]],

    [r"(?i).*fee structure.*|.*tuition fees.*|.*cost of studying.*",
     ["Tuition fees vary depending on the program. You can check the official website for the detailed fee structure.",
      "Our fee structure is designed to be affordable, with installment options available. Contact the finance department for details.",
      "The tuition cost depends on the program you choose. For exact figures, visit our website or the admissions office."]],

    [r"(?i).*university location.*|.*campus location.*|.*where is the university.*",
     ["Superior University has multiple campuses. Our main campus is located in Lahore, and we have other campuses across Pakistan.",
      "You can visit our official website to find the campus nearest to you.",
      "The main campus is in Lahore, and we also have regional campuses in different cities."]],

    [r"(?i).*hostel.*|.*accommodation.*|.*on-campus living.*",
     ["Yes, we provide hostel facilities for both male and female students with all necessary amenities.",
      "Our university offers comfortable and secure hostel accommodations. Contact the administration for details.",
      "Hostel rooms are available on a first-come, first-served basis. Check with the hostel management for availability."]],

    [r"(?i).*international students.*|.*foreign students.*|.*apply from abroad.*",
     ["Yes, international students are welcome to apply. They must meet the eligibility criteria and provide necessary documentation.",
      "Foreign students can apply online, and we provide visa assistance if needed.",
      "We accept applications from international students. Contact the international admissions office for details."]],

    [r"(?i).*bye.*|.*goodbye.*|.*see you later.*",
     ["Goodbye! Feel free to ask anytime.",
      "See you later! Good luck with your admission.",
      "Alright, take care! Let us know if you have more questions."]],
]

# Initialize chatbot and sentiment analyzer
chatbot = Chat(pairs, reflections)
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return "Positive"
    elif sentiment_score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    
    if user_input.lower() == "sentiment":
        return jsonify({
            'response': "Tell me a sentence to analyze.",
            'sentiment_mode': True
        })
    
    if 'sentiment_input' in request.form:
        sentiment_input = request.form['sentiment_input']
        sentiment_result = analyze_sentiment(sentiment_input)
        return jsonify({
            'response': f"Sentiment Analysis Result: {sentiment_result}",
            'sentiment_result': sentiment_result
        })
    
    response = chatbot.respond(user_input)
    if not response:
        response = "I'm not sure how to respond to that. Can you try asking differently?"
    
    return jsonify({
        'response': response,
        'sentiment_mode': False
    })

if __name__ == '__main__':
    app.run(debug=True)