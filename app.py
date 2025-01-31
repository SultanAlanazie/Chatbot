from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import numpy as np
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

load_dotenv()

# groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize ChatGroq LLM

llm = ChatGroq(
    model='llama3-70b-8192',
    temperature=0,
)

PROMPT_TEMPLATE = """
### تعليمات:
*أنت خبير في تحليل البيانات والنصوص.*
** مهمتك هي تلخيص الإجابات وتقديم ردود دقيقة بناءً على البيانات المقدمة من الاجتماع السابق و إذا كانت بداية المحادثة فرحب أولا.**
*يجب أن تكون الإجابة شاملة، واضحة، ومبنية على التفاصيل المتاحة.*
*الرد يجب أن يكون باللغة العربية الفصحى، مع مراعاة التوضيح والشرح عند الحاجة.*
*إذا كانت هناك معلومات غير واضحة أو تحتاج إلى توضيح إضافي، يمكنك طرح أسئلة لتحديد النقاط الغامضة.*
---
### السؤال: {subject}
---
### الإجابة:
"""

# Define prompt template for the chatbot
prompt = PromptTemplate(
    input_types={'subject': 'string', 'language': 'string'},
    template=PROMPT_TEMPLATE
)

# Initialize the chain for the chatbot
chain = LLMChain(llm=llm, prompt=prompt)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/meeting')
def models():
    return render_template('Meeting.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Initialize messages in session if not already present
        if 'messages' not in session:
            session['messages'] = []

        # Append user input to session messages
        session['messages'].append({"role": "You", "content": user_input})

        # Construct the conversation history for context
        history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in session['messages']]
        )
        
        # Update the prompt to include history
        subject = f"{history}\nYou: {user_input}"
        language = "Arabic"

        # Generate response from the chain
        response = chain.run(subject=subject, language=language)

        # Append the chatbot response to session messages
        session['messages'].append({"role": "KhabeerAI", "content": response})

        # Mark session as modified to save changes
        session.modified = True

        return redirect(url_for('chatbot'))

    # Retrieve all messages stored in session
    messages = session.get('messages', [])
    return render_template('chatbot.html', messages=messages)


if __name__ == '__main__':
    app.run(debug=True)