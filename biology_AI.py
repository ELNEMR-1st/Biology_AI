from flask import Flask, request, jsonify, render_template_string
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)

# Load your AI components
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
vector_db = Chroma(persist_directory="./bio_db", embedding_function=embeddings)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC1K699ZyyHNVNddfSf70QURUoCWWDFYNw"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ======== UNIQUE CUSTOM PROMPT ========
CUSTOM_PROMPT_TEMPLATE = """You are BiologyGPT, an expert biology assistant. Use the following context to answer the question in a CLEAR, STRUCTURED way.

CONTEXT:
{context}

QUESTION: 
{question}

Answer with this EXACT structure:
1. **Main Answer**: [2-3 sentence summary]
2. **Key Points**:
   - Point 1
   - Point 2  
   - Point 3
3. **Source Context**: [Briefly mention which biological concepts this relates to]

If the context doesn't contain the answer, say "I don't have enough biological context to answer this precisely."
"""

PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Create the RAG chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}  # ← UNIQUE PROMPT ADDED!
)

# [Keep the rest of your Flask code the same...]
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🧬 BiologyGPT - Expert Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }
        input[type="text"] { width: 400px; padding: 12px; font-size: 16px; border: 2px solid #4CAF50; border-radius: 5px; }
        button { padding: 12px 24px; font-size: 16px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .answer { background: #f8fff8; padding: 25px; margin: 25px 0; border-left: 5px solid #4CAF50; }
        .source { color: #666; font-size: 14px; margin-top: 15px; }
    </style>
</head>
<body>
    <h1>🧬 BiologyGPT - Expert Research Assistant</h1>
    <form action="/ask" method="post">
        <input type="text" name="question" placeholder="Ask any biology question..." required>
        <button type="submit">Ask BiologyGPT</button>
    </form>
    
    {% if answer %}
    <div class="answer">
        {{ answer|safe }}
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    try:
        result = qa_chain.invoke({"query": question})
        answer = result['result'].replace('\n', '<br>')  # Convert newlines to HTML
        return render_template_string(HTML_TEMPLATE, answer=answer)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, answer=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(port=5000, debug=True)