from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import re
import os

# Create your views here.

os.environ['LANGCHAIN_API_KEY'] = 'AIzaSyB9GVhtXGTC49nFCdPfkfwpZjlIB2cgsKI'
# os.environ["Google_API_KEY"]='AIzaSyCAy9dh_mP21KL0FZvdfNFKuX4y6ZnBHHs'
# os.environ["Google_API_KEY"]='AIzaSyB9GVhtXGTC49nFCdPfkfwpZjlIB2cgsKI'
# llm = GoogleGenerativeAI(model="gemini-pro",google_api_key='AIzaSyDpbCpx4NQj9zrHZBg6zLPRB7oOD_IhsZA', temperature=0.7)

def index(request):
    return render(request,"index.html")

def getSummary(full_text):
    llm = GoogleGenerativeAI(model="gemini-pro",google_api_key='AIzaSyDpbCpx4NQj9zrHZBg6zLPRB7oOD_IhsZA', temperature=0.7,max_retries=100)

    # Define prompt template
    prompt_template = """
    Please summarize the following legal document in concise and clear language. Provide the output as a single paragraph, formatted with relevant line breaks where needed.
    
    Text: {text}

    Summary (200+ words):
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    output_parser = StrOutputParser()

    # Create the chain
    chain =  PROMPT | llm | output_parser

    # Process the document
    result = chain.invoke({"text": full_text})

    return result

def getKeyPoints(full_text):
    llm = GoogleGenerativeAI(model="gemini-pro",google_api_key='AIzaSyBFIDG4v2UWeRKdgrRd4cjdxZklygyD4tU', temperature=0.7,max_retries=100)

    prompt_template = """
    You are a legal expert tasked with extracting key points from a legal document. Analyze the following legal text carefully first give a short 4 to 5 lines summary about document and then provide a comprehensive list of the most important points and please dont't make sub points of a point:

    Present your findings as a numbered list and please give output in a nice clean format:

    Legal Text: {text}
    
    Format your response as follows:
    Summary : (write 4-5 lines short summary here about document)
    
    1. (Write key point 1 here)
    2. (Write key point 2 here)
    3. (Write key point 3 here)
    4. (Write key point 4 here)
    5. (Write key point 5 here)
    
    and so on for all the key points you found.

    Key Points: 

    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    output_parser = StrOutputParser()

    # Create the chain
    chain =  PROMPT | llm | output_parser

    # Process the document
    result = chain.invoke({"text": full_text})
    key_points = re.findall(r'\d+\.\s*(.*)', result)
    print(result)

    return result
    
def doRiskAssesment(full_text):
    llm = GoogleGenerativeAI(model="gemini-pro",google_api_key='AIzaSyBUTrs8eF0ezuBNZfha7uKObjI2TFwnjaA', temperature=0.7,max_retries=100)

    risk_criteria = {
    "Compliance issues": ["regulation", "compliance", "law", "licensing", "reporting requirements"],
    "Unfavorable terms": ["penalty", "liability", "termination", "indemnity", "damages", "warranties", "representations"],
    "Missing key clauses": ["indemnification", "confidentiality", "dispute resolution", "governing law", "force majeure", "limitation of liability", "intellectual property rights", "non-compete", "non-solicitation"],
    "Ambiguous language": ["may", "might", "could", "should", "vague terms"],
    "Financial risks": ["payment terms", "interest rates", "currency exchange", "price adjustment clauses"],
    "Operational risks": ["delivery terms", "performance obligations", "service levels", "maintenance", "support"],
    "Data privacy and security": ["data protection", "data breach notification", "data handling", "data storage", "privacy policies"],
    "Employment and labor issues": ["employment contracts", "labor law compliance", "employee benefits", "termination conditions"],
    "Environmental and safety compliance": ["environmental regulations", "health and safety standards", "hazardous materials handling"],
    "Ethical and social responsibility": ["anti-bribery", "corruption", "human rights", "corporate social responsibility"]
}

    criteria_str = ""
    for category, terms in risk_criteria.items():
        criteria_str += f"- {category.lower()}: {', '.join(terms)}\n"
    prompt_template = """
    You are a legal expert tasked with identifying and assessing potential legal risks in a document. Analyze the following legal text carefully 
    and provide a list of potential risks based on these criteria:
    {criteria_str}

    For each risk you find in the text:
    1. Describe the potential legal risk.
    2. Assign a risk level (Low, Medium, or High).
    3. Provide a brief explanation for the assigned risk level.
    4. Mention the actual words or phrases in the agreement that you found risky.

    Use the following criteria for risk levels:
    - Low: Minor issues that are unlikely to result in significant legal or financial consequences.
    - Medium: Moderate issues that could potentially lead to legal disputes or financial losses if not addressed.
    - High: Serious issues that are likely to result in significant legal or financial consequences if not addressed immediately.

    Important: Only mention a risk if there is clear evidence in the text. Do not invent risks. Provide specific lines from {text} as proof for each identified risk.

    Format your response as follows:

    RISK: [Risk name]
    RISK LEVEL: [High/Medium/Low]
    EXPLANATION: [Explanation]
    EVIDENCE: [Evidence from the text that is identified as risk]



    Legal Text: {text}

    Risk Assessment: 

    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text","criteria_str"])
    output_parser = StrOutputParser()
    chain =  PROMPT | llm | output_parser
    result = chain.invoke({"text": full_text,"criteria_str":criteria_str})
    # final_result=re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', result)
    print(result)

    return result

def clauseSuggestions(full_text):
    llm = GoogleGenerativeAI(model="gemini-pro",google_api_key='AIzaSyBc8hc0zlhxRlJibJaelx6Sm3uVYPEhntE', temperature=0.7,max_retries=100)
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key='AIzaSyCAy9dh_mP21KL0FZvdfNFKuX4y6ZnBHHs')
    vector_db = FAISS.load_local("C:/Users/RTC/Desktop/clone/Legal-Docs-Analyzer-Generative-Ai/Legal_Docs_Project/webapp/faiss_index", google_embeddings,allow_dangerous_deserialization=True)
    docs = vector_db.similarity_search(full_text)
    
    prompt_template = """
    You are a legal expert tasked with suggesting clauses for this {text} legal document based on the following similar clauses {similar_clauses}. 
    Please rephrase and suggest these clauses in your own wording as a numbered list, ensuring they are suitable for inclusion in a legal document.

    Legal Document: {text}

    Similar Clauses: {similar_clauses}


    IMPORTANT:

    -Please ensure that each clause length should not be more than 30 words.
    -Suggest only that clause that you think is not present in legal document.Don't suggest clauses that are already present in legal document

    Suggested Clauses:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text","similar_clauses"])
    output_parser = StrOutputParser()
    # Create the chain
    chain =  PROMPT | llm | output_parser

    # Process the document
    result = chain.invoke({"text": full_text,"similar_clauses":docs})
    print(result)

    return result
    
def mainApi(request):
    upload_dir='C:/Users/RTC/Desktop/clone/Legal-Docs-Analyzer-Generative-Ai/uploads'
    
    for file_name in os.listdir(upload_dir):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(upload_dir, file_name)
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    full_text = " ".join([doc.page_content for doc in docs])
    
    summary_and_keypoints=getKeyPoints(full_text)
    risks=doRiskAssesment(full_text)
    clauses=clauseSuggestions(full_text)
    
    return render(request,"detail.html",{"summary_and_keypoints":summary_and_keypoints,"risks":risks,"clauses":clauses})

@csrf_exempt
@require_http_methods(["POST"])

def uploadPdf(request):
    print("request received")
    
    # Define the path where you want to save the uploaded files
    upload_dir = '../uploads'  # Replace with your desired path
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists
    
    # Check if the request has a file
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    uploaded_file = request.FILES['file']
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    # Save the uploaded file to the specified path without chunking
    with open(file_path, 'wb') as destination:
        destination.write(uploaded_file.read())
    return JsonResponse({"Success ": True},status=200)
    

