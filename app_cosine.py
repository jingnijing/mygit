from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# 1. 산업별 모델 경로 설정 (models02 폴더 아래에 있는 8개의 산업별 모델)
model_paths = {
    "IT": "./models02/trained_qa_roberta_IT_industry",
    "교육": "./models02/trained_qa_roberta_교육_industry",
    "금융": "./models02/trained_qa_roberta_금융_industry",
    "디자인": "./models02/trained_qa_roberta_디자인_industry",
    "마케팅": "./models02/trained_qa_roberta_마케팅_industry",
    "의료보건": "./models02/trained_qa_roberta_의료보건_industry",
    "법률": "./models02/trained_qa_roberta_법률_industry",
    "에너지": "./models02/trained_qa_roberta_에너지_industry"
}

# 2. 산업별 기준 답변 설정
reference_answers = {
    "IT": ["IT 산업에서 팀 리더로서의 경험을 설명해주세요."],
    "교육": ["교육 프로그램을 기획하고 운영한 경험에 대해 설명해주세요."],
    "금융": ["금융 데이터 분석 프로젝트를 설명해주세요."],
    "디자인": ["디자인 프로젝트에서 팀을 이끌었던 경험에 대해 설명해주세요."],
    "마케팅": ["마케팅 캠페인을 기획한 경험을 공유해주세요."],
    "의료보건": ["의료 프로젝트에서 팀을 이끌었던 경험을 설명해주세요."],
    "법률": ["법률 관련 프로젝트를 수행했던 경험을 설명해주세요."],
    "에너지": ["에너지 산업에서 진행했던 프로젝트 경험을 공유해주세요."]
}

# 3. 모델 로드 함수 (산업별 모델 로드)
def load_model_and_tokenizer(industry_name):
    model_dir = model_paths[industry_name]
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = RobertaModel.from_pretrained(model_dir, local_files_only=True)
    return tokenizer, model

# 4. 임베딩 추출 함수
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # 평균 임베딩 사용
    return embedding.numpy()

# 5. 유사도 계산 함수 (코사인 유사도)
def calculate_similarity(vector1, vector2):
    return cosine_similarity(vector1, vector2)[0][0]

# 6. 홈 페이지에서 고정된 질문을 사용자에게 보여주는 라우트
@app.route('/')
def index():
    industry = request.args.get('industry', 'IT')  # 기본 산업은 IT
    question = reference_answers[industry][0]  # 해당 산업의 첫 번째 질문을 표시
    return render_template('index.html', question=question, industry=industry)

# 7. POST 요청을 처리하는 라우트 (유사도 기반 예측 수행)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text')  # 사용자가 입력한 답변
    industry = data.get('industry')  # 사용자가 선택한 산업분야

    # 모델과 토크나이저 로드
    if industry not in model_paths:
        return jsonify({"error": "잘못된 산업 분야입니다."}), 400

    tokenizer, model = load_model_and_tokenizer(industry)

    # 사용자 입력 벡터 계산
    user_vector = get_embedding(input_text, tokenizer, model)

    # 산업별 기준 답변 벡터 계산
    reference_texts = reference_answers[industry]
    reference_vectors = [get_embedding(text, tokenizer, model) for text in reference_texts]

    # 유사도 계산
    similarities = [calculate_similarity(user_vector, ref_vector) for ref_vector in reference_vectors]
    average_similarity = sum(similarities) / len(similarities)

    # 0에서 100점 범위로 점수 변환
    final_score = average_similarity * 100  # 유사도 값을 0~100으로 변환

    return jsonify({'predicted_score': round(final_score, 2)})

if __name__ == '__main__':
    app.run(debug=True)
