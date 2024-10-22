from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

app = Flask(__name__)

# 모델 로드 함수 (산업별 모델을 로드)
def load_model(industry_name):
    model_dir = f'C:/mygit/models02/trained_qa_roberta_{industry_name}_industry'
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

# 홈 페이지에서 고정된 질문을 사용자에게 보여주는 라우트
@app.route('/')
def index():
    industry = "IT"  # 기본값 또는 사용자 선택에 따라 조정
    question = "IT 산업에서 팀 리더로서 수행한 경험에 대해 설명해주세요."
    return render_template('index.html', question=question, industry=industry)  # 질문과 산업 정보 전달

# POST 요청을 처리하는 라우트 (산업 분야에 따른 점수 예측)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text')  # 사용자 입력 텍스트 (답변)
    industry = data.get('industry')  # 산업분야 (사용자가 선택한 산업분야)

    # 산업별 모델 로드
    tokenizer, model = load_model(industry)

    # 입력 텍스트를 토크나이즈
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # 모델로부터 점수 예측 (logits에서 점수 추출)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_score = outputs.logits.squeeze().item()  # 예측된 점수

    # 정규화 범위 정의 (학습 데이터의 최소/최대 점수를 여기에 설정)
    min_possible = 0  # 최소 점수 (예: 0)
    max_possible = 1  # 최대 점수 (예: 1) - 실제로 학습한 데이터에 따라 다를 수 있음

    # 예측된 점수를 0과 1 사이로 정규화
    normalized_score = (predicted_score - min_possible) / (max_possible - min_possible) * 100
    final_score = min(normalized_score, 100)  # 100을 초과하지 않도록 제한

    return jsonify({'predicted_score': round(final_score, 2)})


if __name__ == '__main__':
    app.run(debug=True)
