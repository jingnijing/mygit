from flask import Flask, render_template, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

# 산업 분야에 따른 고정 질문 (테스트용 고정 질문)
industry_questions = {
    "IT": "IT 산업에서 팀 리더로서 수행한 경험에 대해 설명해주세요.",
    "교육": "교육 프로그램을 기획하고 운영했던 경험에 대해 설명해주세요.",
    "금융": "금융 데이터 분석을 통해 프로젝트를 성공적으로 이끈 경험에 대해 설명해주세요.",
    "의료보건": "환자의 건강 상태를 모니터링하고 개선한 경험에 대해 설명해주세요."
}

# 모델 로드 함수
def load_model(industry_name):
    model_dir = f'C:/mygit/models/trained_qa_roberta_{industry_name}_industry'
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

# 홈 페이지에서 고정된 질문을 사용자에게 보여주는 라우트
@app.route('/')
def index():
    industry = request.args.get('industry', 'IT')  # 기본값을 IT로 설정 (테스트용)
    question = industry_questions.get(industry, "산업에 맞는 질문을 설정해 주세요.")
    return render_template('index.html', question=question, industry=industry)  # 질문을 넘겨줌

# POST 요청을 처리하는 라우트 (예측 수행)
@app.route('/predict', methods=['POST'])
def predict():
    # POST 요청에서 JSON 데이터를 받음
    data = request.json
    input_text = data.get('text')  # 'text' 필드에서 텍스트 데이터 추출
    industry = data.get('industry')  # 'industry' 필드에서 산업 분야 추출

    # 모델 로드
    tokenizer, model = load_model(industry)

    # 입력 텍스트를 토크나이즈
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    # 토큰화된 입력을 출력하여 확인
    print(f"Tokenized input: {inputs}")

    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Softmax를 적용하여 확률 계산
        probabilities = torch.softmax(logits, dim=-1)
        
        # 각 클래스별 확률 값 출력
        print(f"Probabilities: {probabilities}")

        # 여러 클래스가 있을 경우, 확률 값의 평균을 사용하여 점수 예측
        predicted_score = torch.mean(probabilities).item() * 100

    # 예측된 점수를 사용자에게 반환
    return jsonify({'predicted_score': round(predicted_score, 2)})  # 소수점 둘째 자리까지 반환

if __name__ == '__main__':
    app.run(debug=True)
