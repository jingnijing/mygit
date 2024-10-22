# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 산업 분야에 따른 고정 질문 (테스트용 고정 질문)
industry_questions = {
    "IT": "지금까지 살아오면서 가장 많은 노력을 쏟아부었던 성공 혹은 실패 경험과 그 과정을 통해 무엇을 배웠는지 소개해주세요.",
    "교육": "교육 프로그램을 기획하고 운영했던 경험에 대해 설명해주세요.",
    "금융": "금융 데이터 분석을 통해 프로젝트를 성공적으로 이끈 경험에 대해 설명해주세요.",
    "의료보건": "환자의 건강 상태를 모니터링하고 개선한 경험에 대해 설명해주세요."
}

# 모델 로드 함수
def load_model(industry_name):
    model_dir = f'C:/mygit/models/trained_qa_roberta_{industry_name}_industry'
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True)
    return tokenizer, model

# 벡터 유사도 계산 함수 (코사인 유사도 사용)
def calculate_similarity(vector1, vector2):
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0][0]

# 홈 페이지에서 고정된 질문을 사용자에게 보여주는 라우트
@app.route('/')
def index():
    industry = request.args.get('industry', 'IT')  # 기본값을 IT로 설정 (테스트용)
    question = industry_questions.get(industry, "산업에 맞는 질문을 설정해 주세요.")
    return render_template('index.html', question=question, industry=industry)  # 질문을 넘겨줌

# POST 요청을 처리하는 라우트 (유사도 기반 예측 수행)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text')  # form에서 전달된 텍스트
    industry = data.get('industry')  # form에서 전달된 산업 분야

    # 모델 로드
    tokenizer, model = load_model(industry)

    # 입력 텍스트를 토크나이즈
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # 모델로부터 벡터 추출 (hidden state에서 추출)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.hidden_states[-1]  # 마지막 레이어의 hidden state 벡터 사용

    # 3차원 벡터에서 평균을 내어 2차원 벡터로 변환
    embeddings = torch.mean(embeddings, dim=1)

    # 기준 벡터 (해당 산업의 기준 벡터를 미리 생성)
    reference_vector = generate_reference_vector_for_industry(industry)

    # PyTorch 텐서를 NumPy 배열로 변환
    embeddings_2d = embeddings.detach().numpy().reshape(1, -1)
    reference_vector_2d = reference_vector.detach().numpy().reshape(1, -1)

    # 유사도 계산 (2차원 벡터로 변환 후)
    similarity_score = cosine_similarity(embeddings_2d, reference_vector_2d)

    # 유사도 스케일링 (0.0 ~ 1.0 범위를 확장하여 50~100점 사이로 변환)
    scaled_score = similarity_score[0][0] * 50 + 50

    return jsonify({'predicted_score': round(scaled_score, 2)})

# 산업별 기준 벡터를 생성하는 함수
def generate_reference_vector_for_industry(industry):
    """
    해당 산업의 예시 자소서 답변을 사용하여 기준 벡터를 생성합니다.
    """
    example_texts = {
        'IT': [
            {"질문": "지원하신 모집분야에 해당하는 특화문항을 확인하여 답변을 기술해주세요.", "답변": "공정기술엔지니어는 일정에 맞춰 효율적으로 업무를 처리해야 합니다. 저는 학생 때 주어진 일을 빨리, 잘 끝냈을 때 성취감을 느끼는 것을 깨달았고 매사에 업무효율을 높이기 위해 고민해왔습니다. 고등학생 때 선생님께서 문제를 빨리 풀면 집에 갈 수 있다고 하셨습니다. 처음엔 빨리 푸는 것에만 집중하니 오답률이 높아 오히려 시간이 오래걸렸습니다. 무조건 빠른 것이 아니라 잘하는 것이 효율적임을 체감했습니다. 이후에는 한 문제라도 실수 없이 풀도록 했고 수십 명의 학생들 중 1등으로 학원을 나서며 뿌듯해하곤 했습니다. 이후 사무아르바이트에서 약 200개의 리서치를 4일 내에 해야 했습니다. 퇴근 후 개인 시간을 내어 자동화기능을 학습했습니다. 저만의 매크로를 생성해 3명이 4일 동안 할 일을 혼자서 하루만에 끝내 팀장님께서 좋아하셨습니다. 건조주행과 같은 새로운 분야에 대한 도전정신이 필요합니다. 저는 성취를 위해선 새로운 것을 경험해 경험의 스펙트럼을 넓히고 활성화에너지를 낮추는 일이 중요하다고 생각합니다. 이에 지난 5년 간 웨이트 트레이닝, 혼자서 국내/해외여행 등 다양한 경험을 하기 위해 노력했습니다. 그 중 7일 간의 제주도일주에서 5일동안 태풍이 왔던 일이 있었습니다. 처음엔 망설였지만 소심한 성격을 고치고 싶다는 목표를 이루기 위해 일정을 강행했습니다. 맞으면 아플 정도로 강한 비바람을 뚫고 하루에 3만보씩 걸었습니다. 6일 째에 태풍이 가고 파란 하늘을 보자 그동안의 노고가 씻겨 내려가는 기분이었습니다. 이후 혼자서 아시아일주, 프랑스 3달 살기라는 더 큰 목표를 달성하며 도전적인 성격으로 바뀌었습니다. 아울러 외국인과 거리낌 없이 소통하는 글로벌 역량도 쌓았습니다. 야간근무에도 끄떡없는 체력이 필요합니다. 식당아르바이트에서 가장 바쁜 금/토/일요일에 하루 12시간씩 서빙하고도 퇴근 후엔 웨이트운동을 했습니다. 처음엔 걱정하셨던 사장님도 직원들 중 가장 성실하다며 칭찬하셨습니다."},
            {"질문":"지금까지 살아오면서 가장 많은 노력을 쏟아부었던 성공 혹은 실패 경험과 그 과정을 통해 무엇을 배웠는지 소개해주세요.", "답변":" 실패의 자산화, 수석졸업 귀가 들리지 않아 학교 수업을 듣지 못했지만 매일 새벽까지 공부했다는 어머니의 말씀을 병풍처럼 들으며 자랐습니다. 그런 저에게 꾸준한 노력은 곧 성공이라는 공식이었습니다. 대학입시에 실패했지만 성장의 기회로 바꾸고자 내가 속한 곳에서 최고를 찍어보자고 결심하고 학과 1등이라는 목표를 세웠습니다. 몇 시간이나 끙끙 앓은 문제를 남들은 쉽게 푸는 것을 보면 과연 1등을 할 수 있을까 싶었지만 관둘 수 없었습니다. 남들보다 느린 대신 시간을 많이 투자하기로 했습니다. 지하철에서는 사람들 틈에 껴서 노트를 외우고 밥 먹을 때도 책을 봤습니다. 모르는 내용은 몇 번이고 교수님을 찾아가 질문했습니다. 새벽까지 공부하다 일출을 보며 귀가하는 나날을 반복한 결과 1학년 1학기에 학과 1등을 넘어 전체 공대생 중 한 명만 받는 홍익인간장학금을 받았습니다. 똑같은 수준의 노력을 4년 지속해 졸업까지 한번도 1등을 놓치지 않았고 실패의 경험은 무엇이든 해낼 수 있다는 자신감으로 바뀌었습니다. 자발적 팀워크 발휘, 장려상 수상 업무효율을 위해선 팀원 간 유대감을 키워 긍정적 팀워크를 만들어야 합니다. 교내대회에서 제가 맡은 코딩에 에러가 자주 발생했습니다. 어떻게든 도와주려는 팀원들의 모습을 보니 그 마음이 느껴져 감사했습니다. 좋은 결과물로 보답하고자 정해진 업무시간 외에 밤을 새우며 교재를 읽고 강의를 들어 C언어를 독학했습니다. 원래 목표는 LED와 모터 2가지 기능이었지만 매일 밤을 새워 작업한 끝에 5가지 기능을 6일만에 구현, 모바일어플까지 개발했습니다. 또 팀원들께서 이런 저를 보고 3D설계/회로연결 등에 자원하며 본인의 역량을 발휘했습니다. 그 결과 7일만에 코딩/전자/기계공학이 융합된 스마트블루투스모빌을 만들어 장려상을 받았습니다. 팀원들과의 배려와 유대감이 성과와 직결됨을 깨달았습니다. 꾸준한 노력으로 실패를 자산화하고 수율을 극대화하겠습니다. 또 솔선수범과 배려로 팀원들을 동기부여하고 성과를 높이겠습니다."},
            {"질문": "데이터 분석 경험에 대해 설명해주세요.", "답변": "데이터 분석 및 인공지능 기반 시스템을 구축한 경험이 있습니다."},
            {"질문": "클라우드 환경에서 서비스를 운영한 경험에 대해 설명해주세요.", "답변": "클라우드 환경에서 다양한 서비스를 성공적으로 런칭했습니다."}
        ],
        '교육': [
            {"질문": "교육 프로그램을 기획하고 운영한 경험에 대해 설명해주세요.", "답변": "저는 교육 프로그램을 기획하고 운영하는 역할을 수행했습니다."}
        ],
        '금융': [
            {"질문": "금융 데이터 분석을 통해 프로젝트를 성공적으로 이끈 경험에 대해 설명해주세요.", "답변": "저는 금융 데이터 분석을 통해 성공적인 프로젝트를 완료했습니다."}
        ],
        '의료보건': [
            {"질문": "의료 프로젝트에서 환자 건강 관리를 모니터링한 경험에 대해 설명해주세요.", "답변": "저는 환자 건강 관리를 모니터링하며 개선 방안을 제시한 경험이 있습니다."}
        ]
    }

    # 해당 산업의 예시 자소서를 기준으로 기준 벡터 생성
    texts = example_texts.get(industry, [])
    if not texts:
        return torch.tensor([0.1] * 768)  # 기본값은 임의 벡터

    all_embeddings = []

    # 각 자소서 텍스트에서 임베딩 추출
    tokenizer, model = load_model(industry)
    for text in texts:
        inputs = tokenizer("질문: " + text['질문'] + " 답변: " + text['답변'], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.hidden_states[-1]
            embeddings = torch.mean(embeddings, dim=1)
            all_embeddings.append(embeddings)

    # 임베딩 벡터들의 평균을 계산하여 기준 벡터 생성
    reference_vector = torch.mean(torch.stack(all_embeddings), dim=0)
    return reference_vector

if __name__ == '__main__':
    app.run(debug=True)
