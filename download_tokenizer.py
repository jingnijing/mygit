from transformers import RobertaTokenizer

# 모델 디렉토리와 저장할 경로
models_directory = 'C:/mygit/models02'
tokenizer_save_directory = 'C:/mygit/tokenizers'  # 토크나이저를 저장할 경로

# 산업별 모델 이름 목록
industries = ['디자인', 'IT', '금융', '마케팅', '의료보건', '법률', '교육', '에너지']

# 토크나이저 저장 함수
def save_tokenizer(industry):
    model_dir = f'{models_directory}/trained_qa_roberta_{industry}_industry'
    
    # 로드 토크나이저
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Base 모델에서 가져오기

    # 저장할 경로 설정
    tokenizer.save_pretrained(f'{tokenizer_save_directory}/tokenizer_{industry}')

# 모든 산업에 대해 토크나이저 저장
for industry in industries:
    save_tokenizer(industry)
    print(f"{industry} 산업 모델에서 토크나이저를 저장했습니다.")
