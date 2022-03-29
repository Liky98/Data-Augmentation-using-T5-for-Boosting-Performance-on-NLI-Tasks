sample_outputs = model.generate(
    input_ids,
    do_sample=True, #샘플링 전략 사용
    max_length=50, # 최대 디코딩 길이는 50
    top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
    top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
    num_return_sequences=3 #3개의 결과를 디코딩해낸다
)