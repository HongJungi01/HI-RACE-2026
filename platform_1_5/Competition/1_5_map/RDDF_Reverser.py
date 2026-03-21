import pandas as pd
import os

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, '1_5_map_middle2.csv')
output_file = os.path.join(current_dir, '1_5_map_middle2_reversed.csv')

try:
    # 1. CSV 읽기
    df = pd.read_csv(input_file, header=None)

    # 2. 데이터 부분만 역순으로 뒤집기 (1번 열부터 끝까지)
    # .iloc[::-1]로 행을 뒤집고, .reset_index(drop=True)로 데이터프레임 내부 인덱스 초기화
    data_reversed = df.iloc[::-1, 1:].reset_index(drop=True)

    # 3. 기존의 순차적인 인덱스(0번 열) 가져오기
    original_index = df.iloc[:, 0].reset_index(drop=True)

    # 4. 순차 인덱스와 뒤집힌 데이터를 다시 합치기
    df_final = pd.concat([original_index, data_reversed], axis=1)

    # 5. 저장
    df_final.to_csv(output_file, index=False, header=False)

    print(f"정렬 완료 (인덱스 유지형): {output_file}")

except Exception as e:
    print(f"오류 발생: {e}")