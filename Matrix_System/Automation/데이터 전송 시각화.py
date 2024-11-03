import xlwings as xw
import pandas as pd

# 엑셀 파일 경로
file_path = 'C:/Users/leewa/Desktop/exam.xlsx'

# 새로운 엑셀 워크북 생성
wb = xw.Book()
sheet = wb.sheets[0]  # 첫 번째 시트를 선택합니다.

# 예시 데이터프레임 생성
State_df = pd.read_excel('C:/Users/leewa/Desktop/e-MFG/다이텍/데이터/원본데이터.xlsx')

# 데이터프레임의 컬럼명을 엑셀에 쓰기
for col_num, column in enumerate(State_df.columns):
    sheet.range(1, col_num + 1).value = column

# 데이터프레임의 데이터를 엑셀에 순차적으로 쓰기
for row_num, row in State_df.iterrows():
    for col_num, value in enumerate(row):
        sheet.range(row_num + 2, col_num + 1).value = value

# 엑셀 파일 저장
wb.save(file_path)
wb.close()


import xlwings as xw
import pandas as pd
import time

# 엑셀 파일 경로
file_path = 'C:/Users/leewa/Desktop/exam.xlsx'

# 새로운 엑셀 워크북 생성
wb = xw.Book()
sheet = wb.sheets[0]  # 첫 번째 시트를 선택합니다.

# 예시 데이터프레임 생성
State_df = pd.read_excel('C:/Users/leewa/Desktop/e-MFG/다이텍/데이터/원본데이터.xlsx')

# 데이터프레임의 컬럼명을 엑셀에 쓰기
for col_num, column in enumerate(State_df.columns):
    sheet.range(1, col_num + 1).value = column
    time.sleep(0.1)  # 지연 시간 추가

# 데이터프레임의 데이터를 엑셀에 순차적으로 쓰기
for row_num, row in State_df.iterrows():
    for col_num, value in enumerate(row):
        sheet.range(row_num + 2, col_num + 1).value = value
        time.sleep(0.1)  # 지연 시간 추가

# 엑셀 파일 저장
wb.save(file_path)