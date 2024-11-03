import xlwings as xw
import pandas as pd

# 엑셀 파일 저장 메소드
file_path = 'C:/Users/leewa/Desktop/파일명.xlsx'
   
wb = xw.Book()

# 예시 데이터프레임 생성
State_list = []

State_df = pd.DataFrame(State_list)

# 행 번호 추가
State_df.insert(0, '일별 관리', [f'{i}일차' for i in range(1, len(State_list) + 1)])

sheet_count = len(wb.sheets)
sheet_name=f'Episode{sheet_count+1}'

if sheet_name in [sheet.name for sheet in wb.sheets]:
    sheet = wb.sheets[sheet_name]
else:    
    sheet = wb.sheets.add(name=sheet_name,after=wb.sheets[-1])
    
sheet.range('A1').value = [State_df.columns.tolist()] + State_df.values.tolist()

# 엑셀 파일 저장
wb.save(file_path)

print(f'Episode {sheet_count+1} State 저장 완료')

# 엑셀 파일 닫기
wb.close()

print("학습 완료")


# 예시
action_list = [[1,2,3],[12,3,4],[12,3,4]]

#끝
action_df = pd.DataFrame(action_list)

# 행 번호 추가로 인한 열 수 증가 반영
num_columns = action_df.shape[1]  
column_names = [f'{i+1}일차' for i in range(num_columns)]

action_df.columns = column_names

# 행 번호 추가
action_df.insert(0, 'Episode Number', [f'{i}번 Episode' for i in range(1, len(action_list) + 1)])

print(action_df)

action_df.to_excel('C:/Users/leewa/Desktop/action.xlsx',index= False)