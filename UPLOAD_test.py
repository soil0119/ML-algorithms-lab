

# Add 변경된 부분 중 remote에 보낼 코드 선택&준비
# Commit 올릴코드 설명
# Push 코드 업데이트 Local과 Remote코드 동기화

## 사용 명령어

# git add (파일명)
# git commit -m "first commit" (-m은 메세지형태)
# git branch -M main (브랜치 이름을 메인으로 한다)
# git remote add origin [본인 git repo주소].git (origin은 주소로 설정됨)
# git push -u origin main


# Commit Message

# 어떤것을 변화시켰는지를 파악하기위해 알아보기 쉽게 적을 것

## 추천 Commit 카테고리 
# Add : ~기능을 하는 모듈 / 함수 / 파일 등을 추가
# Modify : ~기능 추가 / 업데이트 등을 위해 코드 변경
# Fix : ~에러 수정, 논리 재구조화, 입/출력 변경

# Git Pull

# 원격 repo의 내용을 로컬 repo로 가지고 와서 덮어 씌우는 과정

print('github에 올라간 파일을 변경하면 line 옆에 초록줄이 생김')
print('파일이름도 노란색으로 바뀜')
