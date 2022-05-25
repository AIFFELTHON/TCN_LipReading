#coding=utf-8
import os
import cv2  # OpenCV 라이브러리
import numpy as np


# -- IO utils
# 텍스트 라인 불러오기
def read_txt_lines(filepath):
    # 파일이 있는지 확인, 없으면 AssertionError 메시지를 띄움
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)  # 원하는 조건의 변수값을 보증하기 위해 사용
    
    # 파일 불러오기
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()  # 문자열을 '\n' 기준으로 쪼갠 후 list 생성
    return content


# npz 저장
def save2npz(filename, data=None):                     
    # 데이터가 비어있는지 확인, 없으면 AssertionError 메시지를 띄움               
    assert data is not None, "data is {}".format(data)          
    
    # 파일 없을 경우                 
    if not os.path.exists(os.path.dirname(filename)):                            
        os.makedirs(os.path.dirname(filename))  # 디렉토리 생성
    np.savez_compressed(filename, data=data)  # 압축되지 않은 .npz 파일 형식 으로 여러 배열 저장


# 비디오 불러오기
def read_video(filename):
    cap = cv2.VideoCapture(filename)  # 영상 객체(파일) 가져오기

    while(cap.isOpened()):  # 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부  
        # ret: 정상적으로 읽어왔는가?
        # frame: 한 장의 이미지(frame) 가져오기                                                      
        ret, frame = cap.read() # BGR                                            
        if ret:  # 프레임 정보를 정상적으로 읽지 못하면                                                                  
            yield frame  # 프레임을 함수 바깥으로 전달하면서 코드 실행을 함수 바깥에 양보                                                        
        else:  # 프레임 정보를 정상적으로 읽지 못하면                                                                    
            break  # while 빠져나가기                                                                 
    cap.release()  # 영상 파일(카메라) 사용 종료
