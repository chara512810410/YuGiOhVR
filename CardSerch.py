import cv2

# YOLOの構成ファイルと重みファイルのパスを指定
net = cv2.dnn.readNet("C:\\Users\\Chara\\YOLO\\yolov3.cfg", 
                      "C:\\Users\\Chara\\YOLO\\yolov3.weights")

# 残りのコードは同じ
cap = cv2.VideoCapture(0)

# 検出ループのカウンタを初期化
loop_counter = 0
detection_interval = 100  # 100ループごとに検出

while True:
    # フレームを取得
    _, frame = cap.read()  # retは使わないので _ で受ける

    # 検出を特定のループごとにのみ実行
    if loop_counter % detection_interval == 0:
        # 物体検出を行う
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)
        print(f"{detection_interval} ループごとの物体検出を実行中")

        # 検出結果の処理
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]
                if confidence > 0.5:
                    print(f"検出されたクラス: {class_id}, 信頼度: {confidence}")
    
    # 映像を表示
    cv2.imshow("Live Video", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ループカウンタを増加
    loop_counter += 1

# リソースの解放
cap.release()
cv2.destroyAllWindows()
