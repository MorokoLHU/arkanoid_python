<?php
// 確認表單已經提交
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // 取得選項 1、選項 2 和選項 3 的值
    $option1 = $_POST['option1'];
    $option2 = $_POST['option2'];
    $option3 = $_POST['option3'];

    // 顯示選擇的選項
    echo "選項 1: " . $option1 . "<br>";
    echo "選項 2: " . $option2 . "<br>";
    echo "選項 3: " . $option3 . "<br>";

    // 指定 Python 路徑
    $pypath = 'C:\\Users\\Moroco\\Documents\\python\\FUNAI\\.venv\\Scripts\\python'; // 使用雙斜線處理路徑

    // 呼叫 Python 腳本並傳遞 $option2 作為參數
    $command = $pypath . " C:\\Users\\Moroco\\Documents\\python\\FUNAI\\.venv\\Scripts\\MLGAME\\lhu-csie-arkanoid-main\\ml\\model_train_KNN_classification.py " . escapeshellarg($option2);

    // 執行命令並取得輸出
    $output = shell_exec($command);
    echo "訓練中"."<br>";
    // 顯示 Python 腳本的輸出
    echo $output;
    echo "訓練完畢";
}
?>
