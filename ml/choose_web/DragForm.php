<?php
// 開啟錯誤顯示
ini_set('display_errors', 1);
error_reporting(E_ALL);

// 設定資料夾路徑
$dataFolder = __DIR__ . '/../save'; // 使用相對路徑指向 save 資料夾

// 確認資料夾是否存在
if (!is_dir($dataFolder)) {
    echo json_encode([
        'error' => true,
        'error_code' => 'FOLDER_NOT_FOUND',
        'message' => '資料夾不存在: ' . $dataFolder
    ]);
    exit;
}

// 讀取資料夾內所有檔案
$files = scandir($dataFolder);

// 檢查是否有讀取錯誤
if ($files === false) {
    echo json_encode([
        'error' => true,
        'error_code' => 'SCANDIR_FAILED',
        'message' => '無法讀取資料夾'
    ]);
    exit;
}

// 過濾掉 "." 和 ".." 目錄，並且只保留 .pickle 的檔案
$fileNames = array_filter($files, function($file) {
    return pathinfo($file, PATHINFO_EXTENSION) === 'pickle';
});

// 去除副檔名，只保留檔名
$fileNames = array_map(function($file) {
    return pathinfo($file, PATHINFO_FILENAME);
}, $fileNames);

// 檢查是否有檔案
if (empty($fileNames)) {
    echo json_encode([
        'error' => true,
        'error_code' => 'NO_FILES_FOUND',
        'message' => '資料夾內沒有 .pickle 檔案'
    ]);
    exit;
}

// 將檔案名稱轉換為數值型的陣列
$fileNames = array_values($fileNames);

// 回傳檔案名稱
header('Content-Type: application/json'); // 設定 Content-Type 為 JSON
echo json_encode([
    'error' => false,
    'file_names' => $fileNames // 回傳檔案名稱陣列
]);
?>
