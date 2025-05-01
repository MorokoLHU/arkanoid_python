<?php
// 設定回傳 JSON
header('Content-Type: application/json');

// save 資料夾路徑
$dataFolder = __DIR__ . '/../save';

// 只接受 DELETE 請求
if ($_SERVER['REQUEST_METHOD'] === 'DELETE') {
    $data = json_decode(file_get_contents('php://input'), true);

    if (isset($data['filename'])) {
        $filename = basename($data['filename']); // 安全只取檔名
        $filePath = $dataFolder . '/' . $filename . ".pickle";
        // 檢查檔案存在才刪
        if (file_exists($filePath)) {
            if (unlink($filePath)) {
                echo json_encode(['success' => true, 'message' => '檔案已刪除']);
            } else {
                echo json_encode(['success' => false, 'message' => '無法刪除檔案']);
            }
        } else {
            echo json_encode(['success' => false, 'message' => '檔案不存在']);
        }
    } else {
        echo json_encode(['success' => false, 'message' => '未提供檔案名稱']);
    }
} else {
    echo json_encode(['success' => false, 'message' => '請使用 DELETE 請求']);
}
?>
