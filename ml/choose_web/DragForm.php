<?php
// 顯示錯誤資訊以方便除錯
ini_set('display_errors', 1);
error_reporting(E_ALL);
 
// 資料夾與檔案路徑
$dataFolder = __DIR__ . '/../save';
$modelCsvPath = __DIR__ . '/../../result/model_name.csv';

// 🧽 正規化模型名稱用的函式
function normalizeModelName($name) {
    $name = preg_replace('/[\x00-\x1F\x7F]/u', '', $name); // 移除控制字元
    $name = preg_replace('/^\xEF\xBB\xBF/', '', $name);    // 移除 UTF-8 BOM
    $name = trim($name);
    return str_replace('.', '', $name); // 可選：移除小數點，避免 "17.12" vs "1712" 不一致
}

// ✅ 檢查資料夾是否存在
if (!is_dir($dataFolder)) {
    echo json_encode([
        'error' => true,
        'error_code' => 'FOLDER_NOT_FOUND',
        'message' => '資料夾不存在: ' . $dataFolder
    ]);
    exit;
}

// ✅ 取得所有 .pickle 檔案（去除副檔名）
$files = scandir($dataFolder);
if ($files === false) {
    echo json_encode([
        'error' => true,
        'error_code' => 'SCANDIR_FAILED',
        'message' => '無法讀取資料夾'
    ]);
    exit;
}

$fileNames = array_filter($files, function($file) {
    return pathinfo($file, PATHINFO_EXTENSION) === 'pickle';
});
$fileNames = array_map(function($file) {
    return pathinfo($file, PATHINFO_FILENAME);
}, $fileNames);

if (empty($fileNames)) {
    echo json_encode([
        'error' => true,
        'error_code' => 'NO_FILES_FOUND',
        'message' => '資料夾內沒有 .pickle 檔案'
    ]);
    exit;
}

// ✅ 載入 CSV 並處理模型名稱 - 修正版本
$modelList = [];
if (file_exists($modelCsvPath)) {
    $csvContent = file_get_contents($modelCsvPath);
    $csvContent = preg_replace('/^\xEF\xBB\xBF/', '', $csvContent); // 移除 BOM
    
    $lines = explode("\n", $csvContent);
    
    foreach ($lines as $lineIndex => $line) {
        $line = trim($line);
        if (empty($line)) continue;
        
        // 跳過表頭
        if ($lineIndex === 0 && stripos($line, 'model_name') !== false) {
            continue;
        }
        
        // 處理可能的多個模型名稱在同一行的情況
        if (strpos($line, ' ') !== false) {
            // 如果包含空格，可能是多個模型名稱用空格分隔
            $models = preg_split('/\s+/', $line);
            foreach ($models as $model) {
                $model = trim($model);
                if (!empty($model) && stripos($model, 'model_name') === false) {
                    $normalizedModel = normalizeModelName($model);
                    if (!empty($normalizedModel)) {
                        $modelList[] = $normalizedModel;
                    }
                }
            }
        } else {
            // 單一模型名稱
            $normalizedModel = normalizeModelName($line);
            if (!empty($normalizedModel) && stripos($normalizedModel, 'model_name') === false) {
                $modelList[] = $normalizedModel;
            }
        }
    }
    
    $modelList = array_unique($modelList); // 去除重複
}

// 除錯：輸出解析到的模型清單
error_log("解析到的模型清單: " . print_r($modelList, true));

// ✅ 檔案清單標記 highlight
$results = array_map(function($name) use ($modelList) {
    $normalizedName = normalizeModelName($name);
    $match = in_array($normalizedName, $modelList);

    // 除錯用 log - 加強版
    error_log("檔案: [$name] => 正規化: [$normalizedName] => 匹配: " . ($match ? 'YES' : 'NO'));
    if (!$match) {
        error_log("CSV中的模型: " . implode(', ', $modelList));
    }

    return [
        'name' => $name,
        'highlight' => $match
    ];
}, $fileNames);

// ✅ 回傳 JSON 結果
header('Content-Type: application/json');
echo json_encode([
    'error' => false,
    'file_names' => array_values($results),
    'debug' => [
        'model_list_count' => count($modelList),
        'first_few_models' => array_slice($modelList, 0, 5)
    ]
]);
?>