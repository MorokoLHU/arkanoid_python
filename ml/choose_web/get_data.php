<?php
header('Content-Type: application/json');
$stages_brick = [1 => 5, 2 => 16, 3 => 27, 4 => 32, 5 => 43];
$modelFile = __DIR__ . '/../../result/Model_name.csv';
$resultFile = __DIR__ . '/../../result/result.csv';
// 讀取模型名稱 (假設你的資料在 Model_name.csv 中)
$modelNames = array_map('str_getcsv', file($modelFile));
$modelRaw = array_map('str_getcsv', file($modelFile));
array_shift($modelRaw); // 移除標題列
$modelNames = array_column($modelRaw, 0);

// 讀取結果 (假設你的資料在 result.csv 中)
$resultRaw = array_map('str_getcsv', file($resultFile));
array_shift($resultRaw); // 移除標題列
$results = $resultRaw;
$combined = [];
foreach ($resultRaw as $index => $result) {
    if (!isset($modelNames[$index]) || count($result) < 4) continue;
    $combined[] = array_merge([$modelNames[$index]], $result);
}

$passRates = $breakRates = $totalRuns = $passtimes = [];
$stats = [];

foreach ($combined as $row) {
    list($model, $stage, $brick_remain, $catch_count, $state) = $row;
    $stage = (int)$stage;
    $brick_remain = (int)$brick_remain;
    $catch_count = (int)$catch_count;

    if (!isset($stats[$model])) {
        $stats[$model] = [
            'total_runs' => 0,
            'FINISH' => 0,
            'FAIL' => 0,
            'total_catch' => 0,
            'total_bricks' => 0,
            'total_remain' => 0
        ];
    }

    if ($state !== 'FINISH' && $state !== 'FAIL') continue;

    $stats[$model]['total_runs']++;
    $stats[$model][$state]++;
    $stats[$model]['total_catch'] += $catch_count;
    $stats[$model]['total_remain'] += $brick_remain;
    $stats[$model]['total_bricks'] += $stages_brick[$stage] ?? 0;
}

$summary = [];
foreach ($stats as $model => $data) {
    if ($data['total_runs'] > 0 && $data['total_bricks'] > 0) {
        $summary[] = [
            'model' => $model,
            'pass_rate' => $data['FINISH'] / $data['total_runs'],
            'passtimes' => $data['FINISH'],
            'break_rate' => 1 - ($data['total_remain'] / $data['total_bricks']),
            'total_runs' => $data['total_runs']
        ];
    }
}

// 根據通關率排序（高到低）
usort($summary, function ($a, $b) {
    return $b['pass_rate'] <=> $a['pass_rate'];
});

// 拆出資料欄位
$labels = array_column($summary, 'model');
$passRates = array_column($summary, 'pass_rate');
$passtimes = array_column($summary, 'passtimes');
$breakRates = array_column($summary, 'break_rate');
$totalRuns = array_column($summary, 'total_runs');

echo json_encode([
    "labels" => $labels,
    "passtimes" => $passtimes,
    "pass_rate" => $passRates,
    "break_rate" => $breakRates,
    "total_runs" => $totalRuns
]);

