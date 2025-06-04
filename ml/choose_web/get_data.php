<?php
header('Content-Type: application/json');

$stages_brick = [1 => 5, 2 => 16, 3 => 27, 4 => 32, 5 => 43];
$filter_stage = $_GET['stage'] ?? 'all';
$total_bricks = array_sum($stages_brick);

$modelFile = __DIR__ . '/../../result/Model_name.csv';
$resultFile = __DIR__ . '/../../result/result.csv';

// 讀取模型名稱
$modelRaw = array_map('str_getcsv', file($modelFile));
array_shift($modelRaw); // 移除標題列
$modelNames = array_column($modelRaw, 0);

// 讀取結果
$resultRaw = array_map('str_getcsv', file($resultFile));
array_shift($resultRaw); // 移除標題列

$combined = [];
foreach ($resultRaw as $index => $result) {
    if (!isset($modelNames[$index]) || count($result) < 5) continue;
    $combined[] = array_merge([$modelNames[$index]], $result);
}

$stats = [];
$stageStats = [];

foreach ($combined as $row) {
    list($model, $stage, $brick_remain, $catch_count, $state) = $row;
    $stage = (int)$stage;
    $brick_remain = (int)$brick_remain;
    $catch_count = (int)$catch_count;

    if ($filter_stage !== 'all' && $stage != (int)$filter_stage) continue;
    if ($state !== 'FINISH' && $state !== 'FAIL') continue;

    // 模型總體統計初始化
    if (!isset($stats[$model])) {
        $stats[$model] = [
            'total_runs' => 0,
            'FINISH' => 0,
            'FAIL' => 0,
            'total_catch' => 0,
            'total_remain' => 0,
            'total_bricks' => 0,
            'weighted_pass_total' => 0,
            'weighted_break_total' => 0,
        ];
    }

    // 每關資料初始化
    if (!isset($stageStats[$model][$stage])) {
        $stageStats[$model][$stage] = [
            'total_runs' => 0,
            'total_remain' => 0,
            'total_bricks' => 0,
            'finish_count' => 0
        ];
    }

    $stats[$model]['total_runs']++;
    $stats[$model][$state]++;
    $stats[$model]['total_catch'] += $catch_count;
    $stats[$model]['total_remain'] += $brick_remain;
    $stats[$model]['total_bricks'] += $stages_brick[$stage] ?? 0;

    $stageStats[$model][$stage]['total_runs']++;
    $stageStats[$model][$stage]['total_remain'] += $brick_remain;
    $stageStats[$model][$stage]['total_bricks'] += $stages_brick[$stage];
    if ($state === 'FINISH') {
        $stageStats[$model][$stage]['finish_count']++;
    }
}

// 加權計算
foreach ($stats as $model => &$data) {
    foreach ($stages_brick as $stage => $brick_count) {
        if (!isset($stageStats[$model][$stage])) continue;

        $runs = $stageStats[$model][$stage]['total_runs'];
        if ($runs === 0) continue;

        $finish = $stageStats[$model][$stage]['finish_count'];
        $remain = $stageStats[$model][$stage]['total_remain'];
        $totalStageBricks = $stageStats[$model][$stage]['total_bricks'];

        if ($totalStageBricks === 0) continue;

        $pass_rate = $finish / $runs;
        $break_rate = 1 - ($remain / $totalStageBricks);
        $weight = $brick_count / $total_bricks;

        $data['weighted_pass_total'] += $pass_rate * $weight * 100;
        $data['weighted_break_total'] += $break_rate * $weight * 100;
    }
}
unset($data);

// 整理 summary 結果
$summary = [];
foreach ($stats as $model => $data) {
    if ($data['total_runs'] === 0 || $data['total_bricks'] === 0) continue;

    $pass_rate = $data['FINISH'] / $data['total_runs'];
    $break_rate = 1 - ($data['total_remain'] / $data['total_bricks']);
    $weighted_score = ($data['weighted_pass_total'] + $data['weighted_break_total']) / 2;

    $summary[] = [
        'model' => $model,
        'pass_rate' => round($pass_rate * 100, 1),
        'break_rate' => round($break_rate * 100, 1),
        'passtimes' => $data['FINISH'],
        'total_runs' => $data['total_runs'],
        'weighted_score' => round($weighted_score, 1)
    ];
}

// 排序
$threshold = 20;

usort($summary, function ($a, $b) use ($filter_stage, $threshold) {
    // 依照門檻分組
    $a_group = ($a['total_runs'] >= $threshold) ? 0 : 1;
    $b_group = ($b['total_runs'] >= $threshold) ? 0 : 1;

    if ($a_group !== $b_group) {
        return $a_group - $b_group; // 優先顯示達門檻者
    }

    // all：用加權總分排序
    if ($filter_stage === 'all') {
        return $b['weighted_score'] <=> $a['weighted_score'];
    }

    // 單關卡：用擊破率排序
    return $b['break_rate'] <=> $a['break_rate'];
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
], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
