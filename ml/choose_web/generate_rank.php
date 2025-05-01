<?php
// 假設這是預設的磚塊數量資料
$stages_brick = array(
    1 => 5,
    2 => 16,
    3 => 27,
    4 => 32,
    5 => 43
);
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
foreach ($results as $index => $result) {
    if (!isset($modelNames[$index]) || count($result) < 4) continue;
    $combined[] = array_merge([$modelNames[$index]], $result);
}
$RANKFile = __DIR__ . '/../../result/RANKresult.csv';
// 建立新的 RANKresult.csv
$fp = fopen($RANKFile, 'w');
foreach ($combined as $row) {
    fputcsv($fp, $row);
}
fclose($fp);

// 統計資料
$stats = [];
$passRates = [];   // 新增陣列
$breakRates = [];  // 新增陣列
$totalRuns = [];   // 新增陣列
foreach ($combined as $row) {
    list($model, $stage, $brick_remain, $catch_count, $state) = $row;

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

    if ($state !== 'FINISH' && $state !== 'FAIL') continue; // 忽略非預期狀態

    $stats[$model]['total_runs']++;
    $stats[$model][$state]++;
    $stats[$model]['total_catch'] += $catch_count;
    $stats[$model]['total_remain'] += $brick_remain;
    $stats[$model]['total_bricks'] += $stages_brick[$stage] ?? 0;

}

// 計算比率與排序
foreach ($stats as $model => &$data) {
    if ($data['total_runs'] > 0 && $data['total_bricks'] > 0) {
        $data['pass_rate'] = $data['FINISH'] / $data['total_runs'];
        $data['break_rate'] = 1 - ($data['total_remain'] / $data['total_bricks']);
        $data['break_bricks'] = ($data['total_bricks'] - $data['total_remain']);
    } else {
        $data['pass_rate'] = 0;
        $data['break_rate'] = 0;
    }

}
unset($data);

// 以通關率排序
uasort($stats, fn($a, $b) => $b['pass_rate'] <=> $a['pass_rate']);

// 顯示表格
echo '<table border="1">';
echo '<tr><th>排名</th><th>模型</th><th>執行次數</th><th>總通關次數</th><th>接到球的次數</th><th>擊破磚塊數</th><th>磚塊擊破率</th><th>整體通關率</th></tr>';
$rank = 1;
foreach ($stats as $model => $data) {
    echo "<tr>
        <td>{$rank}</td>
        <td>{$model}</td>
        <td>{$data['total_runs']}</td>
        <td>{$data['FINISH']}</td>
        <td>{$data['total_catch']}</td>
        <td>{$data['break_bricks']}</td>
        <td>" . round($data['break_rate'] * 100, 1) . "%</td>
        <td>" . round($data['pass_rate'] * 100, 0) . "%</td>
    </tr>";
    $rank++;
}
echo '</table>';

?>
