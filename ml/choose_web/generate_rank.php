<?php
// 計算後輸出 CSV，不顯示表格
$stages_brick = [1 => 5, 2 => 16, 3 => 27, 4 => 32, 5 => 43];
$total_bricks = array_sum($stages_brick);

$modelFile = __DIR__ . '/../../result/Model_name.csv';
$resultFile = __DIR__ . '/../../result/result.csv';
$RANKFile = __DIR__ . '/../../result/RANKresult.csv';

$modelRaw = array_map('str_getcsv', file($modelFile));
array_shift($modelRaw);
$modelNames = array_column($modelRaw, 0);

$resultRaw = array_map('str_getcsv', file($resultFile));
array_shift($resultRaw);

$combined = [];
foreach ($resultRaw as $index => $result) {
    if (!isset($modelNames[$index]) || count($result) < 4) continue;
    $combined[] = array_merge([$modelNames[$index]], $result);
}

$fp = fopen($RANKFile, 'w');
fputcsv($fp, ['modelname', 'stage', 'brick_remain', 'catch_count', 'state', 'run_sec']);
foreach ($combined as $row) {
    fputcsv($fp, $row);
}
fclose($fp);


?>
