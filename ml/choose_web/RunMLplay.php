<?php

$Pylocation = trim(file_get_contents("user_pypath.txt"));

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $times = isset($_POST["times"]) ? intval($_POST["times"]) : 0;
    $stage = isset($_POST["selectedStage"]) ? intval($_POST["selectedStage"]) : 0;
    $fileInputValue = isset($_POST["fileInputTexthold"]) ? $_POST["fileInputTexthold"] : "";

    // 對應關卡的總磚塊數
    $stages_brick = array(
        1 => 5,
        2 => 16,
        3 => 27,
        4 => 32,
        5 => 43
    );

    if ($times > 0 && $stage > 0) {
        echo "<h3>✅ 模型執行資訊</h3>";
        echo "執行次數：{$times}<br>";
        echo "選擇的關卡：{$stage}<br>";
        echo "標記模型名稱：<strong>" . htmlspecialchars($fileInputValue) . "</strong><br><br>";

        $csvPath = __DIR__ . "\\..\\..\\result\\Nowresult.csv";
        if (file_exists($csvPath)) {
            unlink($csvPath); // 執行前先刪除舊結果
        }

        // 組成執行指令
        $command = "\"{$Pylocation}\" -m mlgame -f 100 -1 -i ../MLplay.py ../../ --difficulty NORMAL --level {$stage}";
        for ($i = 1; $i <= $times; $i++) {
            echo "⚙️ 執行第 {$i} 次...<br>";
            shell_exec($command); // 不顯示輸出內容
        }

        // 顯示結果 CSV 表格
        if (file_exists($csvPath)) {
            echo "<h3>📊 結果 CSV</h3>";
            echo "<table border='1' cellpadding='5'>";
            if (($handle = fopen($csvPath, "r")) !== false) {
                $header = fgetcsv($handle); // 讀取標題列
                echo "<tr>
                        <th>檔案標記</th>
                        <th>" . implode("</th><th>", array_map('htmlspecialchars', $header)) . "</th>
                        <th>擊破率 (%)</th>
                        <th>平均每打碎1個磚塊需接幾顆球</th>
                    </tr>";

                $successCount = 0;
                $totalCount = 0;

                while (($row = fgetcsv($handle)) !== false) {
                    $remain_brick = isset($row[1]) ? intval($row[1]) : 0; // 剩下磚塊數
                    $catch_ball = isset($row[2]) ? intval($row[2]) : 0;   // 接到球次數
                    $result = isset($row[3]) ? trim($row[3]) : "";

                    $total_brick = $stages_brick[$stage];
                    $broken_brick = $total_brick - $remain_brick;
                    $hit_rate = $total_brick > 0 ? round(($broken_brick / $total_brick) * 100, 2) : 0;
                    $avg_ball_per_brick = ($broken_brick > 0) ? round($catch_ball / $broken_brick, 2) : '∞';

                    if (strtoupper($result) === "FINISH") $successCount++;
                    $totalCount++;

                    echo "<tr><td>" . htmlspecialchars($fileInputValue) . "</td>";
                    foreach ($row as $cell) {
                        echo "<td>" . htmlspecialchars($cell) . "</td>";
                    }
                    echo "<td>{$hit_rate}</td><td>{$avg_ball_per_brick}</td></tr>";
                }

                fclose($handle);
                $pass_rate = ($totalCount > 0) ? round(($successCount / $totalCount) * 100, 2) : 0;
                echo "</table>";
                echo "<p>✅ 通過率（FINISH 次數 / 執行次數）為：<strong>{$pass_rate}%</strong></p>";
            }
        } else {
            echo "<p>❌ 找不到結果檔案：<code>{$csvPath}</code></p>";
        }
    } else {
        echo "<p>❌ 參數錯誤，請確認輸入是否完整。</p>";
    }
} else {
    echo "<p>請使用 POST 方法呼叫此程式。</p>";
}
?>
