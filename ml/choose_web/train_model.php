<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // 取得使用者選擇的模型與任務類型
    $model = $_POST["model"];
    $task_type = $_POST["task_type"];

    // Python 腳本對應表
    $python_scripts = [
        "DecisionTree" => [
            "classification" => "../model_train_Decisiontree_classification.py",
            "regression" => "../model_train_Decisiontree_regression.py"
        ],
        "KNeighbors" => [
            "classification" => "../model_train_KNN_classification.py",
            "regression" => "../model_train_KNN_regression.py"
        ],
        "RandomForest" => [
            "classification" => "../model_train_RandomForest_classification.py",
            "regression" => "../model_train_RandomForest_regression.py"
        ],
        "LinearSVM" => [
            "classification" => "../model_train_linearSVC_classification.py",
            "regression" => "../model_train_linearSVR_regression.py"
        ],
        "LogisticRegression" => [
            "classification" => "../model_train_LogisticRegression_classification.py",
            "regression" => null
        ],
        "LinearRegression" => [
            "classification" => null,
            "regression" => "../model_train_Linear_regression.py"
        ],
        "SVM" => [
            "classification" => "../model_train_SVM_classification.py",
            "regression" => "../model_train_SVM_regression.py"
        ]
    ];

    // 確認是否有指定的腳本
    if (!isset($python_scripts[$model][$task_type]) || !$python_scripts[$model][$task_type]) {
        die("<h3>錯誤：未支援的模型或任務類型！</h3>");
    }

    $script_path = $python_scripts[$model][$task_type];

    // 檢查腳本是否存在
    if (!file_exists($script_path)) {
        die("<h3>錯誤：找不到 Python 腳本！</h3>");
    }

    // 收集參數
    $params = [];

    switch ($model) {
        case "KNeighbors":
            $params[] = escapeshellarg($_POST["k_value"]);
            break;
        case "DecisionTree":
            $params[] = escapeshellarg($_POST["max_depth"]);
            break;
        case "RandomForest":
            $params[] = escapeshellarg($_POST["max_depth"]);
            $params[] = escapeshellarg($_POST["random_state"]);
            $params[] = escapeshellarg($_POST["n_estimators"]);
            break;
        case "LinearSVM":
            if ($task_type === "classification") {
                $params[] = escapeshellarg($_POST["svm_C"]);
            } elseif ($task_type === "regression") {
                $params[] = escapeshellarg($_POST["svm_C"]);
                $params[] = escapeshellarg($_POST["svm_epsilon"]);
            }
            break;
        case "LogisticRegression":
            $params[] = escapeshellarg($_POST["logistic_C"]);
            break;
        case "LinearRegression":
            $params[] = escapeshellarg($_POST["linear_fit_intercept"]);
            break;
        case "SVM":
            $params[] = escapeshellarg($_POST["svm_kernel"]);
            $params[] = escapeshellarg($_POST["svm_C"]);
            break;
    }

    // 讀取 Python 執行路徑
    $Pylocation = trim(file_get_contents("user_pypath.txt"));

    // 組成指令
    $command = "\"{$Pylocation}\" " . escapeshellarg($script_path) . " " . implode(" ", $params);
    $output = shell_exec($command);

    echo "<h3>訓練結果：</h3>";
    echo "<p>$output</p>";
    echo "<br><p>🚩點擊Run Model去試試吧!<br><img id=\"happyturn\"class=\"happyturn\" src=\"image/Happy.png\"></p>";
}
?>
