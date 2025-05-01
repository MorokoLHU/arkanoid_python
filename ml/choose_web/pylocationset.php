<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $userPath = trim($_POST["Pylocation"]);
    if (!empty($userPath)) {
        file_put_contents("user_pypath.txt", $userPath);
        echo "路徑已儲存：$userPath";
    } else {
        echo "請輸入有效的路徑";
    }
}
?>