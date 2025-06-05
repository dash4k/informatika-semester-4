<?php
include 'config.php';

try {
    $data = json_decode(file_get_contents("php://input"), true);
    $user_id = $data['id'];
    
    $stmt = $conn->prepare("DELETE FROM users WHERE user_id = ?");
    $stmt->bind_param("i", $user_id);
    $stmt->execute();
    
    echo json_encode(['status' => 'deleted']);
} catch (mysqli_sql_exception $e) {
  http_response_code(500);
  echo json_encode([
    "status" => "error",
    "message" => $e->getMessage()
  ]);
}

?>