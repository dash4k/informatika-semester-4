<?php
include 'config.php';

$data = json_decode(file_get_contents("php://input"), true);
$user_id = $data['id'];

$stmt = $conn->prepare("DELETE FROM users WHERE user_id = ?");
$stmt->bind_param("i", $user_id);
$stmt->execute();

echo json_encode(['status' => 'deleted']);