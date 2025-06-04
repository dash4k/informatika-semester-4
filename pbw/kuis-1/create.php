<?php
include 'config.php';

$user_id = $_POST['user_id'];
$email = $_POST['email'];
$password = password_hash($_POST['password'], PASSWORD_DEFAULT);
$name = $_POST['name'];
$role = $_POST['role'];
$profile = isset($_FILES['profile']) ? file_get_contents($_FILES['profile']['tmp_name']) : null;

$stmt = $conn->prepare("INSERT INTO users (user_id, email, password, name, role, profile) VALUES (?, ?, ?, ?, ?, ?)");
$stmt->bind_param("ssssss", $user_id, $email, $password, $name, $role, $profile);
$stmt->send_long_data(5, $profile);
$stmt->execute();

echo json_encode(['status' => 'success']);