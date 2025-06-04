<?php
include 'config.php';

$user_id = $_POST['user_id'];
$email = $_POST['email'];
$password = password_hash($_POST['password'], PASSWORD_DEFAULT);
$name = $_POST['name'];
$role = $_POST['role'];

if (isset($_FILES['profile']) && $_FILES['profile']['size'] > 0) {
    $profile = file_get_contents($_FILES['profile']['tmp_name']);
    $stmt = $conn->prepare("UPDATE users SET email=?, password=?, name=?, role=?, profile=? WHERE user_id=?");
    $stmt->bind_param("sssssi", $email, $password, $name, $role, $profile, $user_id);
    $stmt->send_long_data(4, $profile);
} else {
    $stmt = $conn->prepare("UPDATE users SET email=?, password=?, name=?, role=? WHERE user_id=?");
    $stmt->bind_param("ssssi", $email, $password, $name, $role, $user_id);
}
$stmt->execute();

echo json_encode(['status' => 'updated']);