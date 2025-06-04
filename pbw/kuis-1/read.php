<?php
include 'config.php';

$result = $conn->query("SELECT * FROM users");
$users = [];

while ($row = $result->fetch_assoc()) {
    if ($row['profile']) {
        $row['profile'] = base64_encode($row['profile']);
    }
    $users[] = $row;
}

header('Content-Type: application/json');
echo json_encode($users);