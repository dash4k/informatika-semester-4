<?php

$host = '127.0.0.1';
$user = 'johntor';
$pass = 'katasandi';
$dbname = 'user_warja';

$conn = new mysqli($host, $user, $pass, $dbname);

if ($conn->connect_error) {
    die("Failed to establish connection to the database: " . $conn->connect_error);
}

?>