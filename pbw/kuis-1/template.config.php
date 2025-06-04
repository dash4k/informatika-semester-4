<?php

$host = '';
$user = '';
$pass = '';
$dbname = 'user_warja';

$conn = new mysqli($host, $user, $pass, $dbname);

if ($conn->connect_error) {
    die("Failed to establish connection to the database: " . $conn->connect_error);
}

?>