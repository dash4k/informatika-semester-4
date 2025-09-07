<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $namaLengkap = htmlspecialchars($_POST['namaLengkap']);
    $nama = htmlspecialchars($_POST['nama']);
    $nim = htmlspecialchars($_POST['nim']);
    $email = htmlspecialchars($_POST['email']);
    $alamat = htmlspecialchars($_POST['alamat']);
    $tanggalLahir = htmlspecialchars($_POST['tanggalLahir']);
    $tempatLahir = htmlspecialchars($_POST['tempatLahir']);
    $gender = htmlspecialchars($_POST['gender']);

    // Handle file upload
    $uploadDir = "uploads/";
    $uploadFile = $uploadDir . basename($_FILES["pasFoto"]["name"]);
    $uploadOk = 1;
    $imageFileType = strtolower(pathinfo($uploadFile, PATHINFO_EXTENSION));

    if (isset($_FILES["pasFoto"]) && $_FILES["pasFoto"]["error"] == 0) {
        // Check if file is an image
        $check = getimagesize($_FILES["pasFoto"]["tmp_name"]);
        if ($check !== false) {
            if (!is_dir($uploadDir)) {
                mkdir($uploadDir, 0777, true);
            }

            if (move_uploaded_file($_FILES["pasFoto"]["tmp_name"], $uploadFile)) {
                $uploadStatus = "File uploaded successfully.";
            } else {
                $uploadStatus = "Sorry, there was an error uploading your file.";
            }
        } else {
            $uploadStatus = "File is not an image.";
            $uploadOk = 0;
        }
    } else {
        $uploadStatus = "No file uploaded or upload error.";
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Data Diri</title>
    <link rel="stylesheet" href="main-style.css">
</head>
<body>
    <h1>Data Diri Mahasiswa</h1>
    <p><strong>Nama Lengkap:</strong> <?= $namaLengkap ?></p>
    <p><strong>Nama Panggilan:</strong> <?= $nama ?></p>
    <p><strong>NIM:</strong> <?= $nim ?></p>
    <p><strong>Email:</strong> <?= $email ?></p>
    <p><strong>Alamat:</strong> <?= nl2br($alamat) ?></p>
    <p><strong>Tanggal Lahir:</strong> <?= $tanggalLahir ?></p>
    <p><strong>Tempat Lahir:</strong> <?= $tempatLahir ?></p>
    <p><strong>Jenis Kelamin:</strong> <?= $gender ?></p>
    <p><strong>Status Upload Foto:</strong> <?= $uploadStatus ?></p>
    <?php if ($uploadOk && file_exists($uploadFile)): ?>
        <img src="<?= $uploadFile ?>" alt="Foto Profil" style="max-width:200px;">
    <?php endif; ?>
</body>
</html>
